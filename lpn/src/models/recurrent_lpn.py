# --- START OF FILE recurrent_lpn.py ---

from typing import Literal, Optional
import math
from functools import partial
import os

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.tree_util import tree_map
import optax
import numpy as np
from src.models.transformer import EncoderTransformer, DecoderTransformer
from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig
from src.data_utils import make_leave_one_out

# Define DecoderStep globally for use in @nn.compact methods
class DecoderStep(nn.Module):
    decoder_to_use: DecoderTransformer
    fixed_dropout_eval: bool

    @nn.compact
    def __call__(self, carry, scan_input):
        target_seq_prev_step, fixed_input_seq, fixed_context_col = carry
        current_token_idx_to_predict = scan_input
        _, _, grid_logits = self.decoder_to_use(
            fixed_input_seq, target_seq_prev_step, fixed_context_col, self.fixed_dropout_eval
        )
        current_token_logits = grid_logits[..., current_token_idx_to_predict, :]
        predicted_token = jnp.argmax(current_token_logits, axis=-1).astype(jnp.int32)
        target_seq_updated = target_seq_prev_step.at[..., 2 + current_token_idx_to_predict].set(predicted_token)
        new_carry = (target_seq_updated, fixed_input_seq, fixed_context_col)
        return new_carry, None

class LPN(nn.Module):
    encoder: EncoderTransformer
    decoder: DecoderTransformer

    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
        matrix_size_rows: jnp.int32, # Added for recurrent GA and matrix mode
        matrix_size_cols: jnp.int32, # Added for recurrent GA and matrix mode
        mode: Literal["direct_sample", "mean_sample", "cross_attention_sample", "matrix", "recurrent_ga", "random_search"], # Added random_search
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        program_kl_coeff: Optional[float] = None,
        **mode_kwargs,
    ):
        assert pairs.shape[-4] > 1, f"Number of pairs > 1 required, got {pairs.shape[-4]}."
        latents_mu_all, latents_logvar_all = self.encoder(pairs, grid_shapes, dropout_eval)

        kl_metrics = {}
        original_prior_kl_loss = None
        original_pairwise_kl_loss = None
        combined_program_kl_loss = None

        if latents_logvar_all is None:
            raise ValueError("VAE (latents_logvar) is required for this architecture.")

        original_prior_kl_loss = jnp.mean(
            -0.5 * jnp.sum(1 + latents_logvar_all - latents_mu_all**2 - jnp.exp(latents_logvar_all), axis=-1)
        )
        kl_metrics["original_prior_kl"] = original_prior_kl_loss
        original_pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu_all, latents_logvar_all).mean()
        kl_metrics["original_pairwise_kl"] = original_pairwise_kl_loss
        kl_metrics["latents_mu_all_mean"] = latents_mu_all.mean()
        kl_metrics["latents_logvar_all_mean"] = latents_logvar_all.mean()

        leave_one_out_mu_all = make_leave_one_out(latents_mu_all, axis=-2)
        leave_one_out_logvar_all = make_leave_one_out(latents_logvar_all, axis=-2)

        program_mu_loo, program_logvar_loo = self._compute_attention_params_context(
            target_mu=latents_mu_all,
            loo_mu=leave_one_out_mu_all,
            loo_logvar=leave_one_out_logvar_all
        )

        if program_kl_coeff is not None:
            combined_program_kl_loss = jnp.mean(
                -0.5 * jnp.sum(1 + program_logvar_loo - program_mu_loo**2 - jnp.exp(program_logvar_loo), axis=-1)
            )
            kl_metrics["combined_program_kl"] = combined_program_kl_loss
        kl_metrics["program_mu_loo_mean"] = program_mu_loo.mean()
        kl_metrics["program_logvar_loo_mean"] = program_logvar_loo.mean()

        key_sample_effective_context = self.make_rng("effective_context_sample")
        program_std_loo = jnp.exp(0.5 * program_logvar_loo)
        initial_effective_contexts = program_mu_loo + program_std_loo * jax.random.normal(
            key_sample_effective_context, program_mu_loo.shape
        ) # (*B, N, H)

        effective_contexts = None

        if mode == "direct_sample":
            effective_contexts = initial_effective_contexts
        elif mode == "mean_sample":
            effective_contexts = program_mu_loo
        elif mode == "cross_attention_sample":
            loo_program_samples = make_leave_one_out(initial_effective_contexts, axis=-2)
            effective_contexts = self._compute_loo_cross_attention_context(loo_program_samples)
        elif mode == "matrix":
            effective_contexts = initial_effective_contexts
        elif mode == "recurrent_ga":
            for arg in ["num_steps", "lr"]: # GA specific args
                assert arg in mode_kwargs, f"'{arg}' argument required for 'recurrent_ga' mode."
            key_ga = self.make_rng("recurrent_ga_optim")
            
            # For __call__, GA optimizes each of the N contexts independently
            # using its corresponding single pair.
            # vmap _optimize_context_recurrently over the N dimension.
            # _optimize_context_recurrently will take:
            #   initial_context_for_one_pair (*B,H)
            #   single_pair_data (*B, R,C,2)
            #   single_grid_shape_data (*B, 2,2)
            #   other_ga_params

            # To vmap over pairs and initial_contexts:
            # initial_effective_contexts: (*B, N, H)
            # pairs: (*B, N, R, C, 2)
            # grid_shapes: (*B, N, 2, 2)
            # We want to call optimize for each of the N items.
            # So, vmap over axis 1 (the N dimension) for these inputs.

            partial_optimize_fn = partial(self._optimize_context_recurrently_for_loss,
                                          matrix_size_rows=matrix_size_rows,
                                          matrix_size_cols=matrix_size_cols,
                                          dropout_eval=dropout_eval, # Fixed for GA opt
                                          **mode_kwargs)
            
            # JAX's vmap maps over the leading axis by default.
            # To map over axis 1 (N dim), we can transpose, vmap, then transpose back,
            # or write the vmapped function more carefully.
            # Let's try to make inputs suitable for direct vmap over N.
            # If initial_effective_contexts is (B, N, H), pairs is (B, N, ...),
            # we want to map (B, H)_i with (B, ...)_i.
            # This means the vmapped function should expect inputs without the N dim.

            # Transpose B and N for easier vmapping over N
            # (N, B, H), (N, B, R,C,2), (N, B, 2,2)
            transposed_initial_contexts = jnp.moveaxis(initial_effective_contexts, -2, 0)
            transposed_pairs = jnp.moveaxis(pairs, -2, 0)
            transposed_grid_shapes = jnp.moveaxis(grid_shapes, -2, 0)
            
            # If key_ga needs to be different per optimization, split it N times.
            # For now, assume same base key or handle inside optimize if needed.
            # key_ga_n = jax.random.split(key_ga, pairs.shape[-4]) # (N, *B_keyshape)

            # vmap over the first dimension (originally N)
            # The function `partial_optimize_fn` expects (*B,H), (*B,R,C,2), (*B,2,2)
            vmapped_optimized_contexts = jax.vmap(
                partial_optimize_fn, 
                in_axes=(0, 0, 0, None) # Map over 0th axis of transposed inputs, key is None (broadcast)
            )(transposed_initial_contexts, transposed_pairs, transposed_grid_shapes, key_ga)
            
            # Transpose back: (B, N, H)
            effective_contexts = jnp.moveaxis(vmapped_optimized_contexts, 0, -2)

        elif mode == "random_search":
            # In __call__, random search optimizes a context for *each* of the N examples,
            # using the *other N-1 examples* as support for evaluation.
            # This matches the original LPN's random search during training.
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' training mode."
            key_rs = self.make_rng("random_search_call")
            
            # leave_one_out_latents (here, use initial_effective_contexts as base for search)
            # The search will be for *each* of the N examples.
            # The "latents" for _get_random_search_context_original are the N-1 other effective contexts.
            
            # This is tricky. The original _get_random_search_context took leave_one_out_latents (*B,N,N-1,H)
            # and leave_one_out_pairs (*B,N,N-1,...).
            # We need to decide what "latents" _get_random_search_context_original operates on.
            # Option 1: It operates on samples from the N-1 other `program_mu_loo`.
            # Option 2: It operates on the `initial_effective_contexts` that were already sampled.
            
            # Let's use `initial_effective_contexts` as the base.
            loo_initial_effective_contexts = make_leave_one_out(initial_effective_contexts, axis=-2) # (*B,N,N-1,H)
            loo_pairs = make_leave_one_out(pairs, axis=-4) # (*B,N,N-1,...)
            loo_grid_shapes = make_leave_one_out(grid_shapes, axis=-3) # (*B,N,N-1,...)

            # _get_random_search_context_original expects latents (*B, num_search_bases, H)
            # and pairs/grid_shapes (*B, num_eval_pairs, ...)
            # Here, for each of the N examples, num_search_bases are N-1, num_eval_pairs are N-1.
            # We need to vmap this process.
            
            # Vmap over the N dimension
            # Transpose B and N
            # (N, B, N-1, H), (N, B, N-1,...), (N, B, N-1,...)
            transposed_loo_initial_contexts = jnp.moveaxis(loo_initial_effective_contexts, 1, 0)
            transposed_loo_pairs = jnp.moveaxis(loo_pairs, 1, 0)
            transposed_loo_grid_shapes = jnp.moveaxis(loo_grid_shapes, 1, 0)
            keys_rs_n = jax.random.split(key_rs, pairs.shape[1]) # N keys

            # The _get_random_search_context_original works on (B, N_search_base, H) and (B, N_eval, ...)
            # Here, for each element of the vmap, N_search_base=N-1, N_eval=N-1.
            vmapped_rs_contexts, _ = jax.vmap( # Second output is second_best, ignore for now
                partial(self._get_random_search_context_original, # Use the old name
                        matrix_size_rows=matrix_size_rows, # Pass these if needed by its loss
                        matrix_size_cols=matrix_size_cols,
                        dropout_eval=dropout_eval,
                        **mode_kwargs),
                in_axes=(0, 0, 0, 0) # vmap over N for all inputs
            )(transposed_loo_initial_contexts, transposed_loo_pairs, transposed_loo_grid_shapes, keys_rs_n)

            effective_contexts = jnp.moveaxis(vmapped_rs_contexts, 0, 1) # (B,N,H)


        else:
            raise ValueError(f"Unsupported mode for __call__: {mode}")

        loss, metrics = self._loss_from_pair_and_context(
            effective_contexts, pairs, grid_shapes, dropout_eval,
            matrix_size_rows, matrix_size_cols
        )

        if latents_logvar_all is not None:
            key_sample_orig_latents = self.make_rng("latents_sample_orig_for_metrics")
            latents_std_all = jnp.exp(0.5 * latents_logvar_all)
            original_sampled_latents = latents_mu_all + latents_std_all * jax.random.normal(
                key_sample_orig_latents, latents_mu_all.shape
            )
        else:
            original_sampled_latents = latents_mu_all

        metrics.update(
            original_sampled_latents_norm=norm(original_sampled_latents, axis=-1),
            effective_contexts_norm=norm(effective_contexts, axis=-1),
            distance_effective_context_vs_original_sample=norm(effective_contexts - original_sampled_latents, axis=-1),
        )
        
        loss, metrics = tree_map(jnp.mean, (loss, metrics))
        metrics.update(kl_metrics)

        total_loss = loss
        if original_prior_kl_loss is not None and prior_kl_coeff is not None:
            total_loss += prior_kl_coeff * original_prior_kl_loss
        if original_pairwise_kl_loss is not None and pairwise_kl_coeff is not None:
            total_loss += pairwise_kl_coeff * original_pairwise_kl_loss
        if combined_program_kl_loss is not None and program_kl_coeff is not None:
             total_loss += program_kl_coeff * combined_program_kl_loss

        metrics["total_loss_unweighted_reconstruction"] = loss
        metrics["total_loss_final"] = total_loss
        return total_loss, metrics

    @staticmethod
    def _compute_pairwise_gaussian_kl(mu: chex.Array, log_var: chex.Array, eps: float = 1e-7) -> chex.Array:
        # ... (implementation remains the same) ...
        mu1 = mu[..., :, None, :]
        mu2 = mu[..., None, :, :]
        log_var1 = log_var[..., :, None, :]
        log_var2 = log_var[..., None, :, :]
        var1, var2 = jnp.exp(log_var1), jnp.exp(log_var2)
        log_var_ratio = log_var2 - log_var1
        var_ratio = var1 / (var2 + eps)
        mu_diff_sq = (mu1 - mu2) ** 2 / (var2 + eps)
        kl = jnp.sum(0.5 * (log_var_ratio + var_ratio + mu_diff_sq - 1), axis=-1)
        num_pairs = mu.shape[-2]
        # Mask diagonal and average (ensure division by non-zero for N=1 case, although prevented by assertion)
        denom = max(1, num_pairs * (num_pairs - 1))
        kl = jnp.sum(jnp.where(jnp.eye(num_pairs) == 0, kl, 0), axis=(-1, -2)) / denom
        return kl

    def _convert_to_matrix(
        self,
        matrix_size_rows: jnp.int32,
        matrix_size_cols: jnp.int32,
        latents: chex.Array):
        # ... (implementation remains the same) ...
        batch_shape = latents.shape[:-1]
        latent_dim = latents.shape[-1]
        expected_dim = matrix_size_rows * matrix_size_cols
        # Ensure latent_dim matches matrix dimensions product if possible, or handle mismatch
        if latent_dim != expected_dim:
             # Option 1: Pad or truncate (Example: Padding)
             # padding_amount = expected_dim - latent_dim
             # if padding_amount > 0:
             #      pad_width = [(0, 0)] * (latents.ndim - 1) + [(0, padding_amount)]
             #      latents = jnp.pad(latents, pad_width, mode='constant')
             # elif padding_amount < 0:
             #      latents = latents[..., :expected_dim]
             # Option 2: Raise error
             raise ValueError(f"Latent dimension {latent_dim} does not match matrix size {matrix_size_rows}*{matrix_size_cols}={expected_dim}")

        static_shape = (*batch_shape, matrix_size_rows, matrix_size_cols)
        latents_reshaped = latents.reshape(static_shape)
        return latents_reshaped

    def _normalized_mean_over_sequence(
        self, grid_seq: chex.Array, num_rows: chex.Array, num_cols: chex.Array
    ) -> chex.Array:
        # ... (implementation remains the same) ...
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        row_arange_broadcast = jnp.arange(max_rows).reshape(*len(num_rows.shape) * (1,), max_rows)
        col_arange_broadcast = jnp.arange(max_cols).reshape(*len(num_cols.shape) * (1,), max_cols)
        grid_row_mask = row_arange_broadcast < num_rows[..., None]
        grid_col_mask = col_arange_broadcast < num_cols[..., None]
        grid_pad_mask = grid_row_mask[..., None] & grid_col_mask[..., None, :]
        grid_pad_mask = grid_pad_mask.reshape(*grid_pad_mask.shape[:-2], -1)
        grid_seq = jnp.where(grid_pad_mask, grid_seq, 0)
        mean_seq = jnp.sum(grid_seq, axis=-1) / (jnp.sum(grid_pad_mask, axis=-1) + 1e-5)
        return mean_seq

        # --- Attention Helper Functions ---

    def _compute_loo_cross_attention_context(self, loo_latents: chex.Array) -> chex.Array:
        """ Computes context using attention on Leave-One-Out sampled latents.
            Query: Mean of N-1 latents. Key/Value: The N-1 latents.
            Input: loo_latents shape (*B, N, N-1, H)
            Output: context shape (*B, N, H)
        """
        batch_dims = loo_latents.shape[:-3]
        N = loo_latents.shape[-3]
        N_minus_1 = loo_latents.shape[-2]
        H = loo_latents.shape[-1]
        sqrt_dh = jnp.sqrt(float(H))

        # Query: Mean of the N-1 latents for each target i. Shape (*B, N, 1, H)
        query = loo_latents.mean(axis=-2, keepdims=True)

        # Key/Value are the loo_latents themselves. Shape (*B, N, N-1, H)
        key = loo_latents
        value = loo_latents

        # Attention scores. Shape (*B, N, 1, N-1)
        attn_scores = jnp.einsum('...qh,...kh->...qk', query, key) / sqrt_dh

        # Attention weights. Shape (*B, N, 1, N-1)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Weighted sum (attended context). Shape (*B, N, 1, H)
        attended_context = jnp.einsum('...qk,...kh->...qh', attn_weights, value)

        # Remove the query dimension. Shape (*B, N, H)
        return attended_context.squeeze(axis=-2)

    def _compute_global_attention_context(self, latents: chex.Array) -> chex.Array:
        """ Computes a single context vector using attention over all N sampled latents.
            Used for Generation mode 'cross_attention'.
            Query: Mean of N latents. Key/Value: The N latents.
            Input: latents shape (*B, N, H)
            Output: context shape (*B, H)
        """
        batch_dims = latents.shape[:-2]
        N = latents.shape[-2]
        H = latents.shape[-1]
        sqrt_dh = jnp.sqrt(float(H))

        # Query: Mean of all N latents. Shape (*B, 1, H)
        query = latents.mean(axis=-2, keepdims=True)

        # Key/Value are the latents themselves. Shape (*B, N, H)
        key = latents
        value = latents

        # Attention scores. Shape (*B, 1, N)
        attn_scores = jnp.einsum('...qh,...kh->...qk', query, key) / sqrt_dh

        # Attention weights. Shape (*B, 1, N)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Weighted sum (attended context). Shape (*B, 1, H)
        attended_context = jnp.einsum('...qk,...kh->...qh', attn_weights, value)

        # Remove the query dimension. Shape (*B, H)
        return attended_context.squeeze(axis=-2)


    def _compute_attention_params_context(self,
                                            target_mu: chex.Array,
                                            loo_mu: chex.Array,
                                            loo_logvar: chex.Array
                                            ) -> tuple[chex.Array, chex.Array]:
        """ Combines latent parameters using Leave-One-Out attention.
            Query: target_mu. Key: loo_mu. Value: loo_mu and loo_logvar.
            Used for Training mode 'attention_params'.
            Inputs: target_mu (*B, N, H), loo_mu (*B, N, N-1, H), loo_logvar (*B, N, N-1, H)
            Outputs: context_mu (*B, N, H), context_logvar (*B, N, H)
        """
        batch_dims = target_mu.shape[:-2]
        N = target_mu.shape[-2]
        # N_minus_1 = loo_mu.shape[-2] # Should be N-1
        H = target_mu.shape[-1]
        sqrt_dh = jnp.sqrt(float(H))

        # Query: The mu of the target example 'i'. Shape (*B, N, 1, H)
        query = target_mu[..., None, :]

        # Key: The mus of the other N-1 examples. Shape (*B, N, N-1, H)
        key = loo_mu

        # Values: The mus and logvars of the other N-1 examples. Shapes (*B, N, N-1, H)
        value_mu = loo_mu
        value_logvar = loo_logvar

        # Attention scores. Shape (*B, N, 1, N-1)
        attn_scores = jnp.einsum('...qh,...kh->...qk', query, key) / sqrt_dh

        # Attention weights. Shape (*B, N, 1, N-1)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Weighted sum for mu. Shape (*B, N, 1, H)
        context_mu = jnp.einsum('...qk,...kh->...qh', attn_weights, value_mu)
        # Weighted sum for logvar. Shape (*B, N, 1, H)
        # Note: Weighting logvars is a heuristic, not strictly variance combination.
        context_logvar = jnp.einsum('...qk,...kh->...qh', attn_weights, value_logvar)

        # Remove the query dimension. Shapes (*B, N, H)
        return context_mu.squeeze(axis=-2), context_logvar.squeeze(axis=-2)

    def _compute_global_attention_params_context(self,
                                                 latents_mu: chex.Array,
                                                 latents_logvar: chex.Array
                                                 ) -> tuple[chex.Array, chex.Array]:
        """ Combines latent parameters using global attention over all N examples.
            Query: Mean of N mus. Key: The N mus. Value: N mus and N logvars.
            Used for Generation mode 'attention_params'.
            Inputs: latents_mu (*B, N, H), latents_logvar (*B, N, H)
            Outputs: context_mu (*B, H), context_logvar (*B, H)
        """
        batch_dims = latents_mu.shape[:-2]
        N = latents_mu.shape[-2]
        H = latents_mu.shape[-1]
        sqrt_dh = jnp.sqrt(float(H))

        # Query: Mean of all N mus. Shape (*B, 1, H)
        query = latents_mu.mean(axis=-2, keepdims=True)

        # Key: The N mus. Shape (*B, N, H)
        key = latents_mu

        # Values: The N mus and logvars. Shapes (*B, N, H)
        value_mu = latents_mu
        value_logvar = latents_logvar

        # Attention scores. Shape (*B, 1, N)
        attn_scores = jnp.einsum('...qh,...kh->...qk', query, key) / sqrt_dh

        # Attention weights. Shape (*B, 1, N)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Weighted sum for mu. Shape (*B, 1, H)
        context_mu = jnp.einsum('...qk,...kh->...qh', attn_weights, value_mu)
        # Weighted sum for logvar. Shape (*B, 1, H)
        context_logvar = jnp.einsum('...qk,...kh->...qh', attn_weights, value_logvar)

        # Remove the query dimension. Shapes (*B, H)
        return context_mu.squeeze(axis=-2), context_logvar.squeeze(axis=-2)

    @classmethod
    def _flatten_input_output_for_decoding(
        cls, pairs: chex.Array, grid_shapes: chex.Array
    ):
        # ... (implementation remains the same) ...
        flattened_pairs = jnp.reshape(pairs, (*pairs.shape[:-3], -1, 2))
        input_seq = jnp.concatenate([grid_shapes[..., 0,:], flattened_pairs[..., 0]], axis=-1) # Use input shape grid_shapes[..., 0]
        output_seq = jnp.concatenate([grid_shapes[..., 1,:], flattened_pairs[..., 1]], axis=-1)# Use output shape grid_shapes[..., 1]
        return input_seq, output_seq

    @classmethod
    def _select_best_and_second_best_latents(
        cls, log_probs: chex.Array, latents: chex.Array
    ): # log_probs: (*B,CANDS), latents: (*B,CANDS,H)
        # Argsort sorts in ascending order, so for log_probs, we want descending.
        # Or, sort negative log_probs (losses) in ascending.
        # If using log_probs, sort descending.
        sorted_indices = jnp.argsort(log_probs, axis=-1)[..., ::-1] # Descending sort

        best_idx = sorted_indices[..., 0:1] # (*B, 1)
        # Need to gather from latents along the CANDS dim (-2 for latents, -1 for log_probs)
        best_context = jnp.take_along_axis(
            latents, best_idx[..., None], axis=-2 # Add H dim for index
        ).squeeze(axis=-2) # (*B, H)

        if latents.shape[-2] > 1: # If more than one candidate
            second_best_idx = sorted_indices[..., 1:2] # (*B, 1)
            second_best_context = jnp.take_along_axis(
                latents, second_best_idx[..., None], axis=-2
            ).squeeze(axis=-2)
        else:
            second_best_context = best_context
        return best_context, second_best_context

    def _get_last_non_padded_logits(self, grid_logits: chex.Array, num_cols: chex.Array) -> chex.Array:
        """
        Selects the grid logits from the last non-padded column of each row.
        This is used to prepare logits for a specific loss calculation pattern where
        the end-of-row token's logit is effectively moved to the start-of-next-row
        position in a flattened sequence representation.

        Args:
            grid_logits: Grid logits, shape (*S, L, V) where L is max_rows * max_cols.
            num_cols: Number of columns for each grid, shape (*S, 1, 1).
                      Indicates the actual width of the content in each grid.

        Returns:
            Logits from the end of each actual row (0 to max_rows-2).
            Shape (*S, max_rows-1, V).
        """
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        # S = grid_logits.shape[:-2] # Leading dimensions (e.g., batch, num_pairs)
        # V = grid_logits.shape[-1]  # Vocab size

        # Ensure num_cols is int32 for indexing
        num_cols_int = num_cols.astype(jnp.int32)

        collected_logits = []
        # Iterate for rows 0 to max_rows-2 (max_rows-1 iterations)
        # The loop variable 'r_idx_plus_1' goes from 1 to max_rows-1 (inclusive).
        # This corresponds to 0-indexed actual rows 'r = 0, ..., max_rows-2'.
        for r_idx_plus_1 in range(1, max_rows):
            # Current 0-indexed row
            r = r_idx_plus_1 - 1

            # Calculate the 0-indexed flattened position of the last token in row 'r'
            # Index is r * max_cols + (actual_num_cols_for_this_grid - 1)
            # num_cols_int has shape (*S, 1, 1)
            index_in_flat_sequence = r * max_cols + (num_cols_int - 1)
            # index_in_flat_sequence will also have shape (*S, 1, 1)

            # Ensure index is within bounds [0, L-1] if necessary, though typically
            # num_cols should be <= max_cols.
            # index_in_flat_sequence = jnp.clip(index_in_flat_sequence, 0, max_rows * max_cols - 1)

            # Use take_along_axis to gather the logits
            # grid_logits shape: (*S, L, V)
            # index_in_flat_sequence shape: (*S, 1, 1) - needs to select from L dimension
            # We want to select 1 logit vector per item in S, from the L dimension.
            # axis=-2 refers to the L dimension.
            # `indices` must be broadcastable to `arr.shape` except for `axis`.
            # `index_in_flat_sequence` (..., 1, 1) is broadcastable to (..., 1, V)
            # to match `grid_logits` (..., L, V) for gathering.
            end_of_row_logit_vector = jnp.take_along_axis(
                grid_logits,
                index_in_flat_sequence, # Indices for the L dimension
                axis=-2  # The sequence dimension (L)
            )
            # end_of_row_logit_vector will have shape (*S, 1, V)
            collected_logits.append(end_of_row_logit_vector)

        # Concatenate the (max_rows-1) collected logit vectors.
        # Each is (*S, 1, V), concatenating on axis -2 gives (*S, max_rows-1, V).
        if not collected_logits: # Should not happen if max_rows > 1
            # Handle case for max_rows=1 (e.g. return empty array of correct rank)
            leading_dims = grid_logits.shape[:-2]
            vocab_size = grid_logits.shape[-1]
            return jnp.empty((*leading_dims, 0, vocab_size), dtype=grid_logits.dtype)

        return jnp.concatenate(collected_logits, axis=-2)

    def _loss_from_pair_and_context(
        self,
        context: chex.Array, # Shape (*B_dims, H) or (*B_dims, N, H)
        pairs: chex.Array,   # Shape (*B_dims, R, C, 2) or (*B_dims, N, R, C, 2)
        grid_shapes: chex.Array, # Shape (*B_dims, 2, 2) or (*B_dims, N, 2, 2)
        dropout_eval: bool,
        matrix_size_rows: int, # New
        matrix_size_cols: int, # New
    ):
        config = self.decoder.config
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Ensure context has the N dimension if pairs/grid_shapes do
        # This logic might be too simplistic if B_dims varies.
        # Assume if pairs has N, context should too.
        if pairs.ndim == context.ndim + 3: # pairs=(B,N,R,C,2), context=(B,H) -> needs (B,N,H)
            context = context[:, None, :].repeat(pairs.shape[-4], axis=-2)
        elif pairs.ndim == context.ndim + 2 and pairs.shape[-4] == context.shape[-2]: # pairs=(B,N,R,C,2), context=(B,N,H)
            pass # Already aligned
        elif pairs.ndim == 5 and context.ndim == 2: # pairs=(B,R,C,2), context=(B,H) - for single eval
             pass
        # else:
        #     print(f"Warning: _loss_from_pair_and_context shape mismatch. Context: {context.shape}, Pairs: {pairs.shape}")


        try:
            context_matrix = self._convert_to_matrix(
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                latents=context, # Context is now correctly shaped before this call
            )
        except ValueError as e:
             print(f"Error converting context to matrix in loss: {e}, context_shape={context.shape}")
             batch_shape = context.shape[:-1] 
             dummy_loss = jnp.full(batch_shape, jnp.nan)
             dummy_metrics = tree_map(lambda x: jnp.full(batch_shape, jnp.nan),
                                     {"shape_row_loss": 0.0, "shape_col_loss": 0.0, "grid_loss": 0.0})
             return dummy_loss, dummy_metrics
        
        current_input_seq = input_seq
        final_row_logits, final_col_logits, final_grid_logits = None, None, None

        for t in range(matrix_size_cols):
            context_col = context_matrix[..., :, t]
            row_logits, col_logits, grid_logits = self.decoder(
                current_input_seq, output_seq, context_col, dropout_eval
            )
            if t == matrix_size_cols - 1:
                final_row_logits, final_col_logits, final_grid_logits = row_logits, col_logits, grid_logits
        
        if final_row_logits is None:
             raise ValueError("Final logits not computed in loss.")

        # grid_shapes[..., 1, 0] is output rows, grid_shapes[..., 1, 1] is output cols
        grid_shapes_row_out, grid_shapes_col_out = grid_shapes[..., 1, 0], grid_shapes[..., 1, 1]
        
        one_hot_grid_shapes_row_labels = jax.nn.one_hot(grid_shapes_row_out - 1, config.max_rows)
        row_loss = -jnp.sum(jax.nn.log_softmax(final_row_logits) * one_hot_grid_shapes_row_labels, axis=-1)

        one_hot_grid_shapes_col_labels = jax.nn.one_hot(grid_shapes_col_out - 1, config.max_cols)
        col_loss = -jnp.sum(jax.nn.log_softmax(final_col_logits) * one_hot_grid_shapes_col_labels, axis=-1)

        last_non_padded_logits = self._get_last_non_padded_logits(
            final_grid_logits, grid_shapes_col_out[..., None, None]
        )
        # Adjust slicing if final_grid_logits has an N dimension
        if final_grid_logits.ndim > 3 and last_non_padded_logits.ndim == final_grid_logits.ndim-1 : # B,N,L,V vs B,N,R-1,V
             # This part is tricky, ensure shapes match for .at[].set()
             # If final_grid_logits is (B,N,L,V) and last_non_padded_logits is (B,N,max_rows-1,V)
             # We need to ensure the slicing target matches the replacement shape.
             # Original: final_grid_logits = final_grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)
             # This slice assumes last_non_padded_logits is (..., (max_rows-1)*V_or_1_if_squeezed , :) which isn't right.
             # The original logic was: for each row k (0 to max_rows-2), the logit for token (k+1)*max_cols
             # (start of next row) is set to logit from end of row k.
             # `last_non_padded_logits` has shape (*S, max_rows-1, V).
             # So, `final_grid_logits.at[..., k*max_cols + max_cols, :]` (which is `(k+1)*max_cols`)
             # should be set by `last_non_padded_logits[..., k, :]`.
             
             # Let's apply this carefully
             grid_logits_updated = final_grid_logits
             for r_idx in range(config.max_rows - 1): # For rows 0 to max_rows-2
                 # Logits for the start of the *next* physical row (r_idx+1)
                 target_slice_start_idx = (r_idx + 1) * config.max_cols
                 # Logits from the end of the *current* row (r_idx)
                 replacement_logits = last_non_padded_logits[..., r_idx, :] # Shape (*S, V)
                 
                 # Expand dims if needed for broadcasting with target slice.
                 # Target slice shape will be (*S, V)
                 grid_logits_updated = grid_logits_updated.at[..., target_slice_start_idx, :].set(replacement_logits)
             final_grid_logits = grid_logits_updated

        else: # Simpler case, no N dim or shapes already align for broadcast set
            final_grid_logits = final_grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)


        target_grid_tokens = pairs[..., 1].reshape(*pairs.shape[:-3], -1)
        one_hot_grid_labels = jax.nn.one_hot(target_grid_tokens, config.vocab_size)
        grid_losses = -jnp.sum(jax.nn.log_softmax(final_grid_logits) * one_hot_grid_labels, axis=-1)
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row_out, grid_shapes_col_out)

        loss = row_loss + col_loss + grid_loss
        metrics = {
            "shape_row_loss": row_loss, "shape_col_loss": col_loss, "grid_loss": grid_loss,
        }
        return loss, metrics

    @nn.compact
    def _generate_output_from_context_v2( # Renamed from original LPN file, used by self.generate_output
        self, context: chex.Array, input: chex.Array, input_grid_shape: chex.Array,
        dropout_eval: bool, matrix_size_rows: int, matrix_size_cols: int,
        save_intermediate: bool = False,
    ) -> tuple[chex.Array, chex.Array, Optional[dict]]:
        config = self.decoder.config
        max_rows, max_cols, max_len = config.max_rows, config.max_cols, config.max_len

        try:
            context_matrix = self._convert_to_matrix(
                matrix_size_rows, matrix_size_cols, context
            )
        except ValueError as e:
             print(f"Error converting context to matrix in generation: {e}, context_shape={context.shape}")
             return jnp.zeros_like(input), jnp.ones_like(input_grid_shape), None

        current_input_grid = input
        final_predicted_shape_overall = input_grid_shape
        intermediate_outputs = {}

        for t in range(matrix_size_cols):
            context_col = context_matrix[..., :, t]
            flattened_grid_for_step_t = jnp.reshape(current_input_grid, (*current_input_grid.shape[:-2], -1))
            # Shape prediction uses the original input_grid_shape for the task
            current_input_seq_for_step_t = jnp.concatenate([input_grid_shape, flattened_grid_for_step_t], axis=-1)
            
            target_seq_for_shape = jnp.zeros(current_input_seq_for_step_t.shape[:-1] + (max_len + 2,), dtype=jnp.int32)

            def predict_shape_token_gen(target_s, is_row, cur_inp_s, ctx_col, drp_eval):
                r_logits, c_logits, _ = self.decoder(cur_inp_s, target_s, ctx_col, drp_eval)
                logits = r_logits if is_row else c_logits
                pred_token = jnp.argmax(logits, axis=-1).astype(jnp.int32) + 1
                token_idx = 0 if is_row else 1
                return target_s.at[..., token_idx].set(pred_token)

            target_seq_for_shape = predict_shape_token_gen(target_seq_for_shape, True, current_input_seq_for_step_t, context_col, dropout_eval)
            target_seq_for_shape = predict_shape_token_gen(target_seq_for_shape, False, current_input_seq_for_step_t, context_col, dropout_eval)
            predicted_shape_this_step = target_seq_for_shape[..., :2]

            initial_carry_scan = (target_seq_for_shape, current_input_seq_for_step_t, context_col)
            
            scan_constructor = nn.scan(
                DecoderStep, variable_broadcast="params",
                split_rngs={"params": False, "dropout": False}, length=max_len
            )
            scanned_step_module = scan_constructor(self.decoder, dropout_eval)
            final_carry_scan, _ = scanned_step_module(initial_carry_scan, jnp.arange(max_len))
            final_target_seq = final_carry_scan[0]

            predicted_grid_tokens = final_target_seq[..., 2:]
            current_input_grid = jnp.reshape(predicted_grid_tokens, (*predicted_grid_tokens.shape[:-1], max_rows, max_cols))
            
            if t == matrix_size_cols - 1:
                final_predicted_shape_overall = predicted_shape_this_step
            if save_intermediate:
                intermediate_outputs[t] = {"grid": current_input_grid, "shape": predicted_shape_this_step, "context_col": context_col}
        
        return current_input_grid, final_predicted_shape_overall, intermediate_outputs if save_intermediate else None

    def _optimize_context_recurrently_for_loss(
        self,
        initial_context_one_pair: chex.Array, # (*B, H)
        single_pair_data: chex.Array,         # (*B, R, C, 2)
        single_grid_shape_data: chex.Array,   # (*B, 2, 2)
        key_for_optim: chex.PRNGKey,          # (*B_keyshape) or broadcastable
        matrix_size_rows: int,
        matrix_size_cols: int,
        dropout_eval: bool, # Should be True for optimization
        num_steps: int,
        lr: float,
        optimizer_name: str = "adam", # Changed from optimizer to optimizer_name
        optimizer_kwargs: Optional[dict] = None,
        **other_ga_kwargs # e.g. lr_schedule etc. from original GA
    ):
        """Optimizes a single context vector for a single pair using recurrent decoder."""
        
        # Define the loss function for gradient ascent (negative loss for maximization)
        # It takes the context and returns scalar loss.
        def loss_for_ga(context_to_optimize): # context_to_optimize: (*B, H)
            # _loss_from_pair_and_context expects context potentially (*B,N,H) if N is present in pairs
            # Here, we are optimizing for a single pair, so N=1 effectively.
            # Reshape single_pair_data to have an N=1 dimension if _loss_from_pair_and_context expects it
            # pairs_for_loss = single_pair_data[:, None, ...] # Not needed if loss fn handles it
            # grid_shapes_for_loss = single_grid_shape_data[:, None, ...]

            loss_val, _ = self._loss_from_pair_and_context(
                context_to_optimize, # Pass it directly as (*B, H)
                single_pair_data,    # Pass as (*B, R, C, 2)
                single_grid_shape_data, # Pass as (*B, 2, 2)
                dropout_eval,
                matrix_size_rows,
                matrix_size_cols
            ) # loss_val is (*B,)
            return loss_val.mean() # Mean over batch if B > 1, else just scalar

        grad_fn = jax.value_and_grad(loss_for_ga)

        if optimizer_name == "adam":
            opt = optax.adam(lr, **(optimizer_kwargs or {}))
        elif optimizer_name == "sgd":
            opt = optax.sgd(lr, **(optimizer_kwargs or {}))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        opt_state = opt.init(initial_context_one_pair)
        current_context = initial_context_one_pair

        def ga_step(carry, _):
            ctx, opt_s = carry
            loss_value, grads = grad_fn(ctx)
            updates, new_opt_s = opt.update(grads, opt_s, ctx)
            new_ctx = optax.apply_updates(ctx, updates)
            return (new_ctx, new_opt_s), loss_value

        # We need to ensure the GA step respects the @nn.compact context if self.decoder is called
        # by grad_fn -> loss_for_ga -> _loss_from_pair_and_context -> self.decoder.
        # Since _loss_from_pair_and_context is not @compact itself, this is tricky.
        # The value_and_grad should be fine as long as no new Modules are init inside.
        # The `self.decoder` call is on an existing instance.

        (final_context, _), losses_over_steps = jax.lax.scan(ga_step, (current_context, opt_state), None, length=num_steps)
        return final_context


    def _optimize_context_recurrently_for_generation(
        self,
        initial_context_program: chex.Array, # (*B, H) - Single program context
        support_pairs_all: chex.Array,        # (*B, N, R, C, 2) - All N support pairs
        support_grid_shapes_all: chex.Array,  # (*B, N, 2, 2)
        key_for_optim: chex.PRNGKey,
        matrix_size_rows: int,
        matrix_size_cols: int,
        dropout_eval: bool, # Should be True
        num_steps: int,
        lr: float,
        optimizer_name: str = "adam",
        optimizer_kwargs: Optional[dict] = None,
        **other_ga_kwargs
    ):
        """Optimizes a single program context vector using all N support pairs."""
        
        def loss_for_ga_gen(context_to_optimize): # context_to_optimize: (*B, H)
            # Repeat the context_to_optimize N times to match support_pairs_all
            context_repeated_N_times = context_to_optimize[:, None, :].repeat(
                support_pairs_all.shape[1], axis=1
            ) # (*B, N, H)
            
            loss_val_per_pair, _ = self._loss_from_pair_and_context(
                context_repeated_N_times,
                support_pairs_all,
                support_grid_shapes_all,
                dropout_eval,
                matrix_size_rows,
                matrix_size_cols
            ) # loss_val_per_pair is (*B, N)
            return loss_val_per_pair.mean() # Mean over batch and N pairs

        grad_fn = jax.value_and_grad(loss_for_ga_gen)

        if optimizer_name == "adam":
            opt = optax.adam(lr, **(optimizer_kwargs or {}))
        # ... (add sgd and error handling like above)
        else: opt = optax.sgd(lr, **(optimizer_kwargs or {}))


        opt_state = opt.init(initial_context_program)
        current_context = initial_context_program

        # GA scan loop (same as in _for_loss version)
        def ga_step_gen(carry, _):
            ctx, opt_s = carry
            loss_value, grads = grad_fn(ctx)
            updates, new_opt_s = opt.update(grads, opt_s, ctx)
            new_ctx = optax.apply_updates(ctx, updates)
            return (new_ctx, new_opt_s), loss_value
        
        (final_context, _), _ = jax.lax.scan(ga_step_gen, (current_context, opt_state), None, length=num_steps)
        return final_context # Return only the optimized context


    # --- generate_output method (incorporating recurrent_ga and random_search) ---
    def generate_output(
        self,
        pairs: chex.Array, input: chex.Array, grid_shapes: chex.Array, input_grid_shape: chex.Array, # Swapped pairs and input for consistency
        key: Optional[chex.PRNGKey], dropout_eval: bool,
        matrix_size_rows: int, matrix_size_cols: int,
        mode: Literal["direct_sample", "mean_program_params", "cross_attention_multi_sample", "matrix", "recurrent_ga", "random_search"],
        return_two_best: bool = False,
        **mode_kwargs,
    ):
        # <<< DEBUG PRINT 0 (LPN level) >>>
        jax.debug.print("LPN.generate_output: pairs.shape = {shape}", shape=pairs.shape)
        jax.debug.print("LPN.generate_output: grid_shapes.shape = {shape}", shape=grid_shapes.shape)
        
        latents_mu_all, latents_logvar_all = self.encoder(pairs, grid_shapes, dropout_eval)
        if latents_logvar_all is None: raise ValueError("VAE required.")
        if key is None: raise ValueError("Key required for generation.")

        key_prog_params, key_sampling, key_optim = jax.random.split(key, 3) # Split for GA/RS

        program_mu, program_logvar = self._compute_global_attention_params_context(
            latents_mu_all, latents_logvar_all
        )
        program_std = jnp.exp(0.5 * program_logvar)
        info = {"program_mu": program_mu, "program_logvar": program_logvar}

        initial_effective_context = program_mu + program_std * jax.random.normal(
            key_sampling, program_mu.shape
        ) # Base sample for some modes
        
        effective_context = None
        second_effective_context = None # For random_search

        if mode == "direct_sample":
            effective_context = initial_effective_context
        elif mode == "mean_program_params":
            effective_context = program_mu
        elif mode == "cross_attention_multi_sample":
            num_samples_for_attn = mode_kwargs.get("num_samples_for_attn", 16)
            multi_samples = program_mu[:, None, :] + program_std[:, None, :] * jax.random.normal(
                key_sampling, (program_mu.shape[0], num_samples_for_attn, program_mu.shape[-1])
            )
            effective_context = self._compute_global_attention_context(multi_samples)
            info["multi_samples_for_attn"] = multi_samples
        elif mode == "matrix":
            effective_context = initial_effective_context
        elif mode == "recurrent_ga":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"GA mode requires '{arg}'"
            effective_context = self._optimize_context_recurrently_for_generation(
                initial_context_program=initial_effective_context, # Start from a sample
                support_pairs_all=pairs,
                support_grid_shapes_all=grid_shapes,
                key_for_optim=key_optim,
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                dropout_eval=dropout_eval, # True for GA optimization
                **mode_kwargs # num_steps, lr, optimizer_name, etc.
            )
        elif mode == "random_search":
            # Ensure matrix_size_rows/cols are passed if _get_random_search_context_original uses them
            effective_context, second_effective_context = self._get_random_search_context_original( # Use old name
                latents=latents_mu_all, # Base search on original example latents
                pairs=pairs,            # Evaluate against all N support pairs
                grid_shapes=grid_shapes,
                key=key_optim,
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                dropout_eval=dropout_eval,
                **mode_kwargs
            )


        else:
            raise ValueError(f"Unsupported generation mode: {mode}")

        if effective_context is None: raise RuntimeError("Effective context not set.")
        if return_two_best and second_effective_context is None:
            second_effective_context = effective_context
        
        info["final_effective_context"] = effective_context
        if return_two_best and mode == "random_search": # Only RS naturally provides two distinct
             info["second_final_effective_context"] = second_effective_context

        if return_two_best and mode == "random_search":
            contexts_to_generate = jnp.stack([effective_context, second_effective_context], axis=0)
            output_grids_stack, output_shapes_stack, intermediate_dict_stack = jax.vmap(
                partial(self._generate_output_from_context_v2, input=input, input_grid_shape=input_grid_shape,
                        dropout_eval=dropout_eval, matrix_size_rows=matrix_size_rows,
                        matrix_size_cols=matrix_size_cols,
                        save_intermediate=mode_kwargs.get("save_intermediate_outputs", False)),
                in_axes=0
            )(contexts_to_generate)
            f_grids, s_grids = output_grids_stack[0], output_grids_stack[1]
            f_shapes, s_shapes = output_shapes_stack[0], output_shapes_stack[1]
            inter_dict = None
            if intermediate_dict_stack is not None:
                inter_dict = {"best": intermediate_dict_stack[0], "second_best": intermediate_dict_stack[1]}
            return f_grids, f_shapes, s_grids, s_shapes, info, inter_dict
        else:
            grids, shapes, inter_dict = self._generate_output_from_context_v2(
                effective_context, input, input_grid_shape, dropout_eval,
                matrix_size_rows, matrix_size_cols,
                mode_kwargs.get("save_intermediate_outputs", False)
            )
            if return_two_best: # For other modes, just duplicate if asked
                return grids, shapes, grids, shapes, info, inter_dict
            else:
                return grids, shapes, info, inter_dict

    def _get_random_search_context_original(
        self,
        latents: chex.Array, # These are the N example latents from encoder (*B,N,H)
        pairs: chex.Array,   # These are the N example pairs (*B,N,R,C,2) for evaluation
        grid_shapes: chex.Array,
        key: chex.PRNGKey,
        matrix_size_rows: int, # Added
        matrix_size_cols: int, # Added
        dropout_eval: bool,    # Added
        num_samples: int,
        scale: float,
        scan_batch_size: Optional[int] = None,
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        **kwargs, # e.g. grid_log_prob_weight for _compute_log_probs
    ):
        # `latents` here are the search base candidates: (*B, num_base, H)
        # This might be different from the N example latents directly.
        search_base_latents = self._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents # latents are (*B,N,H)
        ) # search_base_latents is (*B, num_search_bases, H)

        if num_samples > 0:
            num_base = search_base_latents.shape[-2]
            # Ensure num_padded_samples is multiple of num_base if num_base > 0
            if num_base > 0:
                num_padded_samples = math.ceil(num_samples / num_base) * num_base
                random_vectors = jax.random.normal(
                    key,
                    (*search_base_latents.shape[:-2], num_base, num_padded_samples // num_base, search_base_latents.shape[-1]),
                )
                random_latents_generated = search_base_latents[..., None, :] + scale * random_vectors
                random_latents_generated = random_latents_generated.reshape(
                    *random_latents_generated.shape[:-3], -1, random_latents_generated.shape[-1]
                )[..., :num_samples, :]
            else: # If no base latents, generate from zero or mean
                mean_for_random = latents.mean(axis=-2, keepdims=True) # Use original mean
                random_vectors = jax.random.normal(key, (*mean_for_random.shape[:-2], 1, num_samples, mean_for_random.shape[-1]))
                random_latents_generated = mean_for_random[...,None,:] + scale * random_vectors
                random_latents_generated = random_latents_generated.squeeze(axis=-3)


            search_base_latents = jnp.concatenate([search_base_latents, random_latents_generated], axis=-2)

        all_candidate_contexts = search_base_latents # Shape (*B, total_candidates, H)
        
        # Evaluate each candidate context against ALL N pairs
        # For each candidate_context in all_candidate_contexts (*B,H):
        #   loss = _loss_from_pair_and_context( expand(candidate_context, N), pairs, grid_shapes) -> (*B,N)
        #   score = mean(loss) -> (*B,)
        # We need scores of shape (*B, total_candidates)

        def score_one_candidate(candidate_ctx_one): # (*B, H)
            # Expand to (*B, N, H) to match `pairs`
            ctx_expanded = candidate_ctx_one[:, None, :].repeat(pairs.shape[1], axis=1)
            loss_all_N, _ = self._loss_from_pair_and_context(
                ctx_expanded, pairs, grid_shapes, dropout_eval,
                matrix_size_rows, matrix_size_cols
            ) # (*B, N)
            return loss_all_N.mean(axis=-1) # Score is mean loss over N pairs (*B,)

        # Vmap over the `total_candidates` dimension of `all_candidate_contexts`
        # all_candidate_contexts is (*B, total_candidates, H)
        # We want output scores (*B, total_candidates)
        # So vmap in_axes=-2 (candidate dim), out_axes=-1 (candidate score dim)
        
        # Ensure input for vmap is correct.
        # If all_candidate_contexts = (B, C, H), then after vmap(in_axes=1),
        # score_one_candidate gets (B,H).
        candidate_losses = jax.vmap(score_one_candidate, in_axes=1, out_axes=1)(all_candidate_contexts)
        # candidate_losses will be (B, total_candidates)
        
        # log_probs are negative losses
        log_probs = -candidate_losses

        best_context, second_best_context = self._select_best_and_second_best_latents(
            log_probs, all_candidate_contexts
        )
        return best_context, second_best_context

    @classmethod
    def _prepare_latents_before_search(
        cls,
        include_mean_latent: bool,
        include_all_latents: bool,
        latents: chex.Array,
        random_perturbation: Optional[dict] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """
        Selects the latents from which to start the search. If include_mean_latent is True, the mean latent
        is included in the latents from which to start the search. If include_all_latents is True, all the pair
        latents are included in the latents from which to start the search. If both are True, the mean latent
        is concatenated to the latents from which to start the search. If both are False, an error is raised.

        Args:
            include_mean_latent: if true, includes the mean latent in the latents from which to start the search.
            include_all_latents: if true, includes all the pair latents in the latents from which to start the
                search.
            latents: latents from the encoder. Shape (*B, N, H).
            random_perturbation: dictionary of random perturbation arguments. If not None, the following
                arguments are required:
                - num_samples: number of random samples to generate around the mean latent.
                - scale: Gaussian scale of the random perturbations.
            key: random key to generate the random perturbation. Shape (2,).

        Returns:
            latents: latents from which to start the search. Shape (*B, 1, H), (*B, N, H), or (*B, N+1, H).
        """
        if include_mean_latent:
            mean_latent = latents.mean(axis=-2, keepdims=True)
            if include_all_latents:
                # Include the mean latent in the latents from which to start the search.
                prep_latents = jnp.concatenate([mean_latent, latents], axis=-2)
            else:
                # Only start the search from the mean latent.
                prep_latents = mean_latent
        else:
            # Start the search from all the pair latents.
            if not include_all_latents:
                raise ValueError(
                    "At least one of 'include_mean_latent' or 'include_all_latents' should be True."
                )
            prep_latents = latents
        if random_perturbation is not None:
            assert key is not None, "'key' argument required for random perturbation."
            for arg in ["num_samples", "scale"]:
                assert arg in random_perturbation, f"'{arg}' argument required for random perturbation."
            num_samples = random_perturbation["num_samples"]
            scale = random_perturbation["scale"]
            random_vectors = jax.random.normal(key, (*latents.shape[:-2], num_samples, latents.shape[-1]))
            random_latents = latents.mean(axis=-2, keepdims=True) + scale * random_vectors
            prep_latents = jnp.concatenate([prep_latents, random_latents], axis=-2)
        return prep_latents


# --- Dummy classes for testing if run directly ---
if __name__ == "__main__":
    # Example usage or test setup would go here
    print("LPN class defined with cross_attention and attention_params modes.")
    # Need to define dummy Transformer configs and modules for instantiation.
    # from src.models.utils import TransformerLayerConfig # Assuming this exists

    # Example Configs (adjust dimensions as needed)
    latent_dim = 64
    embed_dim = 64
    num_heads = 4
    num_layers = 3
    max_len = 25 # 5x5
    vocab_size = 10
    max_rows = 5
    max_cols = 5

    enc_config = EncoderTransformerConfig(
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        latent_dim=latent_dim, vocab_size=vocab_size, use_vae=True, # Enable VAE
        max_rows=max_rows, max_cols=max_cols
    )
    dec_config = DecoderTransformerConfig(
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        latent_dim=latent_dim, vocab_size=vocab_size,
        max_rows=max_rows, max_cols=max_cols
    )

    # Dummy Modules (replace with actual imports if available)
    class DummyEncoder(nn.Module):
        config: EncoderTransformerConfig
        @nn.compact
        def __call__(self, pairs, grid_shapes, dropout_eval):
             B, N, R, C, _ = pairs.shape
             H = self.config.latent_dim
             mu = jnp.zeros((B, N, H))
             logvar = jnp.zeros((B, N, H)) # Log variance = 0 -> Variance = 1
             return mu, logvar if self.config.use_vae else None

    class DummyDecoder(nn.Module):
        config: DecoderTransformerConfig
        @nn.compact
        def __call__(self, input_seq, output_seq, context, dropout_eval):
            B = input_seq.shape[0]
            N = input_seq.shape[1] if input_seq.ndim == 3 else 1 # Handle generation case
            R, C, V = self.config.max_rows, self.config.max_cols, self.config.vocab_size
            L = R*C

            # Adjust output shape based on input sequence shape (training vs generation)
            if input_seq.ndim == 3: # Training: (*B, N, L+2)
                row_logits = jnp.zeros((B, N, R))
                col_logits = jnp.zeros((B, N, C))
                grid_logits = jnp.zeros((B, N, L, V))
            else: # Generation: (*B, L+2)
                row_logits = jnp.zeros((B, R))
                col_logits = jnp.zeros((B, C))
                grid_logits = jnp.zeros((B, L, V))
            return row_logits, col_logits, grid_logits

    # Instantiate LPN
    lpn = LPN(encoder=DummyEncoder(enc_config), decoder=DummyDecoder(dec_config))

    # Example Dummy Data
    B, N_pairs, R_max, C_max = 2, 4, max_rows, max_cols
    dummy_pairs = jnp.zeros((B, N_pairs, R_max, C_max, 2), dtype=jnp.int32)
    dummy_grid_shapes = jnp.ones((B, N_pairs, 2, 2), dtype=jnp.int32) * 3 # e.g., 3x3 grids
    dummy_grid_shapes = jnp.clip(dummy_grid_shapes, 1, max(R_max, C_max))

    # --- Test __call__ ---
    key = jax.random.PRNGKey(0)
    key, call_key = jax.random.split(key)
    params = lpn.init(call_key, dummy_pairs, dummy_grid_shapes, True, max_rows, 1, "mean")["params"] # Init with mean mode

    print("\nTesting __call__ with 'attention_params'...")
    try:
        loss, metrics = lpn.apply(
            {"params": params},
            dummy_pairs,
            dummy_grid_shapes,
            dropout_eval=True,
            matrix_size_rows=latent_dim, # Example: latent_dim rows
            matrix_size_cols=1,        # Example: 1 column (no recurrence needed if 1)
            mode="attention_params",
            prior_kl_coeff=0.1,
            pairwise_kl_coeff=0.01,
            context_kl_coeff=0.1, # Required for attention_params
            rngs={"latents_sample_orig": key, "latents_sample_context": key+1} # Provide RNGs
        )
        print("Loss:", loss)
        # print("Metrics:", metrics) # Can be verbose
        print("'attention_params' call successful.")
    except Exception as e:
        print(f"'attention_params' call failed: {e}")
        import traceback
        traceback.print_exc()


    # --- Test generate_output ---
    dummy_input = jnp.zeros((B, R_max, C_max), dtype=jnp.int32)
    dummy_input_shape = jnp.ones((B, 2), dtype=jnp.int32) * 3
    dummy_input_shape = jnp.clip(dummy_input_shape, 1, max(R_max, C_max))
    key, gen_key = jax.random.split(key)

    print("\nTesting generate_output with 'attention_params'...")
    try:
        grids, shapes, info, _ = lpn.apply(
             {"params": params},
             dummy_pairs, dummy_grid_shapes, # Support examples
             dummy_input, dummy_input_shape, # New input
             key=gen_key, # RNG key for sampling context
             dropout_eval=True,
             matrix_size_rows=latent_dim,
             matrix_size_cols=1,
             mode="attention_params",
             return_two_best=False,
             mutable=False, # generate_output doesn't modify state typically
             method=lpn.generate_output # Specify the method
        )
        print("Generated grid shape:", grids.shape)
        print("Generated shape shape:", shapes.shape)
        print("'attention_params' generation successful.")
    except Exception as e:
        print(f"'attention_params' generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting generate_output with 'cross_attention'...")
    try:
         grids, shapes, info, _ = lpn.apply(
              {"params": params},
              dummy_pairs, dummy_grid_shapes, # Support examples
              dummy_input, dummy_input_shape, # New input
              key=gen_key+1, # RNG key for sampling latents
              dropout_eval=True,
              matrix_size_rows=latent_dim,
              matrix_size_cols=1,
              mode="cross_attention",
              return_two_best=False,
              mutable=False,
              method=lpn.generate_output
         )
         print("Generated grid shape:", grids.shape)
         print("Generated shape shape:", shapes.shape)
         print("'cross_attention' generation successful.")
    except Exception as e:
         print(f"'cross_attention' generation failed: {e}")
         import traceback
         traceback.print_exc()

# --- END OF FILE ---