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
    # New: Define how program parameters are combined (e.g. 'attention', 'mean_params')
    # For now, we'll hardcode attention-based combination as per the request.
    # program_param_combination_method: Literal["attention"] = "attention" # Could be a config

    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
        matrix_size_rows: jnp.int32,
        matrix_size_cols: jnp.int32,
        # Mode now dictates how to use the combined program parameters/samples
        mode: Literal["direct_sample", "mean_sample", "cross_attention_sample", "matrix", "recurrent_ga"],
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        program_kl_coeff: Optional[float] = None, # KL for the combined program parameters vs prior
        **mode_kwargs,
    ):
        assert pairs.shape[-4] > 1, f"Number of pairs > 1 required, got {pairs.shape[-4]}."
        # 1. Encode all N example pairs
        latents_mu_all, latents_logvar_all = self.encoder(pairs, grid_shapes, dropout_eval) # (*B, N, H)

        kl_metrics = {}
        original_prior_kl_loss = None
        original_pairwise_kl_loss = None
        combined_program_kl_loss = None

        if latents_logvar_all is None:
            raise ValueError("VAE (latents_logvar) is required for this architecture.")

        # --- KL for original N latent distributions ---
        original_prior_kl_loss = jnp.mean(
            -0.5 * jnp.sum(1 + latents_logvar_all - latents_mu_all**2 - jnp.exp(latents_logvar_all), axis=-1)
        )
        kl_metrics["original_prior_kl"] = original_prior_kl_loss
        original_pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu_all, latents_logvar_all).mean()
        kl_metrics["original_pairwise_kl"] = original_pairwise_kl_loss
        kl_metrics["latents_mu_all_mean"] = latents_mu_all.mean()
        kl_metrics["latents_logvar_all_mean"] = latents_logvar_all.mean()


        # 2. MANDATORY: Combine N latent parameters into N leave-one-out "program parameters"
        # These are the parameters of the distribution P(program | N-1 examples) for each of the N examples.
        leave_one_out_mu_all = make_leave_one_out(latents_mu_all, axis=-2)     # (*B, N, N-1, H)
        leave_one_out_logvar_all = make_leave_one_out(latents_logvar_all, axis=-2) # (*B, N, N-1, H)

        # `program_mu_loo[b,i,:]` is the mu for the program derived for example `i` using others.
        program_mu_loo, program_logvar_loo = self._compute_attention_params_context(
            target_mu=latents_mu_all,        # (*B, N, H) - Query for each of N contexts
            loo_mu=leave_one_out_mu_all,     # (*B, N, N-1, H) - Keys/Values
            loo_logvar=leave_one_out_logvar_all
        ) # Shapes: (*B, N, H)

        # KL for these N combined program parameter sets vs prior N(0,I)
        if program_kl_coeff is not None:
            combined_program_kl_loss = jnp.mean(
                -0.5 * jnp.sum(1 + program_logvar_loo - program_mu_loo**2 - jnp.exp(program_logvar_loo), axis=-1)
            )
            kl_metrics["combined_program_kl"] = combined_program_kl_loss
        kl_metrics["program_mu_loo_mean"] = program_mu_loo.mean()
        kl_metrics["program_logvar_loo_mean"] = program_logvar_loo.mean()

        # 3. Determine the "effective context" to be used for loss calculation based on mode
        # This context will have shape (*B, N, H) - one for each (pair_i, program_derived_for_i)
        
        key_sample_effective_context = self.make_rng("effective_context_sample")

        if mode == "direct_sample":
            # Sample directly from each of the N program_parameter_sets.
            program_std_loo = jnp.exp(0.5 * program_logvar_loo)
            effective_contexts = program_mu_loo + program_std_loo * jax.random.normal(
                key_sample_effective_context, program_mu_loo.shape
            ) # (*B, N, H)
        
        elif mode == "mean_sample": # Formerly 'mean'
            # "Mean" here means using the mean of the program parameters directly, no sampling noise.
            effective_contexts = program_mu_loo # (*B, N, H)

        elif mode == "cross_attention_sample":
            # 1. Sample N "program proposals" from P(program | N-1 examples)
            program_std_loo = jnp.exp(0.5 * program_logvar_loo)
            program_samples_loo = program_mu_loo + program_std_loo * jax.random.normal(
                key_sample_effective_context, program_mu_loo.shape
            ) # (*B, N, H) - program_samples_loo[b,i,:] is a sample for program_i

            # 2. For each target 'i', combine the *other N-1* program_samples_loo[b,j,:] (j!=i)
            #    using cross-attention.
            loo_program_samples = make_leave_one_out(program_samples_loo, axis=-2) # (*B, N, N-1, H)
            effective_contexts = self._compute_loo_cross_attention_context(
                loo_program_samples
            ) # (*B, N, H)

        elif mode == "matrix":
            # The matrix mode operates on a single context vector per pair.
            # We'll use a sample from the combined program parameters.
            program_std_loo = jnp.exp(0.5 * program_logvar_loo)
            effective_contexts = program_mu_loo + program_std_loo * jax.random.normal(
                key_sample_effective_context, program_mu_loo.shape
            ) # (*B, N, H)
            # The matrix reshaping happens inside _loss_from_pair_and_context
        
        elif mode == "recurrent_ga":
             # This mode is complex. It originally optimized latents.
             # Now it should optimize the *program parameters* or a *sample* from them.
             # For now, let's assume it starts with sampled effective_contexts and refines them.
             # This needs careful re-thinking of _get_recurrent_ga_context.
             # As a placeholder, let's use direct samples.
            program_std_loo = jnp.exp(0.5 * program_logvar_loo)
            initial_contexts_for_ga = program_mu_loo + program_std_loo * jax.random.normal(
                key_sample_effective_context, program_mu_loo.shape
            ) # (*B, N, H)
            
            key_ga_pert = self.make_rng("gradient_ascent_random_perturbation") if mode_kwargs.get("random_perturbation") else None
            # Note: _get_recurrent_ga_context needs to be adapted to take these initial_contexts
            # and the (pairs, grid_shapes) for its optimization objective.
            # The original _get_recurrent_ga_context took leave_one_out_latents, etc.
            # This is a MAJOR change for recurrent_ga.
            # For now, let's bypass actual GA and just use the initial samples.
            print("Warning: 'recurrent_ga' mode in __call__ is using direct samples without GA optimization due to architectural change. Needs full rework.")
            effective_contexts = initial_contexts_for_ga
            # effective_contexts, _ = self._get_recurrent_ga_context(
            #     initial_contexts_for_ga, # Instead of loo_latents
            #     pairs, # Not leave-one-out pairs, but all N pairs to optimize against
            #     grid_shapes,
            #     key_ga_pert,
            #     matrix_size_rows=matrix_size_rows,
            #     matrix_size_cols=matrix_size_cols,
            #     **mode_kwargs
            # )
        else:
            raise ValueError(f"Unsupported mode for __call__: {mode}")

        # 4. Loss Calculation using the effective_contexts
        loss, metrics = self._loss_from_pair_and_context(
            effective_contexts, # (*B, N, H)
            pairs,              # (*B, N, R, C, 2)
            grid_shapes,        # (*B, N, 2, 2)
            dropout_eval,
            matrix_size_cols=matrix_size_cols,
            matrix_size_rows=matrix_size_rows
        ) # loss shape (*B, N), metrics shapes (*B, N, ...)

        # --- Aggregate Metrics and Final Loss ---
        # Sample original latents for comparison metrics if VAE was used for them
        if latents_logvar_all is not None:
            key_sample_orig_latents = self.make_rng("latents_sample_orig_for_metrics")
            latents_std_all = jnp.exp(0.5 * latents_logvar_all)
            original_sampled_latents = latents_mu_all + latents_std_all * jax.random.normal(
                key_sample_orig_latents, latents_mu_all.shape
            ) # (*B, N, H)
        else: # Should not happen given earlier check
            original_sampled_latents = latents_mu_all


        # Metrics comparing effective_contexts to original_sampled_latents
        metrics.update(
            # Norms are per-example
            original_sampled_latents_norm=norm(original_sampled_latents, axis=-1), # (*B,N)
            effective_contexts_norm=norm(effective_contexts, axis=-1), # (*B,N)
            distance_effective_context_vs_original_sample=norm(effective_contexts - original_sampled_latents, axis=-1), # (*B,N)
        )
        
        # Average loss and metrics over the N pairs dimension
        loss, metrics = tree_map(jnp.mean, (loss, metrics)) # Average over N dim -> shapes (*B,)
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


    # Removed _sample_latents as its logic is now integrated into __call__

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


    def _loss_from_pair_and_context(
        self,
        context: chex.Array, # Shape (*B, N, H)
        pairs: chex.Array,   # Shape (*B, N, R, C, 2)
        grid_shapes: chex.Array, # Shape (*B, N, 2, 2)
        dropout_eval: bool,
        matrix_size_rows: int = 64,
        matrix_size_cols: int = 1,
    ):
        """
        Computes the loss for each pair given its corresponding context.
        Applies the recurrent matrix logic internally.
        Args:
            context: Context vectors for each pair. Shape (*B, N, H).
            pairs: Input/output pairs. Shape (*B, N, R, C, 2).
            grid_shapes: Grid shapes for each pair. Shape (*B, N, 2, 2).
            dropout_eval: Dropout evaluation mode.
            matrix_size_rows: Rows for matrix reshaping.
            matrix_size_cols: Columns for matrix reshaping (number of recurrent steps).
        Returns:
            loss: Loss value per pair. Shape (*B, N).
            metrics: Dictionary of metrics per pair. Shape (*B, N, ...).
        """
        config = self.decoder.config

        # Make the input and output sequences. Shapes (*B, N, R*C+2)
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Reshape context into matrix format (*B, N, rows, cols)
        try:
            context_matrix = self._convert_to_matrix(
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                latents=context,
            )
        except ValueError as e:
             print(f"Error converting context to matrix: {e}")
             # Handle error appropriately, maybe return NaN loss or raise
             batch_shape = context.shape[:-1] # (*B, N)
             dummy_loss = jnp.full(batch_shape, jnp.nan)
             dummy_metrics = tree_map(lambda x: jnp.full(batch_shape, jnp.nan),
                                     {"shape_row_loss": 0.0, "shape_col_loss": 0.0, "grid_loss": 0.0, "total_loss": 0.0})
             return dummy_loss, dummy_metrics


        current_input_seq = input_seq # Start with the original input sequence
        final_row_logits, final_col_logits, final_grid_logits = None, None, None

        for t in range(matrix_size_cols):
            # Get the context for the current column/step (*B, N, rows)
            context_col = context_matrix[..., :, t] # Correct indexing for (..., rows, cols)

            # Generate logits using the current context column and input sequence
            # We use teacher forcing with the *true* output_seq here for loss calculation
            row_logits, col_logits, grid_logits = self.decoder(
                current_input_seq, # Input sequence for this step
                output_seq,        # True target sequence (teacher forcing)
                context_col,       # Context for this step
                dropout_eval
            )

            # In loss calculation (teacher forcing), the input for the *next* step
            # theoretically shouldn't depend on the prediction of the *current* step.
            # However, if the intention of the recurrent structure is that the *effective context*
            # changes based on intermediate states, then the `current_input_seq` might need
            # updating based on `output_seq` or predictions if it were generation.
            # For standard teacher-forced loss, `current_input_seq` could remain `input_seq`.
            # Let's assume `current_input_seq` stays `input_seq` for loss calculation,
            # matching typical transformer teacher forcing where only the target shifts.
            # If the design intends input to evolve, this needs clarification.
            # current_input_seq = updated_input_seq # If input needed updating based on previous step

            # Store the logits from the *final* recurrent step
            if t == matrix_size_cols - 1:
                final_row_logits = row_logits
                final_col_logits = col_logits
                final_grid_logits = grid_logits

        # Compute cross entropy losses using logits from the final step
        if final_row_logits is None: # Should not happen if matrix_size_cols >= 1
             raise ValueError("Final logits were not computed. matrix_size_cols might be 0?")

        grid_shapes_row, grid_shapes_col = grid_shapes[..., 1, 0], grid_shapes[..., 1, 1] # Target output shapes
        # -1 to shift the tokens to [0, max_rows-1]
        one_hot_grid_shapes_row_labels = jax.nn.one_hot(grid_shapes_row - 1, config.max_rows)
        row_loss = -jnp.sum(jax.nn.log_softmax(final_row_logits) * one_hot_grid_shapes_row_labels, axis=-1)

        # -1 to shift the tokens to [0, max_cols-1]
        one_hot_grid_shapes_col_labels = jax.nn.one_hot(grid_shapes_col - 1, config.max_cols)
        col_loss = -jnp.sum(jax.nn.log_softmax(final_col_logits) * one_hot_grid_shapes_col_labels, axis=-1)

        # Process grid logits (handle padding/wrapping)
        last_non_padded_logits = self._get_last_non_padded_logits(
            final_grid_logits, grid_shapes_col[..., None, None] # Use target col shape
        )
        final_grid_logits = final_grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)

        # Target grid tokens (*B, N, R*C)
        target_grid_tokens = pairs[..., 1].reshape(*pairs.shape[:-3], -1)
        one_hot_grid_labels = jax.nn.one_hot(target_grid_tokens, config.vocab_size)
        grid_losses = -jnp.sum(jax.nn.log_softmax(final_grid_logits) * one_hot_grid_labels, axis=-1) # (*B, N, R*C)
        # Normalize grid loss by actual sequence length (*B, N)
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row, grid_shapes_col)

        loss = row_loss + col_loss + grid_loss # Total loss per pair (*B, N)
        metrics = {
            "shape_row_loss": row_loss,
            "shape_col_loss": col_loss,
            "grid_loss": grid_loss,
            # "total_loss": loss, # This will be added outside after KL terms
        }
        return loss, metrics # Return loss and metrics per pair


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

    def generate_output(
        self,
        pairs: chex.Array, # Supporting examples (*B, N, R, C, 2)
        grid_shapes: chex.Array, # Supporting examples shapes (*B, N, 2, 2)
        input: chex.Array, # New input grid (*B, R, C)
        input_grid_shape: chex.Array, # New input shape (*B, 2)
        key: Optional[chex.PRNGKey],
        dropout_eval: bool,
        matrix_size_rows: int,
        matrix_size_cols: int,
        # Mode dictates how to use the single combined program parameters/samples
        mode: Literal["direct_sample", "mean_program_params", "cross_attention_multi_sample", "matrix", "random_search"],
        return_two_best: bool = False, # Only relevant for random_search
        **mode_kwargs,
    ):
        # 1. Encode all N support examples
        latents_mu_all, latents_logvar_all = self.encoder(pairs, grid_shapes, dropout_eval) # (*B, N, H)

        if latents_logvar_all is None:
            raise ValueError("VAE (latents_logvar) is required for this architecture.")
        if key is None:
            raise ValueError("PRNGKey 'key' is required for generate_output with this architecture.")

        # 2. MANDATORY: Combine N latent parameters into ONE set of "program parameters"
        # These are the parameters P(program | N examples)
        key_prog_params, key_sampling, key_search = jax.random.split(key, 3)

        program_mu, program_logvar = self._compute_global_attention_params_context(
            latents_mu_all, latents_logvar_all
        ) # Shapes: (*B, H)
        program_std = jnp.exp(0.5 * program_logvar)
        info = {"program_mu": program_mu, "program_logvar": program_logvar}


        # 3. Determine the single "effective_context" (*B, H) for generation
        effective_context = None
        second_effective_context = None # For random_search

        if mode == "direct_sample":
            # Sample once from the combined program distribution
            effective_context = program_mu + program_std * jax.random.normal(
                key_sampling, program_mu.shape
            )
        elif mode == "mean_program_params": # Use the mean of the combined program parameters
            effective_context = program_mu
        
        elif mode == "cross_attention_multi_sample":
            num_samples_for_attn = mode_kwargs.get("num_samples_for_attn", 16) # e.g., 16 samples
            # 1. Draw multiple samples from P(program | N examples)
            multi_samples = program_mu[:, None, :] + program_std[:, None, :] * jax.random.normal(
                key_sampling, (program_mu.shape[0], num_samples_for_attn, program_mu.shape[-1])
            ) # (*B, M, H)
            # 2. Combine these M samples using global attention
            effective_context = self._compute_global_attention_context(multi_samples) # (*B, H)
            info["multi_samples_for_attn"] = multi_samples
            
        elif mode == "matrix":
            # Sample once from the combined program distribution for matrix operations
            effective_context = program_mu + program_std * jax.random.normal(
                key_sampling, program_mu.shape
            )
            # Matrix reshaping happens inside _generate_output_from_context_v2

        elif mode == "random_search":
            # This random search now optimizes the single program vector.
            # It uses the N *support pairs* to evaluate candidates.
            # The `latents` arg to _get_random_search_context was (*B, N, H)
            # Now, our "base" for search is program_mu, program_logvar.
            # The original function's `latents` argument: Shape (*B, N, H).
            # This implied it could start search from individual example latents or their mean.
            # Now, we have a single program_mu.
            # We need to adapt _get_random_search_context or write a new version.

            # Let's adapt _get_random_search_context_for_program:
            # It will take program_mu, program_std as input.
            # `pairs` and `grid_shapes` are the *support examples* used for scoring.
            effective_context, second_effective_context = self._get_random_search_context_for_program(
                program_mu=program_mu,               # (*B, H)
                program_std=program_std,               # (*B, H)
                support_pairs=pairs,               # (*B, N, R, C, 2) - for scoring candidates
                support_grid_shapes=grid_shapes,   # (*B, N, 2, 2) - for scoring candidates
                key=key_search,
                matrix_size_rows=matrix_size_rows, # For loss calculation during search
                matrix_size_cols=matrix_size_cols, # For loss calculation during search
                **mode_kwargs # num_samples, scale, etc.
            )
            info["random_search_candidates_evaluated"] = mode_kwargs.get("num_samples",0) + \
                                                       (1 if mode_kwargs.get("include_mean_latent",True) else 0)
                                                       # + (latents_mu_all.shape[-2] if mode_kwargs.get("include_all_latents",False) else 0) # include_all_latents is tricky now

        else:
            raise ValueError(f"Unsupported generation mode: {mode}")

        if effective_context is None:
            raise RuntimeError(f"Effective context not set for mode {mode}")
        if return_two_best and second_effective_context is None: # e.g. for modes other than random_search
            second_effective_context = effective_context # Default second best to best

        info["final_effective_context"] = effective_context
        if return_two_best:
            info["second_final_effective_context"] = second_effective_context


        # 4. Generate output using the effective_context(s)
        # _generate_output_from_context_v2 expects a single context of shape (*B, H)
        
        if return_two_best and mode == "random_search": # Only random_search truly gives two distinct bests
            contexts_to_generate = jnp.stack([effective_context, second_effective_context], axis=0) # (2, *B, H)
            
            # vmap over the stack of 2 contexts
            output_grids_stack, output_shapes_stack, intermediate_dict_stack = jax.vmap(
                partial(
                    self._generate_output_from_context_v2, # This is now @nn.compact
                    input=input, input_grid_shape=input_grid_shape, dropout_eval=dropout_eval,
                    matrix_size_rows=matrix_size_rows, matrix_size_cols=matrix_size_cols,
                    save_intermediate=mode_kwargs.get("save_intermediate_outputs", False)
                ), in_axes=0
            )(contexts_to_generate)

            first_output_grids, second_output_grids = output_grids_stack[0], output_grids_stack[1]
            first_output_shapes, second_output_shapes = output_shapes_stack[0], output_shapes_stack[1]
            intermediate_dict = None
            if intermediate_dict_stack is not None:
                 intermediate_dict = {"best": intermediate_dict_stack[0], "second_best": intermediate_dict_stack[1]}
            return first_output_grids, first_output_shapes, second_output_grids, second_output_shapes, info, intermediate_dict
        else:
            output_grids, output_shapes, intermediate_dict = self._generate_output_from_context_v2(
                effective_context, input, input_grid_shape, dropout_eval,
                matrix_size_rows, matrix_size_cols,
                mode_kwargs.get("save_intermediate_outputs", False)
            )
            # If not random search but return_two_best was True, we just return the same prediction twice.
            if return_two_best:
                return output_grids, output_shapes, output_grids, output_shapes, info, intermediate_dict
            else:
                return output_grids, output_shapes, info, intermediate_dict

    def _prepare_search_candidates_for_program(
        self,
        program_mu: chex.Array, # (*B, H)
        program_std: chex.Array, # (*B, H)
        key: chex.PRNGKey,
        num_samples: int,
        scale: float, # Scale for random noise relative to program_std
        include_program_mu: bool = True,
        # include_original_latents_mean: bool = False, # Could add mean of original N latents
        # original_latents_mu_all: Optional[chex.Array] = None, # (*B, N, H)
    ):
        candidates = []
        if include_program_mu:
            candidates.append(program_mu[..., None, :]) # Add num_candidates dim

        if num_samples > 0:
            # Sample random vectors around program_mu
            noise = jax.random.normal(key, (*program_mu.shape[:-1], num_samples, program_mu.shape[-1]))
            # Scale noise by program_std and the user-provided scale
            random_candidates = program_mu[..., None, :] + (program_std[..., None, :] * scale * noise)
            candidates.append(random_candidates)
        
        if not candidates:
             # Fallback: if no samples and program_mu not included, use program_mu
             return program_mu[..., None, :] 

        return jnp.concatenate(candidates, axis=-2) # (*B, num_total_candidates, H)

    def _get_random_search_context_for_program(
        self,
        program_mu: chex.Array,         # (*B, H)
        program_std: chex.Array,        # (*B, H)
        support_pairs: chex.Array,      # (*B, N, R, C, 2)
        support_grid_shapes: chex.Array,# (*B, N, 2, 2)
        key: chex.PRNGKey,
        matrix_size_rows: int,
        matrix_size_cols: int,
        num_samples: int,
        scale: float, # Scale for random noise on program_mu
        scan_batch_size: Optional[int] = None, # For batching candidate evaluation
        include_program_mu: bool = True, # Whether to include program_mu as a candidate
        **kwargs, # Other args like grid_log_prob_weight for _compute_log_probs
    ):
        # 1. Prepare candidate program vectors
        # `latents` here are the candidate program vectors. Shape (*B, num_candidates, H)
        candidate_program_vectors = self._prepare_search_candidates_for_program(
            program_mu, program_std, key, num_samples, scale, include_program_mu
        )
        num_total_candidates = candidate_program_vectors.shape[-2]

        # 2. Evaluate each candidate program vector against ALL N support_pairs.
        # We need to compute the sum/mean loss for each candidate over the N support_pairs.
        
        # Reshape for vmap/scan:
        # candidate_program_vectors: (*B, CANDS, H)
        # support_pairs:             (*B, N, R, C, 2)
        # We want to evaluate each of CANDS with all N pairs.
        
        # Expand candidate_program_vectors to match N dim of pairs for loss fn:
        # Shape -> (*B, N, CANDS, H) by repeating CANDS for each of N.
        # No, simpler: vmap the loss function over candidates.
        # The loss function _loss_from_pair_and_context expects context (*B, N, H).
        # So, for each candidate, we need to make it (*B, N, H) by repeating it N times.

        # Flatten support_pairs/shapes for decoder (done inside _loss_from_pair_and_context)
        # input_seq_support, output_seq_support = self._flatten_input_output_for_decoding(
        #     support_pairs, support_grid_shapes
        # ) # (*B, N, Len+2)

        dropout_eval = True # No dropout during search evaluation

        def evaluate_one_candidate(candidate_vec):
            # candidate_vec: (*B, H)
            # Repeat this candidate N times to feed into loss function
            context_for_loss = candidate_vec[:, None, :].repeat(support_pairs.shape[-4], axis=-2) # (*B, N, H)
            
            loss_per_pair, _ = self._loss_from_pair_and_context(
                context_for_loss,
                support_pairs,
                support_grid_shapes,
                dropout_eval,
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols
            ) # loss_per_pair: (*B, N)
            # We want to minimize loss, so use negative log_prob (which is loss)
            # Average loss over N pairs for this candidate
            return loss_per_pair.mean(axis=-1) # (*B,) (negative_log_prob for this candidate)

        # Vmap evaluation over all candidates. `candidate_program_vectors` is (*B, CANDS, H)
        # `in_axes=(-2)` means vmap over the CANDS dimension.
        # Since candidate_program_vectors already has batch B, vmap handles it correctly.
        losses_for_candidates = jax.vmap(
            evaluate_one_candidate, 
            in_axes=-2, # Axis of candidates
            out_axes=-1 # Store results in last axis: (*B, CANDS)
        )(candidate_program_vectors) # losses_for_candidates: (*B, CANDS)

        # Log_probs are negative losses
        log_probs_for_candidates = -losses_for_candidates # (*B, CANDS)

        # Select best two based on these log_probs
        # _select_best_and_second_best_latents expects latents (*B, CANDS, H) and log_probs (*B, CANDS)
        best_context, second_best_context = self._select_best_and_second_best_latents(
            log_probs_for_candidates, candidate_program_vectors
        ) # Output shapes: (*B, H)

        return best_context, second_best_context

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


    # --- Other Helper Functions (Assumed to exist or need implementation) ---

    def _get_recurrent_ga_context(self, *args, **kwargs):
         # Original implementation - needs careful review if used for generation
         # For now, this is only called from __call__ with loo inputs
         raise NotImplementedError("_get_recurrent_ga_context needs implementation details from original code.")
         # Simplified placeholder if needed:
         # loo_latents = args[0]
         # return loo_latents.mean(axis=-2), None # Return mean context and None for second best

    # Placeholder for gradient ascent / random search context finding during generation
    # def _get_random_search_context(self, latents, pairs, grid_shapes, key, **mode_kwargs):
    #     # Needs implementation based on original LPN logic
    #     raise NotImplementedError
    # def _get_gradient_ascent_context(self, latents, pairs, grid_shapes, key, **mode_kwargs):
    #     # Needs implementation based on original LPN logic
    #     raise NotImplementedError


    @classmethod
    def _flatten_input_output_for_decoding(
        cls, pairs: chex.Array, grid_shapes: chex.Array
    ):
        # ... (implementation remains the same) ...
        flattened_pairs = jnp.reshape(pairs, (*pairs.shape[:-3], -1, 2))
        input_seq = jnp.concatenate([grid_shapes[..., 0,:], flattened_pairs[..., 0]], axis=-1) # Use input shape grid_shapes[..., 0]
        output_seq = jnp.concatenate([grid_shapes[..., 1,:], flattened_pairs[..., 1]], axis=-1)# Use output shape grid_shapes[..., 1]
        return input_seq, output_seq


    def _generate_logits_from_context(
        self,
        context: chex.Array,
        input_seq: chex.Array,
        true_output_seq: chex.Array,
        dropout_eval: bool,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        # ... (implementation remains the same, used only internally by loss?) ...
        # This seems specific to the original loss calculation? Let's keep it but check usage.
        # It's NOT used by _generate_output_from_context_v2 (autoregressive generation)
        # It IS used by the original _loss_from_pair_and_context if we revert to that logic.
        # The refactored _loss_from_pair_and_context now calls self.decoder directly.
        row_logits, col_logits, grid_logits = self.decoder(input_seq, true_output_seq, context, dropout_eval)
        output_shape = true_output_seq[..., :2]
        updated_input_seq = true_output_seq # Teacher forcing
        return row_logits, col_logits, grid_logits, updated_input_seq

    @nn.compact # Add this decorator
    def _generate_output_from_context_v2(
        self,
        context: chex.Array, # Single context vector (*B, H)
        input: chex.Array,   # Input grid (*B, R, C)
        input_grid_shape: chex.Array, # Input shape (*B, 2)
        dropout_eval: bool,
        matrix_size_rows: int,
        matrix_size_cols: int,
        save_intermediate: bool = False,
    ) -> tuple[chex.Array, chex.Array, Optional[dict]]:
        """
        Recurrently generates output grids using per-column context from a *single*
        input context vector. Autoregressive generation.
        Marked as @nn.compact to allow dynamic submodule definition for nn.scan.
        """
        config = self.decoder.config
        max_rows, max_cols = config.max_rows, config.max_cols
        max_len = config.max_len

        try:
            context_matrix = self._convert_to_matrix(
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                latents=context,
            )
        except ValueError as e:
             print(f"Error converting context to matrix during generation: {e}")
             dummy_grid = jnp.zeros_like(input)
             dummy_shape = jnp.ones_like(input_grid_shape)
             return dummy_grid, dummy_shape, None

        current_input_grid = input
        # current_input_shape is updated only at the end using the last step's prediction
        final_predicted_shape_overall = input_grid_shape # Initialize with input task shape
        intermediate_outputs = {}

        # This DecoderStep class definition must be accessible here.
        # It can be defined globally in the file or nested if Python scoping allows.
        # For Flax, it's often cleaner to define it outside or as a static member if possible.
        # Let's assume it's defined at the same level as the LPN class or globally.
        # If it's defined *inside* _generate_output_from_context_v2 (not ideal for @compact),
        # it needs to be a plain class, not an nn.Module, and nn.scan might need careful handling.

        # To work correctly with @nn.compact and nn.scan, DecoderStep should also be an nn.Module
        # And it will be instantiated by nn.scan.
        class DecoderStep(nn.Module):
            # The main decoder is an attribute of LPN, not directly of DecoderStep's definition.
            # We'll access it via `self.parent.decoder` if LPN is `self.parent`.
            # Or, more simply, pass the main decoder module instance to nn.scan.
            decoder_to_use: DecoderTransformer
            fixed_dropout_eval: bool # dropout_eval is fixed per generate_output call

            @nn.compact
            def __call__(self, carry, scan_input):
                # carry: (target_seq_prev_step, fixed_input_seq_for_decoder, fixed_context_col_for_decoder)
                # scan_input: current_token_idx_to_predict
                target_seq_prev_step, fixed_input_seq, fixed_context_col = carry
                current_token_idx_to_predict = scan_input

                _, _, grid_logits = self.decoder_to_use( # Call the passed decoder instance
                    fixed_input_seq, target_seq_prev_step, fixed_context_col, self.fixed_dropout_eval
                )
                current_token_logits = grid_logits[..., current_token_idx_to_predict, :]
                predicted_token = jnp.argmax(current_token_logits, axis=-1).astype(jnp.int32)
                target_seq_updated = target_seq_prev_step.at[..., 2 + current_token_idx_to_predict].set(predicted_token)
                
                # New carry for next step remains the same for fixed_input_seq and fixed_context_col
                new_carry = (target_seq_updated, fixed_input_seq, fixed_context_col)
                return new_carry, None # (new_carry, per_step_output)


        for t in range(matrix_size_cols):
            context_col = context_matrix[..., :, t]

            # Prepare input sequence for this recurrent step 't'
            # For generation, current_input_grid is the output from step t-1
            flattened_current_grid_for_step_t = jnp.reshape(current_input_grid, (*current_input_grid.shape[:-2], -1))
            # input_grid_shape here should be the original task's input shape,
            # as the shape prediction part predicts the *output* shape based on this.
            current_input_seq_for_step_t = jnp.concatenate([input_grid_shape, flattened_current_grid_for_step_t], axis=-1)

            # --- Predict Output Shape for this step 't' ---
            target_seq_so_far_for_shape = jnp.zeros(current_input_seq_for_step_t.shape[:-1] + (max_len + 2,), dtype=jnp.int32)

            def predict_shape_token(target_seq, is_row_token, current_input_seq, context_col_for_shape, dropout_eval_for_shape):
                # self.decoder is accessible because _generate_output_from_context_v2 is a method of LPN
                row_logits, col_logits, _ = self.decoder(
                    current_input_seq, target_seq, context_col_for_shape, dropout_eval_for_shape
                )
                logits = row_logits if is_row_token else col_logits
                predicted_token = jnp.argmax(logits, axis=-1).astype(jnp.int32) + 1
                token_index = 0 if is_row_token else 1
                target_seq = target_seq.at[..., token_index].set(predicted_token)
                return target_seq

            target_seq_so_far_for_shape = predict_shape_token(
                target_seq_so_far_for_shape, True, current_input_seq_for_step_t, context_col, dropout_eval
            )
            target_seq_so_far_for_shape = predict_shape_token(
                target_seq_so_far_for_shape, False, current_input_seq_for_step_t, context_col, dropout_eval
            )
            predicted_output_shape_for_this_step_t = target_seq_so_far_for_shape[..., :2] # Shape predicted at step t

            # --- Autoregressive Grid Token Prediction for step 't' ---
            # Initial carry for the inner scan
            initial_carry_for_scan = (target_seq_so_far_for_shape, current_input_seq_for_step_t, context_col)
            
            # Instantiate DecoderStep for nn.scan by passing its *type* and *static_args*
            # `nn.scan` will then instantiate it correctly within the compact scope.
            # `self.decoder` is the main decoder instance from the LPN model.
            scan_module_constructor = nn.scan(
                DecoderStep, # Pass the class type
                variable_broadcast="params", # Parameters of self.decoder are shared
                split_rngs={"params": False, "dropout": False}, # Dropout is handled by dropout_eval
                length=max_len
                # REMOVED: name=f"decoder_scan_t{t}" -> This was incorrect here
            )
            
            # Construct the module that will be scanned using the arguments for DecoderStep's __init__
            # These are (decoder_to_use, fixed_dropout_eval)
            scanned_decoder_step = scan_module_constructor(self.decoder, dropout_eval) 
            
            # Run the scan
            final_scan_carry, _ = scanned_decoder_step(initial_carry_for_scan, jnp.arange(max_len)) # (initial_carry, xs)
            final_target_seq = final_scan_carry[0] # The first element of the carry tuple is the updated target_seq

            predicted_grid_tokens = final_target_seq[..., 2:]
            predicted_output_grid = jnp.reshape(predicted_grid_tokens,
                                                (*predicted_grid_tokens.shape[:-1], max_rows, max_cols))

            # Update current_input_grid for the next outer loop iteration (t+1)
            current_input_grid = predicted_output_grid
            
            # The overall final shape is taken from the prediction of the *last* recurrent step
            if t == matrix_size_cols - 1:
                 final_predicted_shape_overall = predicted_output_shape_for_this_step_t

            if save_intermediate:
                intermediate_outputs[t] = {
                    "grid": current_input_grid,
                    # Shape saved is the one predicted at this step t, which would inform step t+1 if shape was evolving
                    "shape": predicted_output_shape_for_this_step_t,
                    "context_col": context_col,
                }

        # final_output_grids is current_input_grid after the loop
        # final_output_shapes is the shape predicted in the *last* step t
        return current_input_grid, final_predicted_shape_overall, intermediate_outputs if save_intermediate else None

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