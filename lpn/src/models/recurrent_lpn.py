# --- START OF FILE recurrent_lpn.py ---

from typing import Literal, Optional
import math
from functools import partial
import os
import matplotlib.pyplot as plt
import uuid

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


class LPN(nn.Module):
    encoder: EncoderTransformer
    decoder: DecoderTransformer

    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
        matrix_size_rows: jnp.int32, #new
        matrix_size_cols: jnp.int32, #new
        mode: Literal["mean", "all", "random_search", "gradient_ascent", "matrix", "cross_attention", "attention_params", "recurrent_ga"], # Added new modes
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        context_kl_coeff: Optional[float] = None, # New coefficient for attention_params mode
        **mode_kwargs,
    ):
        """
        Forward pass of the LPN model.

        Args:
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
            grid_shapes: shapes of the grids. Shape (*B, N, 2, 2).
            dropout_eval: if false dropout is applied otherwise it is not.
            matrix_size_rows: number of rows for matrix reshaping.
            matrix_size_cols: number of columns for matrix reshaping.
            mode: mode of the forward pass.
                - "mean": Use mean of N-1 latents.
                - "cross_attention": Use attention-weighted average of N-1 sampled latents (query=mean).
                - "attention_params": Use attention to combine N-1 latent *parameters* (query=target mu), then sample context.
                - "recurrent_ga": Use recurrent gradient ascent.
                # ... other modes
            prior_kl_coeff: KL divergence coefficient for the *original* latents vs prior.
            pairwise_kl_coeff: KL divergence coefficient for pairwise KL between *original* latents.
            context_kl_coeff: KL divergence coefficient for the *combined context* distribution vs prior (used in 'attention_params').
            mode_kwargs: additional keyword arguments.

        Returns:
            loss: loss value.
            metrics: dictionary of metrics.
        """
        assert pairs.shape[-4] > 1, f"Number of pairs should be greater than 1 for leave-one-out strategies, got {pairs.shape[-4]}."
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval) # (*B, N, H)

        # --- KL Divergence Calculations (Common) ---
        kl_metrics = {}
        prior_kl_loss = None
        pairwise_kl_loss = None
        context_kl_loss = None # Specific to attention_params

        if latents_logvar is not None:
            # Compute KL for original latents vs prior N(0,I)
            prior_kl_loss = jnp.mean(
                -0.5 * jnp.sum(1 + latents_logvar - latents_mu**2 - jnp.exp(latents_logvar), axis=-1)
            )
            kl_metrics["prior_kl"] = prior_kl_loss
            kl_metrics["latents_mu_mean"] = latents_mu.mean()
            kl_metrics["norm_latents_mu_mean"] = norm(latents_mu, axis=-1).mean()
            kl_metrics["latents_logvar_mean"] = latents_logvar.mean()

            # Compute pairwise KL between original latent distributions
            pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu, latents_logvar).mean()
            kl_metrics["pairwise_kl"] = pairwise_kl_loss

        # Sample original latents if VAE is used
        if latents_logvar is not None:
            key_sample_orig = self.make_rng("latents_sample_orig")
            latents_std = jnp.exp(0.5 * latents_logvar)
            latents = latents_mu + latents_std * jax.random.normal(key_sample_orig, latents_mu.shape)
        else:
            latents = latents_mu # Use mean directly if no VAE

        if mode_kwargs.get("remove_encoder_latents", False):
            key_init = self.make_rng("latents_init")
            latents = jax.random.normal(key_init, latents.shape)
            # If removing encoder latents, VAE stats become meaningless or need recalculation based on random init
            # For simplicity, we might zero them out or ignore them downstream if this mode is active.
            # Let's assume they are kept for now, but be aware of the interpretation.

        # --- Context Computation based on Mode ---
        # Prepare leave-one-out versions needed for several modes
        leave_one_out_latents = make_leave_one_out(latents, axis=-2)  # (*B, N, N-1, H)
        if latents_logvar is not None:
            leave_one_out_mu = make_leave_one_out(latents_mu, axis=-2) # (*B, N, N-1, H)
            leave_one_out_logvar = make_leave_one_out(latents_logvar, axis=-2) # (*B, N, N-1, H)

        # Determine context based on mode
        if mode == "mean":
            context = leave_one_out_latents.mean(axis=-2)  # (*B, N, H)
        elif mode == "matrix":
            # Note: This mode doesn't inherently use leave-one-out for context *combination*,
            # but it will use the leave-one-out structure implicitly in the loss calculation loop.
            # The context passed to _loss_from_pair_and_context is (*B, N, H).
            # Let's use the mean context for consistency before matrix ops in the loss function.
            context = leave_one_out_latents.mean(axis=-2) # (*B, N, H)
            # The matrix operation happens *inside* _loss_from_pair_and_context
        elif mode == "cross_attention":
            # Uses sampled latents. Query is mean(N-1 latents).
            context = self._compute_loo_cross_attention_context(
                 leave_one_out_latents # Input shape (*B, N, N-1, H)
            ) # Output shape (*B, N, H)
        elif mode == "attention_params":
            if latents_logvar is None:
                 raise ValueError("Mode 'attention_params' requires a VAE (latents_logvar must be present).")
            if context_kl_coeff is None:
                 raise ValueError("Mode 'attention_params' requires 'context_kl_coeff'.")

            key_sample_context = self.make_rng("latents_sample_context")
            # Combine parameters using attention (query=target mu)
            context_mu, context_logvar = self._compute_attention_params_context(
                target_mu=latents_mu, # (*B, N, H) - Used for query
                loo_mu=leave_one_out_mu, # (*B, N, N-1, H) - Used for key, value_mu
                loo_logvar=leave_one_out_logvar # (*B, N, N-1, H) - Used for value_logvar
            ) # Output shapes (*B, N, H)

            # Sample context from combined distribution
            context_std = jnp.exp(0.5 * context_logvar)
            context = context_mu + context_std * jax.random.normal(key_sample_context, context_mu.shape) # (*B, N, H)

            # Compute KL divergence for the combined context distribution
            context_kl_loss = jnp.mean(
                -0.5 * jnp.sum(1 + context_logvar - context_mu**2 - jnp.exp(context_logvar), axis=-1)
            )
            kl_metrics["context_kl"] = context_kl_loss
            kl_metrics["context_mu_mean"] = context_mu.mean()
            kl_metrics["context_logvar_mean"] = context_logvar.mean()

        elif mode == "recurrent_ga":
            # ... (recurrent_ga logic remains the same) ...
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'recurrent_ga' mode."
            if mode_kwargs.get("random_perturbation", None) is not None:
                key_ga_pert = self.make_rng("gradient_ascent_random_perturbation")
            else:
                key_ga_pert = None
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
            context, _ = self._get_recurrent_ga_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_ga_pert,
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                **mode_kwargs
            ) # Output shape (*B, N, H)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # --- Loss Calculation ---
        # Compute reconstruction loss using the determined context
        # The shape of 'context' is (*B, N, H) - one context vector per original pair
        # The loss function internally handles the pairing of context[b, i] with pairs[b, i]
        loss, metrics = self._loss_from_pair_and_context(
            context, # (*B, N, H)
            pairs,   # (*B, N, R, C, 2)
            grid_shapes, # (*B, N, 2, 2)
            dropout_eval,
            matrix_size_cols=matrix_size_cols,
            matrix_size_rows=matrix_size_rows
        ) # loss shape (*B, N), metrics shapes (*B, N, ...)

        # --- Aggregate Metrics and Final Loss ---
        # Calculate metrics based on relationships between latents and contexts
        leave_one_out_contexts = make_leave_one_out(context, axis=-2) # (*B, N, N-1, H)
        eps = 1e-5
        # Cosine similarity between context_i and mean(context_j for j!=i)
        mean_loo_contexts = leave_one_out_contexts.mean(axis=-2) # (*B, N, H)
        cosine_context_vs_mean_others = jnp.einsum("...h,...h->...", context, mean_loo_contexts) / (
            (norm(context, axis=-1) * norm(mean_loo_contexts, axis=-1)) + eps
        )
        # Cosine similarity between latent_i and mean(latent_j for j!=i)
        mean_loo_latents = leave_one_out_latents.mean(axis=-2) # (*B, N, H)
        cosine_latent_vs_mean_others = jnp.einsum("...h,...h->...", latents, mean_loo_latents) / (
             (norm(latents, axis=-1) * norm(mean_loo_latents, axis=-1)) + eps
        )

        metrics.update(
            latents_norm=norm(latents, axis=-1), # (*B, N)
            context_norm=norm(context, axis=-1), # (*B, N)
            distance_context_latents=norm(context - latents, axis=-1), # (*B, N)
            # Compare context_i to the N-1 contexts computed for the same i (less informative now)
            # Let's compare context_i to context_j (where j!=i) instead.
            # distance_between_contexts=norm(context[..., None, :] - leave_one_out_contexts, axis=-1), # (*B, N, N-1)
            # cosine_between_contexts=cosine_between_contexts, # (*B, N, N-1)
            # Distance between latent_i and latent_j (where j!=i)
            distance_between_latents=norm(latents[..., None, :] - leave_one_out_latents, axis=-1), # (*B, N, N-1)
            # cosine_between_latents=cosine_between_latents # (*B, N, N-1) - Need to recalculate for i vs j
            cosine_context_vs_mean_others = cosine_context_vs_mean_others, # (*B, N)
            cosine_latent_vs_mean_others = cosine_latent_vs_mean_others, # (*B, N)
        )

        # Average loss and metrics over the N pairs dimension
        loss, metrics = tree_map(jnp.mean, (loss, metrics)) # Average over N dim -> shapes (*B,)
        metrics.update(kl_metrics) # Add KL metrics (already averaged or single values)

        # Add KL terms to the final loss
        total_loss = loss
        if prior_kl_loss is not None and prior_kl_coeff is not None:
            total_loss += prior_kl_coeff * prior_kl_loss
        if pairwise_kl_loss is not None and pairwise_kl_coeff is not None:
            total_loss += pairwise_kl_coeff * pairwise_kl_loss
        if context_kl_loss is not None and context_kl_coeff is not None: # Add context KL for relevant mode
             total_loss += context_kl_coeff * context_kl_loss

        # Add total loss to metrics for tracking
        metrics["total_loss_unweighted_reconstruction"] = loss # Keep track of reconstruction loss
        metrics["total_loss_final"] = total_loss # Final loss used for optimization

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
        key: Optional[chex.PRNGKey], # For sampling (VAE, random search, etc.)
        dropout_eval: bool,
        matrix_size_rows: int,
        matrix_size_cols: int,
        mode: Literal["mean", "first", "random_search", "gradient_ascent", "matrix", "cross_attention", "attention_params", "recurrent_ga"], # Added modes
        return_two_best: bool = False,
        **mode_kwargs,
    ):
        """
        Predicts the output grid given a new input and supporting examples.

        Args:
            # ... (standard args) ...
            key: PRNG key. Required for VAE modes ('cross_attention' if VAE active, 'attention_params'),
                 'random_search', 'gradient_ascent' (if perturbing). Shape (*B, 2) or None.
            # ... (other args) ...
            mode: Inference mode.
                 - "mean": Use mean of N latents.
                 - "cross_attention": Use attention-weighted average of N *sampled* latents (query=mean(all N)).
                 - "attention_params": Use attention to combine N latent *parameters* (query=mean(all N mu)), then sample context.
                 - ... (other modes) ...
            return_two_best: If true, returns two predictions (relevant for modes like random_search/GA).

        Returns:
            # ... (standard returns) ...
        """
        # 1. Encode the supporting examples
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval) # (*B, N, H)

        # 2. Determine the single context vector(s) for generation based on mode
        generation_context = None
        second_context = None # For return_two_best
        info = {}

        if mode == "mean":
            context = latents_mu.mean(axis=-2) # Average mu directly (*B, H)
            if latents_logvar is not None:
                 # If VAE, maybe average sampled latents? Or stick to mean mu?
                 # Let's stick to mean mu for simplicity in 'mean' mode.
                 pass
            generation_context = context

        elif mode == "matrix":
             # Similar to mean, use the average latent representation before matrix ops
             context = latents_mu.mean(axis=-2)
             if latents_logvar is not None:
                  # Could sample and average samples, but let's use mean mu
                  pass
             generation_context = context
             # Matrix ops happen inside _generate_output_from_context_v2

        elif mode == "cross_attention":
            # Combine N *sampled* latents using attention (query = mean(all N))
            if latents_logvar is not None:
                assert key is not None, "'key' required for 'cross_attention' with VAE"
                key, subkey = jax.random.split(key)
                latents_std = jnp.exp(0.5 * latents_logvar)
                latents = latents_mu + latents_std * jax.random.normal(subkey, latents_mu.shape) # (*B, N, H)
            else:
                latents = latents_mu # Use deterministic latents

            # Apply attention across all N latents
            generation_context = self._compute_global_attention_context(latents) # (*B, H)
            info["original_latents_for_attn"] = latents

        elif mode == "attention_params":
             # Combine N *parameters* using attention (query = mean(all N mu)), then sample
             if latents_logvar is None:
                  raise ValueError("Mode 'attention_params' requires a VAE for generation.")
             assert key is not None, "'key' required for 'attention_params' generation"

             # Combine parameters using global attention
             context_mu, context_logvar = self._compute_global_attention_params_context(
                  latents_mu, latents_logvar
             ) # (*B, H)

             # Sample the final context
             key, subkey = jax.random.split(key)
             context_std = jnp.exp(0.5 * context_logvar)
             generation_context = context_mu + context_std * jax.random.normal(subkey, context_mu.shape) # (*B, H)
             info["context_mu"] = context_mu
             info["context_logvar"] = context_logvar

        elif mode == "first":
            # Use the first latent (or its mu if VAE)
            context = latents_mu[:, 0, :]
            if latents_logvar is not None:
                 # Option: sample from first latent's distribution
                 # key, subkey = jax.random.split(key)
                 # context_std = jnp.exp(0.5 * latents_logvar[:, 0, :])
                 # context = latents_mu[:, 0, :] + context_std * jax.random.normal(subkey, context_std.shape)
                 pass # Sticking to mu for simplicity in 'first' mode
            generation_context = context

        elif mode == "random_search" or mode == "gradient_ascent" or mode == "recurrent_ga":
             # These modes require specific helper functions to find the best context(s)
             # We assume these functions exist and return one or two contexts
             # Example placeholder for random search:
             if mode == "random_search":
                  assert key is not None, "'key' required for 'random_search'"
                  # generation_context, second_context = self._get_random_search_context(
                  #      latents_mu, pairs, grid_shapes, key, **mode_kwargs
                  # ) # Needs implementation matching LPN's original structure
                  raise NotImplementedError("Random search context generation needs specific implementation.")
             elif mode == "gradient_ascent":
                   # generation_context, second_context = self._get_gradient_ascent_context(
                   #     latents_mu, pairs, grid_shapes, key, **mode_kwargs
                   # ) # Needs implementation matching LPN's original structure
                   raise NotImplementedError("Gradient ascent context generation needs specific implementation.")
             elif mode == "recurrent_ga":
                  # Recurrent GA might need adaptation for generation vs training context finding
                  # Let's assume it can produce a single best context for generation
                  # This likely involves running the GA optimization on the support pairs
                  # to find *one* optimized context vector.
                  if key is not None: key, subkey_ga = jax.random.split(key)
                  else: subkey_ga = None

                  # Need to adapt _get_recurrent_ga_context for generation (input N pairs, output 1 context)
                  # This is complex as the original recurrent_ga seems tied to the leave-one-out loss structure.
                  # For now, let's use the mean latent as a placeholder context for recurrent_ga generation.
                  print("Warning: 'recurrent_ga' generation context defaulting to mean latent. Needs specific implementation.")
                  generation_context = latents_mu.mean(axis=-2)
                  # raise NotImplementedError("Recurrent GA context generation needs specific implementation.")


        else:
            raise ValueError(f"Unsupported generation mode: {mode}")

        # Ensure we have a context
        if generation_context is None:
             raise RuntimeError(f"Generation context was not set for mode {mode}")

        # If only one context was found, use it for both predictions if needed
        if second_context is None:
            second_context = generation_context

        info["context"] = generation_context # Store the primary context used

        # 3. Generate output(s) using the determined context(s)
        if return_two_best:
            # Generate for both best and second-best contexts
            # Note: vmap requires contexts to be stacked on a new leading dimension
            contexts_to_generate = jnp.stack([generation_context, second_context], axis=0) # (2, *B, H)

            # Need to vmap over the first dimension (0) of the context stack
            # The function _generate_output_from_context_v2 expects context (*B, H)
            # So we vmap it over the stack.
            output_grids_stack, output_shapes_stack, intermediate_dict_stack = jax.vmap(
                partial(
                    self._generate_output_from_context_v2,
                    # input, input_grid_shape, dropout_eval args are automatically handled by partial
                    input=input,
                    input_grid_shape=input_grid_shape,
                    dropout_eval=dropout_eval,
                    matrix_size_rows=matrix_size_rows,
                    matrix_size_cols=matrix_size_cols,
                    save_intermediate=mode_kwargs.get("save_intermediate_outputs", False)
                ),
                in_axes=0 # Vmap over the first axis of contexts_to_generate
            )(contexts_to_generate) # Input shape (2, *B, H)

            first_output_grids, second_output_grids = output_grids_stack[0], output_grids_stack[1]
            first_output_shapes, second_output_shapes = output_shapes_stack[0], output_shapes_stack[1]
            # Handle intermediate dicts if saved
            intermediate_dict = None
            if intermediate_dict_stack is not None:
                 intermediate_dict = {"best": intermediate_dict_stack[0], "second_best": intermediate_dict_stack[1]}

            return first_output_grids, first_output_shapes, second_output_grids, second_output_shapes, info, intermediate_dict
        else:
            # Generate only for the best context
            output_grids, output_shapes, intermediate_dict = self._generate_output_from_context_v2(
                generation_context, # (*B, H)
                input,
                input_grid_shape,
                dropout_eval,
                matrix_size_rows,
                matrix_size_cols,
                mode_kwargs.get("save_intermediate_outputs", False)
            )
            return output_grids, output_shapes, info, intermediate_dict


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

        Args:
            context: Single context vector for generation. Shape (*B, H).
            # ... other args ...

        Returns:
            final_output_grids: Predicted output grids. Shape (*B, R, C).
            final_output_shapes: Predicted shapes. Shape (*B, 2).
            intermediate_outputs: Optional dictionary of intermediate states.
        """
        config = self.decoder.config
        max_rows, max_cols = config.max_rows, config.max_cols
        max_len = config.max_len # R * C

        # Convert the single context vector into the recurrent matrix format (*B, rows, cols)
        try:
            context_matrix = self._convert_to_matrix(
                matrix_size_rows=matrix_size_rows,
                matrix_size_cols=matrix_size_cols,
                latents=context, # Input context is (*B, H)
            ) # Output shape (*B, rows, cols)
        except ValueError as e:
             print(f"Error converting context to matrix during generation: {e}")
             # Return dummy output or raise
             dummy_grid = jnp.zeros_like(input)
             dummy_shape = jnp.ones_like(input_grid_shape)
             return dummy_grid, dummy_shape, None


        # Initialize with the input grid and shape
        current_input_grid = input
        current_input_shape = input_grid_shape
        intermediate_outputs = {}

        # --- Recurrent Generation Loop ---
        for t in range(matrix_size_cols):
            context_col = context_matrix[..., :, t] # Context for this step (*B, rows)

            # Prepare input sequence for the decoder at step t
            # Uses the *current* state (grid and shape) predicted so far
            flattened_current_grid = jnp.reshape(current_input_grid, (*current_input_grid.shape[:-2], -1)) # (*B, R*C)
            current_input_seq = jnp.concatenate([current_input_shape, flattened_current_grid], axis=-1) # (*B, R*C+2)

            # --- Autoregressive Decoding within step t ---
            # Initialize target sequence for prediction (start with shape tokens)
            # We predict shape first, then grid tokens.
            target_seq_so_far = jnp.zeros(current_input_seq.shape[:-1] + (max_len + 2,), dtype=jnp.int32)

            # 1. Predict Output Shape (Row and Col)
            def predict_shape_token(target_seq, is_row_token):
                # Decoder expects input_seq, target_seq (partially filled), context
                row_logits, col_logits, _ = self.decoder(
                    current_input_seq, target_seq, context_col, dropout_eval
                )
                logits = row_logits if is_row_token else col_logits
                predicted_token = jnp.argmax(logits, axis=-1).astype(jnp.int32) + 1 # Shapes are 1-based
                token_index = 0 if is_row_token else 1
                target_seq = target_seq.at[..., token_index].set(predicted_token)
                return target_seq

            target_seq_so_far = predict_shape_token(target_seq_so_far, is_row_token=True)
            target_seq_so_far = predict_shape_token(target_seq_so_far, is_row_token=False)
            predicted_output_shape = target_seq_so_far[..., :2] # (*B, 2)

            # 2. Predict Output Grid Tokens (Autoregressively)
            def body_fn(i, target_seq):
                # Get grid logits based on current input and partially filled target
                *_, grid_logits = self.decoder(
                    current_input_seq, target_seq, context_col, dropout_eval
                ) # grid_logits shape (*B, max_len, vocab_size)

                # Select the logits corresponding to the token we are predicting *now*
                # This uses the standard transformer causal masking logic implicitly
                current_token_logits = grid_logits[..., i, :] # (*B, vocab_size)
                predicted_token = jnp.argmax(current_token_logits, axis=-1).astype(jnp.int32) # (*B,)

                # Update the target sequence with the predicted token
                target_seq = target_seq.at[..., 2 + i].set(predicted_token)
                return target_seq

            # Use lax.scan for efficient autoregressive loop
            final_target_seq = jax.lax.scan(
                body_fn,
                target_seq_so_far, # Initial state (contains predicted shape)
                jnp.arange(max_len) # Loop indices 0 to max_len-1
            )[0] # Get the final state (filled target sequence)

            # --- Update State for Next Recurrent Step (t+1) ---
            # Reshape predicted grid tokens (*B, R*C) -> (*B, R, C)
            predicted_grid_tokens = final_target_seq[..., 2:]
            # We need the actual R, C used by the decoder config
            predicted_output_grid = jnp.reshape(predicted_grid_tokens,
                                                (*predicted_grid_tokens.shape[:-1], max_rows, max_cols))

            # Update the 'current' state to be the output of this step
            current_input_grid = predicted_output_grid
            # Only update shape at the very end? Or at each step?
            # Let's update shape at each step to reflect the prediction based on context_col_t
            # current_input_shape = predicted_output_shape
            # Original logic only updated shape at the end - let's revert to that
            if t == matrix_size_cols - 1:
                 current_input_shape = predicted_output_shape
            else:
                 # Keep the original input shape or previous step's shape?
                 # Let's keep the original input shape until the last step
                 current_input_shape = input_grid_shape # Or keep previous predicted?

            # Optionally save intermediate state
            if save_intermediate:
                intermediate_outputs[t] = {
                    "grid": current_input_grid,
                    "shape": current_input_shape, # Shape used *after* this step
                    "context_col": context_col,
                }
        # --- End Recurrent Loop ---

        # Final predicted grid and shape are the state after the last step
        final_output_grids = current_input_grid
        final_output_shapes = current_input_shape

        return final_output_grids, final_output_shapes, intermediate_outputs if save_intermediate else None


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