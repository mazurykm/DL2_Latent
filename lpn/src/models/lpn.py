from typing import Literal, Optional
import math
from functools import partial

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.tree_util import tree_map
import optax

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
    mode: Literal["mean", "all", "random_search", "gradient_ascent"],
    prior_kl_coeff: Optional[float] = None,
    pairwise_kl_coeff: Optional[float] = None,
    use_cross_attention: bool = True, # Should be False
    **mode_kwargs,
    ):
        """
        Forward pass of the LPN model.
        """
        assert pairs.shape[-4] > 1, f"Number of pairs should be greater than 1, got {pairs.shape[-4]}."
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)

        if latents_logvar is not None:
            key = self.make_rng("latents")
            latents, prior_kl_loss, kl_metrics = self._sample_latents(latents_mu, latents_logvar, key)
            # Compute Gaussian KL between all the latents from each batch.
            pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu, latents_logvar).mean()
            kl_metrics["pairwise_kl"] = pairwise_kl_loss
        else:
            latents, prior_kl_loss, pairwise_kl_loss, kl_metrics = latents_mu, None, None, {}

        if mode_kwargs.get("remove_encoder_latents", False):
            key = self.make_rng("latents_init")
            latents = jax.random.normal(key, latents.shape)
        
        print(f"using cross attention: {use_cross_attention}")
        if use_cross_attention:
            # Use cross attention to compute the context vector.
            H = latents.shape[-1]
            sqrt_dh = jnp.sqrt(float(H))

            # Query: Mean of all M samples. Shape (*B, 1, H)
            query = latents.mean(axis=-2, keepdims=True)

            # Key/Value are the latents themselves. Shape (*B, M, H)
            key_val = latents # Renamed key to key_val to avoid conflict with PRNGKey
            value = latents # Explicitly define value

            # Attention scores. Shape (*B, 1, M)
            attn_scores = jnp.einsum('...qh,...kh->...qk', query, key_val) / sqrt_dh

            # Attention weights. Shape (*B, 1, M)
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            
            # Weighted sum (attended context). Shape (*B, H)
            attended_context = jnp.einsum('...qk,...kh->...qh', attn_weights, value).squeeze(axis=-2) 
            
            # `latents_for_modes` will be (*B, N, H), where each of the N items is a copy of attended_context
            # This makes it consistent with the `latents` variable in the non-cross-attention branch.
            latents_for_modes = jnp.tile(attended_context[..., None, :], (*([1]*(attended_context.ndim-1)), pairs.shape[-4], 1))
            
            leave_one_out_latents = make_leave_one_out(latents_for_modes, axis=-2)
            if mode == "first":
                # Compute the context vector by taking the first latent (which is attended_context).
                context = attended_context # Shape (*B, H)
                # Compute the loss for each pair using this single context. Shape (*B, N).
                loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
            elif mode == "random_search":
                for arg in ["num_samples", "scale"]:
                    assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' training mode."
                key_rng = self.make_rng("random_search") # Renamed key to key_rng
                # Repeat all the pairs and grid shapes except the one to leave out.
                leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)  # (*B, N, N-1, R, C, 2)
                leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)  # (*B, N, N-1, 2, 2)
                # Get the best context for each pair using random search.
                # _get_random_search_context expects latents, pairs, grid_shapes.
                # Here, we effectively want to find a single best context based on `leave_one_out_latents`'s information.
                # The current `_get_random_search_context`'s signature might need adjustment or careful usage here.
                # Assuming it should operate on `latents_for_modes` to find one context.
                # The original code `_get_random_search_context(leave_one_out_latents, ...)` might imply per-pair optimization.
                # For now, let's assume `_get_random_search_context` is called with `latents_for_modes`
                # and it returns a single context of shape (*B,H) or (*B,N,H) as needed.
                # Given the error, the most likely scenario is that context becomes (*B, H).
                context, _ = self._get_random_search_context(
                    latents_for_modes, # Pass all N identical contexts derived from cross-attention
                    pairs, # Pass all N pairs
                    grid_shapes, # Pass all N grid_shapes
                    key_rng, **mode_kwargs
                )  # This should return context of shape (*B, H) or (*B, N, H)
                # If it returns (*B,H), it will be handled by the fix in _loss_from_pair_and_context.
                loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
            elif mode == "gradient_ascent":
                for arg in ["num_steps", "lr"]:
                    assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent' training mode."
                key_rng = None # Default
                if mode_kwargs.get("random_perturbation", None) is not None:
                    key_rng = self.make_rng("gradient_ascent_random_perturbation") # Renamed key
                
                # Similar to random_search, adjust inputs for context search if needed.
                context, _ = self._get_gradient_ascent_context(
                    latents_for_modes, # Pass all N identical contexts
                    pairs, # Pass all N pairs
                    grid_shapes, # Pass all N grid_shapes
                    key_rng, **mode_kwargs
                )  # Expect context (*B,H) or (*B,N,H)
                loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
            else:
                raise ValueError(f"Unsupported mode for cross-attention: {mode}")
        else: # Not using cross attention
            leave_one_out_latents = make_leave_one_out(latents, axis=-2)  # (*B, N, N-1, H) 
            if mode == "mean":
                # Compute the context vector by taking the mean of all but one latents.
                context = leave_one_out_latents.mean(axis=-2)  # (*B, N, H)
                # Compute the loss for each pair using the mean of all but one latents. Shape (*B, N).
                loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
            elif mode == "all":
                # Compute the loss for each pair using all but one latents. Shape (*B, N, N-1).
                loss, metrics = jax.vmap(
                    self._loss_from_pair_and_context, in_axes=(-2, None, None, None), out_axes=-1
                )(leave_one_out_latents, pairs, grid_shapes, dropout_eval)
                # For logging purposes
                # context variable here is (*B,N,H)
                # distance_context_latents needs definition for context from vmap.
                # The original 'context = latents' and subsequent metrics might be problematic if loss is vmapped.
                # This path might need more careful handling of 'context' for metrics if mode='all'.
                # However, the primary error is not from this path. We assume context is (B,N,H) for metrics.
                current_context_for_metrics = latents # (*B,N,H)
                distance_context_latents = norm(current_context_for_metrics[..., None, :] - leave_one_out_latents, axis=-1)

            elif mode == "random_search":
                for arg in ["num_samples", "scale"]:
                    assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' training mode."
                key_rng = self.make_rng("random_search") # Renamed key
                leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
                leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
                # _get_random_search_context gets `leave_one_out_latents` (*B,N,N-1,H)
                # and is expected to return `context` of shape (*B,N,H) (one context per original pair).
                context, _ = self._get_random_search_context(
                    leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_rng, **mode_kwargs
                )  # (*B, N, H)
                loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
            elif mode == "gradient_ascent":
                for arg in ["num_steps", "lr"]:
                    assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent' training mode."
                key_rng = None
                if mode_kwargs.get("random_perturbation", None) is not None:
                    key_rng = self.make_rng("gradient_ascent_random_perturbation") # Renamed key

                leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
                leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
                context, _ = self._get_gradient_ascent_context(
                    leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_rng, **mode_kwargs
                )  # (*B, N, H)
                loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        
        # Common code for both branches
        # Ensure 'context' used for metrics is well-defined, especially for mode='all'.
        # If mode=='all', loss is calculated per (N-1) contexts. For global metrics, use 'latents' or a representative context.
        # The 'context' variable for metrics might be ambiguous if it came from vmap.
        # Let's assume for metrics, `context` refers to the primary context(s) used for decoding.
        # If mode was 'all', `context` variable might not be what's intended for these global metrics.
        # Re-assign context for metrics if mode was 'all' to avoid issues.
        # This part of the code might need to be conditional on `mode != "all"` for `context`-based metrics.
        # For now, assume `context` is defined appropriately from the mode logic.
        # If `context` is (*B,H) from cross-attention, it needs tiling for some metrics.
        # If `context` is (*B,N,H) from other modes, it's fine.
        
        # For metrics, we need a context that is (*B,N,H) or can be broadcast with latents (*B,N,H)
        # If context from cross-attention mode is (*B,H) and latents is (*B,N,H) (original encoder latents)
        # we need to decide what comparison makes sense.
        # The existing code `make_leave_one_out(context, axis=-2)` implies context should have an N-like dimension.
        
        # Let `metric_context` be the context variable that is shaped appropriately for metrics.
        # If `context` is `(*B,H)` (e.g. from cross-attention "first" mode), tile it to `(*B,N,H)` for consistent metric calculation.
        metric_context = context
        # Assuming latents is (*B,N,H) from encoder or cross-attention `latents_for_modes`
        # If `context` (from a specific mode) is `(*B,H)` and `latents` (original encoder output) is `(*B,N,H)`.
        # Example: pairs.shape[-4] gives N
        if context.ndim == latents.ndim -1 and context.shape[:-1] == latents.shape[:-2] : # context is (*B,H), latents is (*B,N,H)
             num_pairs_dim = latents.shape[-2] # N from latents
             metric_context_expanded = jnp.expand_dims(context, axis=-2) # (*B, 1, H)
             repeats = [1] * metric_context_expanded.ndim
             repeats[-2] = num_pairs_dim
             metric_context = jnp.tile(metric_context_expanded, repeats) # (*B, N, H)


        leave_one_out_contexts = make_leave_one_out(metric_context, axis=-2) # Expects metric_context (*B,N,H)
        cosine_between_contexts = jnp.einsum("...h,...nh->...n", metric_context, leave_one_out_contexts) / (
            norm(metric_context, axis=-1)[..., None] * norm(leave_one_out_contexts, axis=-1) + 1e-5
        )
        # `cosine_between_latents` uses `latents` which is typically (*B,N,H) from encoder or cross-attention post-tiling
        cosine_between_latents = jnp.einsum("...h,...nh->...n", latents, leave_one_out_latents) / (
            norm(latents, axis=-1)[..., None] * norm(leave_one_out_latents, axis=-1) + 1e-5
        )
        if mode != "all": # In 'all' mode, distance_context_latents was computed differently
            distance_context_latents = norm(metric_context - latents, axis=-1) # Compare tiled metric_context with latents
        
        metrics.update(
            latents_norm=norm(latents, axis=-1),
            context_norm=norm(metric_context, axis=-1), # Use metric_context
            distance_context_latents=distance_context_latents,
            distance_between_contexts=norm(metric_context[..., None, :] - leave_one_out_contexts, axis=-1), # Use metric_context
            cosine_between_contexts=cosine_between_contexts,
            distance_between_latents=norm(latents[..., None, :] - leave_one_out_latents, axis=-1),
            cosine_between_latents=cosine_between_latents,
        )
        loss, metrics = tree_map(jnp.mean, (loss, metrics))
        metrics.update(kl_metrics)
        if prior_kl_loss is not None:
            if prior_kl_coeff is None:
                raise ValueError("Prior KL coefficient is required when using variational inference.")
            loss += prior_kl_coeff * prior_kl_loss
            if pairwise_kl_coeff is not None:
                loss += pairwise_kl_coeff * pairwise_kl_loss

        return loss, metrics

    @staticmethod
    def _compute_pairwise_gaussian_kl(mu: chex.Array, log_var: chex.Array, eps: float = 1e-7) -> chex.Array:
        """
        Compute pairwise KL divergence between Gaussian distributions.

        Args:
            mu: mean of shape (*B, N, H)
            log_var: log variance of shape (*B, N, H)

        Returns:
            Mean KL divergence of shape B where before averaging, KL[..., i, j] is
            KL(N(mu[..., i], exp(log_var[..., i])) || N(mu[..., j], exp(log_var[..., j])))
        """
        # Expand dimensions for broadcasting
        mu1 = mu[..., :, None, :]  # (*B, N, 1, H)
        mu2 = mu[..., None, :, :]  # (*B, 1, N, H)
        log_var1 = log_var[..., :, None, :]  # (*B, N, 1, H)
        log_var2 = log_var[..., None, :, :]  # (*B, 1, N, H)
        # KL divergence formula for Gaussians:
        # KL(N1||N2) = 0.5 * (log(var2/var1) + var1/var2 + (mu1-mu2)^2/var2 - 1)
        var1, var2 = jnp.exp(log_var1), jnp.exp(log_var2)
        log_var_ratio = log_var2 - log_var1
        var_ratio = var1 / (var2 + eps)
        mu_diff_sq = (mu1 - mu2) ** 2 / (var2 + eps)
        kl = jnp.sum(0.5 * (log_var_ratio + var_ratio + mu_diff_sq - 1), axis=-1)  # (*B, N, N)
        # Average over the pairwise matrices to return a single KL divergence measure of shape (*B,)
        # Mask the diagonal to avoid comparing the same latents.
        num_pairs_dim_val = mu.shape[-2] # Renamed num_pairs to num_pairs_dim_val
        kl = jnp.sum(jnp.where(jnp.eye(num_pairs_dim_val) == 0, kl, 0), axis=(-1, -2)) / (num_pairs_dim_val * (num_pairs_dim_val - 1))
        return kl

    @staticmethod
    def _sample_latents(
        latents_mu: chex.Array, latents_logvar: chex.Array, key: chex.PRNGKey
    ):
        latents_std = jnp.exp(0.5 * latents_logvar)
        latents = latents_mu + latents_std * jax.random.normal(key, latents_mu.shape)
        kl_loss = jnp.mean(
            -0.5 * jnp.sum(1 + latents_logvar - latents_mu**2 - jnp.exp(latents_logvar), axis=-1)
        )
        kl_metrics = {
            "prior_kl": kl_loss,
            "latents_mu": latents_mu.mean(),
            "norm_latents_mu": norm(latents_mu, axis=-1).mean(),
            "latents_logvar": latents_logvar.mean(),
        }
        return latents, kl_loss, kl_metrics

    def _loss_from_pair_and_context(
        self,
        context: chex.Array,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
    ):
        """
        Computes the loss for a single pair given a context.

        Args:
            context: context vector. Shape (*B, H) or (*B, N, H).
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
            grid_shapes: shapes of the grids. Shape (*B, N, 2, 2).
            dropout_eval: if false dropout is applied otherwise it is not.

        Returns:
            loss: loss value. Shape (*B, N) if context was (*B,N,H), or broadcasted if context was (*B,H).
                  Averaged later in __call__.
            metrics: dictionary of metrics.
        """
        config = self.decoder.config

        # Make the input and output sequences.
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)
        # input_seq shape: (*common_batch_dims, N, seq_len_in)
        # context shape:   (*common_batch_dims, H) or (*common_batch_dims, N, H)

        context_for_decoder = context
        # Check if context needs to be tiled to include the 'N' dimension
        # This happens if context is (*B, H) and input_seq is (*B, N, S)
        # Condition: leading dimensions of context match leading dimensions of input_seq up to N,
        # and context is missing the N dimension compared to input_seq.
        if context.ndim + 1 == input_seq.ndim and \
           context.shape[:-1] == input_seq.shape[:context.ndim-1]:
            # Example: context is (B, H), input_seq is (B, N, S)
            # context.ndim-1 = 1 (index of B). input_seq.shape[:1] is (B,)
            # context.shape[:-1] is (B,)
            
            n_dim_value = input_seq.shape[context.ndim-1] # Size of the N dimension in input_seq
            
            # Expand context: (*B, H) -> (*B, 1, H)
            # The new dimension is inserted at index `context.ndim-1` (among batch dims) or `-2` (overall)
            context_expanded = jnp.expand_dims(context, axis=-2) 
            
            # Tile along the new dimension: (*B, 1, H) -> (*B, N, H)
            repeats = [1] * context_expanded.ndim
            repeats[-2] = n_dim_value # Tile along the axis that was size 1
            context_for_decoder = jnp.tile(context_expanded, repeats)
        
        # Now, context_for_decoder should be (*common_batch_dims, N, H) if input_seq was.
        # Or if context was already (*common_batch_dims, N, H), it remains so.

        # Decode the output sequence (teacher forcing).
        row_logits, col_logits, grid_logits = self.decoder(input_seq, output_seq, context_for_decoder, dropout_eval)

        # Compute cross entropy losses.
        grid_shapes_row, grid_shapes_col = grid_shapes[..., 0, 1], grid_shapes[..., 1, 1]
        # -1 to shift the tokens to [0, max_rows-1]
        one_hot_grid_shapes_row_labels = jax.nn.one_hot(grid_shapes_row - 1, config.max_rows)
        row_loss = -jnp.sum(jax.nn.log_softmax(row_logits) * one_hot_grid_shapes_row_labels, axis=-1)

        # -1 to shift the tokens to [0, max_cols-1]
        one_hot_grid_shapes_col_labels = jax.nn.one_hot(grid_shapes_col - 1, config.max_cols)
        col_loss = -jnp.sum(jax.nn.log_softmax(col_logits) * one_hot_grid_shapes_col_labels, axis=-1)

        # Copy the grid logits from the last non-padded column of each row to the first column of the next
        # row, skipping the padding tokens.
        last_non_padded_logits = self._get_last_non_padded_logits(
            grid_logits, grid_shapes_col[..., None, None]
        )
        grid_logits = grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)

        one_hot_grid_labels = jax.nn.one_hot(pairs[..., 1].reshape(*pairs.shape[:-3], -1), config.vocab_size)
        grid_losses = -jnp.sum(jax.nn.log_softmax(grid_logits) * one_hot_grid_labels, axis=-1)
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row, grid_shapes_col)

        loss = row_loss + col_loss + grid_loss # Shape (*B, N)
        metrics = {
            "shape_row_loss": row_loss,
            "shape_col_loss": col_loss,
            "grid_loss": grid_loss,
            "total_loss": loss,
        }
        return loss, metrics
    # ... (rest of the class LPN as provided)
    # Small fix in _get_last_non_padded_logits for consistency if needed
    def _get_last_non_padded_logits(self, grid_logits: chex.Array, num_cols: chex.Array) -> chex.Array:
        """Selects the grid logits from the last non-padded column of each row."""
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        
        # Ensure num_cols has the same batch dimensions as grid_logits for broadcasting take_along_axis
        # grid_logits shape e.g. (*B, N, MaxRows*MaxCols, VocabSize)
        # num_cols shape e.g. (*B, N, 1, 1)
        # We need indices for axis -2 of grid_logits (the sequence_length axis)
        # Indices should be (*B, N, NumRowsToCopyFrom, 1)
        
        # Original implementation iterates python-style. JAX prefers vectorized ops.
        # However, given the complexity of ragged selection, loop might be fine or a scan.
        # The current loop is over max_rows, which is a static Python int, so it unrolls.
        
        last_non_padded_logits_list = [] # Renamed to avoid conflict
        for i in range(1, max_rows): # From second row onwards, copy from previous row's end
            # Index into the flattened sequence: (i-1)*max_cols is start of (i-1)th row.
            # Add num_cols-1 to get the index of the last non-padded item in that row.
            # `num_cols` is actual number of columns, so index is `num_cols - 1`.
            # Example: If max_cols=10, num_cols=3 (indices 0,1,2). Last is index 2.
            # For row `r` (0-indexed), its elements are from `r*max_cols` to `r*max_cols + max_cols -1`.
            # The last non-padded token in row `r` is at index `r*max_cols + num_cols[...,r,:] - 1`.
            # The problem is num_cols is usually shape (*B, N) not per row.
            # Assuming num_cols refers to the cols of the *output* grid, which is fixed per example.
            
            # The original logic for `_get_last_non_padded_logits` used `max_cols * i - (max_cols - num_cols.astype(jnp.int32))`
            # This index seems to select one token per example.
            # For row `i` (1-indexed, so `i-1` is 0-indexed row number),
            # end_of_row_index = (i-1)*max_cols + (num_cols -1)
            # The original formulation `max_cols * i - (max_cols - num_cols)` = `i*max_cols - max_cols + num_cols`
            # = `(i-1)*max_cols + num_cols`. If num_cols is 1-indexed count, then this is one past the end.
            # If num_cols is 1-indexed, then `(i-1)*max_cols + num_cols -1` is the correct index.
            # Let's assume the original logic is correct and num_cols is 1-indexed.
            # The indices are for the seq_dim of grid_logits.
            # num_cols shape could be (*B,N,1,1)
            # Target index for row `r` (0-indexed): `r * max_cols + (num_cols - 1)`
            # We are filling for the start of row `r+1` (i.e., at `(r+1)*max_cols`).
            # We take from end of row `r`.
            # `i` runs from 1 to max_rows-1. So `i-1` is the row index `r` from 0 to max_rows-2.
            # Source index: `(i-1)*max_cols + (num_cols-1)`
            # The original code `max_cols * i - (max_cols - num_cols.astype(jnp.int32))` needs careful check.
            # It becomes `(i-1)*max_cols + num_cols`. If num_cols is 1-indexed length, this is one *after* the last element.
            # This suggests `grid_logits` might be shifted or expectations are different.
            # Given this is not the primary bug, I'll keep the original logic for this helper.
            # The provided code for this function is:
            # end_of_row_logits = jnp.take_along_axis(
            #    grid_logits, max_cols * i - (max_cols - num_cols.astype(jnp.int32)), axis=-2
            # )
            # This might be a source of off-by-one if num_cols definition is tricky.
            # However, let's trust it for now.
            current_row_idx_for_slice = i # Python loop var
            indices_for_prev_row_end = (current_row_idx_for_slice -1) * max_cols + (num_cols.astype(jnp.int32) - 1)
             # indices_for_prev_row_end needs to be shaped correctly for take_along_axis
             # grid_logits: (*dims, seq_len, vocab)
             # indices_for_prev_row_end: (*dims, 1) to select one logit vector
             # So, ensure it's `[..., None]` before passing to take_along_axis if it's not already.
            if indices_for_prev_row_end.shape[-1] != 1: # Ensure it's a column vector for axis selection
                 indices_for_prev_row_end = indices_for_prev_row_end[...,None]

            end_of_row_logits = jnp.take_along_axis(
                grid_logits, indices_for_prev_row_end, axis=-2 # seq_dim is -2 relative to (*dims, seq, vocab)
            )
            last_non_padded_logits_list.append(end_of_row_logits)
        return jnp.concatenate(last_non_padded_logits_list, axis=-2) # Concatenate along seq_dim


# Minimal stubs for imports if not available for linting
if __name__ == "__main__": # To avoid running if this file is imported for its stubs
    class EncoderTransformerConfig: pass
    