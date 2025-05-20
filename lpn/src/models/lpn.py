from typing import Literal, Optional, Union # Added Union
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
from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig # Assuming these are correct
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
        use_cross_attention: bool = False, # Default to False as in original, can be overridden
        **mode_kwargs,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        
        assert pairs.shape[-4] > 1, f"Number of pairs should be greater than 1, got {pairs.shape[-4]}."
        # B: batch_size, N: num_pairs_per_program, R: rows, C: cols, H: hidden_size
        # pairs: (*B, N, R, C, 2)
        # grid_shapes: (*B, N, 2, 2)
        
        num_program_pairs = pairs.shape[-4] # This is N

        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)
        # latents_mu: (*B, N, H)

        if latents_logvar is not None:
            key_sample_latents = self.make_rng("latents")
            base_latents, prior_kl_loss, kl_metrics = self._sample_latents(latents_mu, latents_logvar, key_sample_latents)
            pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu, latents_logvar).mean()
            kl_metrics["pairwise_kl"] = pairwise_kl_loss
        else:
            base_latents, prior_kl_loss, pairwise_kl_loss, kl_metrics = latents_mu, None, None, {}
        # base_latents: (*B, N, H)

        if mode_kwargs.get("remove_encoder_latents", False):
            key_init_latents = self.make_rng("latents_init")
            base_latents = jax.random.normal(key_init_latents, base_latents.shape)

        # This `latents_for_processing` will be the basis for context derivation.
        # If CA is on, it becomes the tiled attended_context. Otherwise, it's base_latents.
        latents_for_processing = base_latents 
        # `context_for_loss_calculation` will be passed to _loss_from_pair_and_context
        # It should generally be (*B,N,H) or compatible (e.g. (*B,H) if _loss_from_pair_and_context tiles it)

        if use_cross_attention:
            print(f"LPN __call__: using cross attention: {use_cross_attention}")
            # base_latents are (*B, N, H)
            H = base_latents.shape[-1]
            sqrt_dh = jnp.sqrt(float(H))

            query = base_latents.mean(axis=-2, keepdims=True) # (*B, 1, H) Query from mean of N latents
            key_val_attn = base_latents # (*B, N, H) Key/Value are the N latents themselves
            
            attn_scores = jnp.einsum('...qh,...kh->...qk', query, key_val_attn) / sqrt_dh # (*B, 1, N)
            attn_weights = jax.nn.softmax(attn_scores, axis=-1) # (*B, 1, N)
            
            # attended_context: (*B, H) - a single context vector per batch item
            attended_context = jnp.einsum('...qk,...kh->...qh', attn_weights, key_val_attn).squeeze(axis=-2)

            # For subsequent logic that expects N latents/contexts, tile the attended_context.
            # latents_for_processing becomes (*B, N, H), where each of N items is attended_context.
            leading_dims = attended_context.shape[:-1] # (*B)
            tile_repeats = [1] * len(leading_dims) + [num_program_pairs, 1] 
            latents_for_processing = jnp.tile(attended_context[..., None, :], tile_repeats)
            
            # For "leave-one-out" with CA, since all N latents in latents_for_processing are identical,
            # make_leave_one_out(...) will result in (N-1) copies of the same attended_context.
            leave_one_out_latents = make_leave_one_out(latents_for_processing, axis=-2) # (*B, N, N-1, H)
            
            if mode == "mean":
                # Mean of (N-1) identical contexts is just the context itself.
                # Resulting context_for_loss_calculation: (*B, N, H), each N is attended_context
                context_for_loss_calculation = latents_for_processing 
                loss, metrics = self._loss_from_pair_and_context(context_for_loss_calculation, pairs, grid_shapes, dropout_eval)
            elif mode == "all":
                # Each of N-1 contexts is attended_context.
                # context_for_loss_calculation here is effectively leave_one_out_latents (*B,N,N-1,H)
                # This means _loss_from_pair_and_context will be vmapped over N-1 identical contexts.
                loss, metrics = jax.vmap(
                    self._loss_from_pair_and_context, in_axes=(-2, None, None, None), out_axes=-1
                )(leave_one_out_latents, pairs, grid_shapes, dropout_eval)
                # For metrics, 'context' should be consistently shaped, e.g., (*B,N,H)
                # latents_for_processing is already the tiled attended_context (*B,N,H)
                context_for_metrics_logging = latents_for_processing
            elif mode == "random_search" or mode == "gradient_ascent":
                # Search/Optimization starts from the common attended_context (or N copies of it).
                # The search operates over all N pairs to find one best refined context.
                # `leave_one_out_latents` here are N-1 copies of attended_context.
                # `leave_one_out_pairs/shapes` are the actual leave-one-out data.
                # This implies we are finding N different "best" contexts, each optimized for one held-out pair,
                # but all starting from the same global attended_context. This seems consistent with original LOO.

                leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
                leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)

                if mode == "random_search":
                    for arg in ["num_samples", "scale"]:
                        assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search'."
                    key_rs = self.make_rng("random_search")
                    # _get_random_search_context's `latents` arg is LOO latents.
                    # Here, it's `leave_one_out_latents` which are N-1 copies of attended_context.
                    # It will return refined_contexts: (*B, N, H)
                    refined_contexts, _ = self._get_random_search_context(
                        leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_rs, **mode_kwargs
                    )
                else: # gradient_ascent
                    for arg in ["num_steps", "lr"]:
                        assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent'."
                    key_ga = self.make_rng("gradient_ascent_random_perturbation") if mode_kwargs.get("random_perturbation") else None
                    refined_contexts, _ = self._get_gradient_ascent_context(
                        leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_ga, **mode_kwargs
                    )
                context_for_loss_calculation = refined_contexts # (*B, N, H)
                loss, metrics = self._loss_from_pair_and_context(context_for_loss_calculation, pairs, grid_shapes, dropout_eval)
            else:
                raise ValueError(f"Unsupported mode with cross-attention: {mode}")

        else: # Not using cross_attention (original logic)
            print(f"LPN __call__: using cross attention: {use_cross_attention}")
            leave_one_out_latents = make_leave_one_out(base_latents, axis=-2)
            if mode == "mean":
                context_for_loss_calculation = leave_one_out_latents.mean(axis=-2)
                loss, metrics = self._loss_from_pair_and_context(context_for_loss_calculation, pairs, grid_shapes, dropout_eval)
            elif mode == "all":
                loss, metrics = jax.vmap(
                    self._loss_from_pair_and_context, in_axes=(-2, None, None, None), out_axes=-1
                )(leave_one_out_latents, pairs, grid_shapes, dropout_eval)
                context_for_metrics_logging = base_latents # For metrics
            elif mode == "random_search" or mode == "gradient_ascent":
                leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
                leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
                if mode == "random_search":
                    for arg in ["num_samples", "scale"]:
                        assert arg in mode_kwargs, f"'{arg}' argument required."
                    key_rs = self.make_rng("random_search")
                    context_for_loss_calculation, _ = self._get_random_search_context(
                        leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_rs, **mode_kwargs
                    )
                else: # gradient_ascent
                    for arg in ["num_steps", "lr"]:
                        assert arg in mode_kwargs, f"'{arg}' argument required."
                    key_ga = self.make_rng("gradient_ascent_random_perturbation") if mode_kwargs.get("random_perturbation") else None
                    context_for_loss_calculation, _ = self._get_gradient_ascent_context(
                        leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_ga, **mode_kwargs
                    )
                loss, metrics = self._loss_from_pair_and_context(context_for_loss_calculation, pairs, grid_shapes, dropout_eval)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        # Metrics calculation
        # `context_for_metrics_logging` should be the (*B,N,H) representation of context(s) used.
        # `base_latents` is always the original (*B,N,H) from encoder+sample.
        # `latents_for_processing` is (*B,N,H) - either base_latents or tiled attended_context.
        
        # If mode was 'all', context_for_loss_calculation is not a single (*B,N,H) array.
        # So, use `context_for_metrics_logging` which was set appropriately.
        # Otherwise, `context_for_loss_calculation` is what we need for metrics.
        final_context_for_metrics = context_for_metrics_logging if mode == "all" else context_for_loss_calculation

        # `distance_context_latents` compares `final_context_for_metrics` with `base_latents`.
        # If CA is on, `final_context_for_metrics` is derived from `attended_context`.
        # `base_latents` are the original per-pair latents. This comparison is meaningful.
        
        # Ensure leave_one_out_latents (from base_latents) is defined for metrics
        # It was computed from base_latents early on if use_cross_attention=False
        # If use_cross_attention=True, it was computed from latents_for_processing (tiled attended_context)
        # For consistency in metrics like `cosine_between_latents`, we should use LOO of `base_latents`.
        loo_base_latents = make_leave_one_out(base_latents, axis=-2)

        leave_one_out_final_contexts = make_leave_one_out(final_context_for_metrics, axis=-2)
        
        cosine_between_contexts = jnp.einsum("...h,...nh->...n", final_context_for_metrics, leave_one_out_final_contexts) / (
            norm(final_context_for_metrics, axis=-1)[..., None] * norm(leave_one_out_final_contexts, axis=-1) + 1e-5
        )
        # This compares original latents among themselves
        cosine_between_base_latents = jnp.einsum("...h,...nh->...n", base_latents, loo_base_latents) / (
            norm(base_latents, axis=-1)[..., None] * norm(loo_base_latents, axis=-1) + 1e-5
        )
        
        # This measures how far the final contexts (potentially refined/attended) are from original latents.
        # If mode == 'all' and CA is on, final_context_for_metrics is tiled attended_context.
        # distance_context_latents was defined as norm(latents[..., None, :] - leave_one_out_latents, axis=-1)
        # where latents was base_latents if no CA, or latents_for_processing if CA.
        # Let's define it consistently: final_context_for_metrics vs base_latents.
        if mode == "all": # Special handling for distance_context_latents if mode is 'all'
            # Original 'all' mode computed norm(latents[..., None, :] - leave_one_out_latents, axis=-1)
            # where 'latents' was base_latents. This is distance between a latent and other original latents.
            # If CA is on, context_for_metrics_logging is latents_for_processing (tiled attended)
            # leave_one_out_latents was from latents_for_processing. So it's dist between attended and other attended (0).
            # This metric may need re-evaluation for CA + 'all' mode.
            # For now, stick to original meaning if possible, or use final_context_for_metrics vs base_latents.
            # The provided code was: distance_context_latents = norm(latents[..., None, :] - leave_one_out_latents, axis=-1)
            # where `latents` was base_latents.
            dist_ctx_lat = norm(base_latents[..., None, :] - loo_base_latents, axis=-1)
        else:
            dist_ctx_lat = norm(final_context_for_metrics - base_latents, axis=-1)

        metrics.update(
            latents_norm=norm(base_latents, axis=-1), # Norm of original latents
            context_norm=norm(final_context_for_metrics, axis=-1), # Norm of context used for loss
            distance_context_latents=dist_ctx_lat,
            distance_between_contexts=norm(final_context_for_metrics[..., None, :] - leave_one_out_final_contexts, axis=-1),
            cosine_between_contexts=cosine_between_contexts,
            distance_between_latents=norm(base_latents[..., None, :] - loo_base_latents, axis=-1), # Dist between original latents
            cosine_between_latents=cosine_between_base_latents, # Cosine between original latents
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
        if num_pairs > 1: # Avoid division by zero if N=1 (though N>1 is asserted)
            kl = jnp.sum(jnp.where(jnp.eye(num_pairs) == 0, kl, 0), axis=(-1, -2)) / (num_pairs * (num_pairs - 1))
        else:
            kl = jnp.zeros_like(kl[...,0,0])
        return kl

    @staticmethod
    def _sample_latents(
        latents_mu: chex.Array, latents_logvar: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.Array, chex.Array, dict]:
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
        context: chex.Array, # Can be (*B,H) or (*B,N,H)
        pairs: chex.Array,   # (*B,N,R,C,2)
        grid_shapes: chex.Array, # (*B,N,2,2)
        dropout_eval: bool,
    ) -> tuple[chex.Array, dict]:
        config = self.decoder.config
        input_seq, output_seq = LPN._flatten_input_output_for_decoding(pairs, grid_shapes)
        # input_seq, output_seq are (*B,N,SeqLen)

        context_for_decoder = context
        # If context is (*B,H) and input_seq is (*B,N,SeqLen), tile context
        # Check based on ndim and compatible leading batch dims & hidden dim
        if context.ndim == input_seq.ndim - 1 and \
           context.shape[:-1] == input_seq.shape[:-2] and \
           context.shape[-1] == self.decoder.config.hidden_size : # Added check for H dim
            
            num_program_pairs_dim = input_seq.shape[-2] # N from input_seq
            
            leading_dims = context.shape[:-1] # (*B)
            # tile_repeats for context (*B,H) -> (*B,N,H)
            # expand: (*B,H) -> (*B,1,H) at axis len(leading_dims)
            context_expanded = jnp.expand_dims(context, axis=len(leading_dims)) 
            
            tile_repeats_list = [1] * context_expanded.ndim
            tile_repeats_list[len(leading_dims)] = num_program_pairs_dim
            context_for_decoder = jnp.tile(context_expanded, tile_repeats_list)
        
        # Now context_for_decoder should be (*B,N,H)
        row_logits, col_logits, grid_logits = self.decoder(input_seq, output_seq, context_for_decoder, dropout_eval)
        # ... rest of the method is unchanged from your provided code ...
        grid_shapes_row, grid_shapes_col = grid_shapes[..., 0, 1], grid_shapes[..., 1, 1]
        one_hot_grid_shapes_row_labels = jax.nn.one_hot(grid_shapes_row - 1, config.max_rows)
        row_loss = -jnp.sum(jax.nn.log_softmax(row_logits) * one_hot_grid_shapes_row_labels, axis=-1)
        one_hot_grid_shapes_col_labels = jax.nn.one_hot(grid_shapes_col - 1, config.max_cols)
        col_loss = -jnp.sum(jax.nn.log_softmax(col_logits) * one_hot_grid_shapes_col_labels, axis=-1)
        last_non_padded_logits = self._get_last_non_padded_logits(
            grid_logits, grid_shapes_col[..., None, None]
        )
        if config.max_cols > 0 and grid_logits.shape[-2] > config.max_cols :
            grid_logits = grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)
        one_hot_grid_labels = jax.nn.one_hot(pairs[..., 1].reshape(*pairs.shape[:-3], -1), config.vocab_size)
        grid_losses = -jnp.sum(jax.nn.log_softmax(grid_logits) * one_hot_grid_labels, axis=-1)
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row, grid_shapes_col)
        loss = row_loss + col_loss + grid_loss
        metrics = {
            "shape_row_loss": row_loss,
            "shape_col_loss": col_loss,
            "grid_loss": grid_loss,
            "total_loss": loss,
        }
        return loss, metrics

    def _normalized_mean_over_sequence(
        self, grid_seq: chex.Array, num_rows: chex.Array, num_cols: chex.Array
    ) -> chex.Array:
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        
        # Corrected broadcasting for arange
        row_arange_b = jnp.arange(max_rows).reshape( *( (1,)*num_rows.ndim + (max_rows,)) )
        col_arange_b = jnp.arange(max_cols).reshape( *( (1,)*num_cols.ndim + (max_cols,)) )
        
        grid_row_mask = row_arange_b < num_rows[..., None]
        grid_col_mask = col_arange_b < num_cols[..., None]
        
        grid_pad_mask_2d = grid_row_mask[..., :, None] & grid_col_mask[..., None, :]
        grid_pad_mask = grid_pad_mask_2d.reshape(*grid_pad_mask_2d.shape[:-2], -1)
        
        grid_seq = jnp.where(grid_pad_mask, grid_seq, 0)
        mean_seq = jnp.sum(grid_seq, axis=-1) / (jnp.sum(grid_pad_mask, axis=-1) + 1e-5)
        return mean_seq

    def generate_output(
        self,
        pairs: chex.Array, # Context pairs (*B, N_examples, R, C, 2)
        grid_shapes: chex.Array, # (*B, N_examples, 2, 2)
        input_grid: chex.Array, # Target input for generation (*B, R, C)
        input_grid_shape: chex.Array, # (*B, 2)
        key: Optional[chex.PRNGKey], # For sampling, search, perturbation
        dropout_eval: bool,
        mode: Literal["mean", "first", "random_search", "gradient_ascent"],
        return_two_best: bool = False,
        use_cross_attention: bool = False, # Added use_cross_attention
        **mode_kwargs,
    ) -> Union[tuple[chex.Array, chex.Array, dict], tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict]]: # Added Union to type hint
        
        # `latents` from example pairs. Shape: (*B, N_examples, H)
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)

        # This key handling assumes `key` is a single key for the entire generate_output call.
        # If different sub-operations need different keys, split `key` appropriately.
        _key_for_sampling, _key_for_search_or_ga, _key_for_latent_init = None, None, None
        if key is not None:
            key, _key_for_sampling = jax.random.split(key)
            key, _key_for_search_or_ga = jax.random.split(key)
            _, _key_for_latent_init = jax.random.split(key)


        if latents_logvar is not None:
            assert _key_for_sampling is not None, "'key' sub-key required for variational inference in generate_output."
            # `example_latents` are from the provided N_examples pairs
            example_latents, *_ = self._sample_latents(latents_mu, latents_logvar, _key_for_sampling)
        else:
            example_latents = latents_mu # (*B, N_examples, H)

        if mode_kwargs.get("remove_encoder_latents", False):
            assert _key_for_latent_init is not None, "'key' sub-key required for remove_encoder_latents."
            example_latents = jax.random.normal(_key_for_latent_init, example_latents.shape)

        # `context_candidate` will be (*B,H) or (*B,N_examples,H) before search/optimization
        context_candidate_for_search = example_latents

        if use_cross_attention:
            print(f"LPN generate_output: using cross attention: {use_cross_attention}")
            H = example_latents.shape[-1]
            sqrt_dh = jnp.sqrt(float(H))
            query = example_latents.mean(axis=-2, keepdims=True) # (*B, 1, H)
            key_val_attn = example_latents # (*B, N_examples, H)
            attn_scores = jnp.einsum('...qh,...kh->...qk', query, key_val_attn) / sqrt_dh
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            # attended_context: (*B, H) - single context from N_examples
            attended_context = jnp.einsum('...qk,...kh->...qh', attn_weights, key_val_attn).squeeze(axis=-2)
            
            # The search/optimization modes will start from this single attended_context.
            # If they expect N_examples latents, tile it.
            # However, _get_random_search_context `latents` input is `num_latents_to_start_search_from`.
            # Here, we have one primary candidate: attended_context.
            # So, context_candidate_for_search becomes (*B, 1, H) essentially.
            # The `pairs` and `grid_shapes` args to search methods are still the N_examples.
            context_candidate_for_search = attended_context[..., None, :] # Shape (*B, 1, H) to represent one starting point for search
            # Ensure _prepare_latents_before_search can handle this shape if include_mean_latent/all_latents are used.
            # For CA, usually include_mean_latent=True, include_all_latents=False for _prepare_latents.
            # The search will use all N_examples pairs to evaluate this candidate.
        
        # Determine the context(s) to use for actual generation. Should be (*B,H)
        if mode == "mean":
            # If CA, context_candidate_for_search is (*B,1,H) -> mean is (*B,H)
            # If no CA, context_candidate_for_search is (*B,N_examples,H) -> mean is (*B,H)
            final_context = context_candidate_for_search.mean(axis=-2) # Always produces (*B,H)
            first_context, second_context = final_context, final_context
        elif mode == "first":
            # If CA, context_candidate_for_search is (*B,1,H) -> [...,0,:] is (*B,H)
            # If no CA, context_candidate_for_search is (*B,N_examples,H) -> [...,0,:] is (*B,H)
            final_context = context_candidate_for_search[..., 0, :] # Always produces (*B,H)
            first_context, second_context = final_context, final_context
        elif mode == "random_search":
            assert _key_for_search_or_ga is not None, "'key' sub-key for 'random_search' required."
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs, f"'{arg}' argument required."
            # `context_candidate_for_search` is (*B, K, H) where K=1 if CA, K=N_examples if no CA
            # `pairs` & `grid_shapes` are the N_examples ones.
            # Search methods return (*B,H) contexts.
            first_context, second_context = self._get_random_search_context(
                context_candidate_for_search, pairs, grid_shapes, _key_for_search_or_ga, **mode_kwargs
            )
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"'{arg}' argument required."
            # Key for GA random_perturbation is _key_for_search_or_ga
            first_context, second_context = self._get_gradient_ascent_context(
                context_candidate_for_search, pairs, grid_shapes, _key_for_search_or_ga, **mode_kwargs
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        info = {"context": first_context} # first_context is (*B,H)

        # `_generate_output_from_context` expects context of shape (*B,H)
        # and input_grid, input_grid_shape for a single task.
        if return_two_best:
            # Stack first_context and second_context along a new leading dimension for vmap
            contexts_to_generate = jnp.stack([first_context, second_context], axis=0) # (2, *B, H)
            
            # vmap over the first dimension of contexts_to_generate
            output_grids_stacked, output_shapes_stacked = jax.vmap(
                partial(
                    self._generate_output_from_context, # context is first arg of partial
                    input_grid=input_grid,
                    input_grid_shape=input_grid_shape,
                    dropout_eval=dropout_eval,
                )
            )(contexts_to_generate) # Pass (2, *B, H)
            
            first_output_grids, second_output_grids = output_grids_stacked[0], output_grids_stacked[1]
            first_output_shapes, second_output_shapes = output_shapes_stacked[0], output_shapes_stacked[1]
            return first_output_grids, first_output_shapes, second_output_grids, second_output_shapes, info
        else:
            output_grids, output_shapes = self._generate_output_from_context(
                first_context, input_grid, input_grid_shape, dropout_eval
            )
            return output_grids, output_shapes, info


    def _generate_output_from_context(
        self, context: chex.Array, input_grid: chex.Array, input_grid_shape: chex.Array, dropout_eval: bool
    ) -> tuple[chex.Array, chex.Array]:
        # context: (*batch_dims, H)
        # input_grid: (*batch_dims, R, C)
        # input_grid_shape: (*batch_dims, 2)

        batch_dims = input_grid.shape[:-2] 
        H_dim = context.shape[-1]

        flattened_input_grid = jnp.reshape(input_grid, (*batch_dims, -1))
        input_seq_single = jnp.concatenate([input_grid_shape, flattened_input_grid], axis=-1)
        output_seq_single_init = jnp.zeros_like(input_seq_single).at[..., :2].set(1)

        s_len = input_seq_single.shape[-1]
        
        input_seq_N1 = jnp.reshape(input_seq_single, (*batch_dims, 1, s_len))
        context_N1 = jnp.reshape(context, (*batch_dims, 1, H_dim))
        current_output_seq_N1 = jnp.reshape(output_seq_single_init, (*batch_dims, 1, s_len))

        # Inner function for predicting row/col shapes
        def grid_shape_step_fn(output_seq_step_N1: chex.Array, row_flag: bool) -> chex.Array:
            pred_row_logits_N1, pred_col_logits_N1, _ = self.decoder(
                input_seq_N1, output_seq_step_N1, context_N1, dropout_eval
            )
            target_logits_N1 = pred_row_logits_N1 if row_flag else pred_col_logits_N1
            target_logits_single = target_logits_N1.squeeze(axis=-2) 
            
            new_token_val_single = jnp.argmax(target_logits_single, axis=-1).astype(output_seq_single_init.dtype) + 1
            # Expand new_token_val_single to match the slice shape for .set()
            # If new_token_val_single is (*batch_dims,), slice is (*batch_dims,1)
            # new_token_val_expanded shape should be (*batch_dims, 1)
            new_token_val_expanded = new_token_val_single[..., None]

            token_idx = 0 if row_flag else 1
            return output_seq_step_N1.at[..., token_idx].set(new_token_val_expanded) # Use expanded

        current_output_seq_N1 = grid_shape_step_fn(current_output_seq_N1, row_flag=True)
        current_output_seq_N1 = grid_shape_step_fn(current_output_seq_N1, row_flag=False)
        
        output_shapes_predicted = current_output_seq_N1[..., :2].squeeze(axis=-2)
        max_cols_cfg = self.decoder.config.max_cols

        # Inner function for nn.scan to predict grid tokens
        def scan_step_fn(loop_carry_output_seq_N1: chex.Array, grid_token_idx: int):
            *_, pred_grid_lgts_N1 = self.decoder(
                input_seq_N1, loop_carry_output_seq_N1, context_N1, dropout_eval
            )
            num_cols_val = output_shapes_predicted[..., 1].astype(jnp.int32)
            is_start_of_new_row = (grid_token_idx % max_cols_cfg == 0) & (grid_token_idx > 0)
            prev_row_end_log_idx = (grid_token_idx // max_cols_cfg - 1) * max_cols_cfg + num_cols_val
            current_pos_log_idx = jnp.full_like(num_cols_val, grid_token_idx)
            final_sel_idx_single = jnp.where(is_start_of_new_row, prev_row_end_log_idx, current_pos_log_idx)
            idx_for_take_N1 = jnp.reshape(final_sel_idx_single, (*batch_dims, 1, 1, 1))
            sel_logits_N1 = jnp.take_along_axis(pred_grid_lgts_N1, idx_for_take_N1, axis=-2)
            sel_logits_single = sel_logits_N1.squeeze(axis=(-3, -2))
            
            new_grid_token_val_single = jnp.argmax(sel_logits_single, axis=-1).astype(output_seq_single_init.dtype)
            # Expand new_grid_token_val_single for .set()
            # Slice shape is (*batch_dims,1), new_grid_token_val_single is (*batch_dims,)
            new_grid_token_val_expanded = new_grid_token_val_single[..., None]
            
            updated_loop_output_seq_N1 = loop_carry_output_seq_N1.at[..., 2 + grid_token_idx].set(new_grid_token_val_expanded) # Use expanded
            return updated_loop_output_seq_N1, None

        final_gen_output_seq_N1, _ = nn.scan(
            scan_step_fn, variable_broadcast="params", split_rngs={"params": False},
        )(current_output_seq_N1, jnp.arange(self.decoder.config.max_len))

        final_gen_output_seq_single = final_gen_output_seq_N1.squeeze(axis=-2)
        output_grids_final = jnp.reshape(final_gen_output_seq_single[..., 2:], input_grid.shape)

        return output_grids_final, output_shapes_predicted

    # _get_random_search_context and _get_gradient_ascent_context remain largely the same.
    # Their `latents` input shape is (*B, K, H) where K is num starting latents.
    # Their `pairs` shape is (*B, N_pairs, ...)
    # They compute log_probs over N_pairs for each of K latents, returning best K' of shape (*B,H).
    def _get_random_search_context(
        self,
        latents_to_search_from: chex.Array, # (*B, K, H) e.g. K=N_examples or K=1 if CA
        pairs: chex.Array, # (*B, N_pairs_for_eval, R,C,2)
        grid_shapes: chex.Array, # (*B, N_pairs_for_eval, 2,2)
        key: chex.PRNGKey,
        num_samples: int,
        scale: float,
        scan_batch_size: Optional[int] = None,
        include_mean_latent: bool = True, # Applied to latents_to_search_from
        include_all_latents: bool = False, # Applied to latents_to_search_from
        **kwargs,
    ) -> tuple[chex.Array, chex.Array]:
        # latents_prepared: (*B, K_prepared, H)
        latents_prepared = self._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents_to_search_from, key=key # Pass key for potential perturbation in prep
        )

        if num_samples > 0:
            # Sample some random latents around the latents_prepared.
            num_latents_K_prep = latents_prepared.shape[-2]
            # Ensure num_padded_samples is multiple of num_latents_K_prep only if num_latents_K_prep > 0
            if num_latents_K_prep == 0 : # Should not happen if prep_latents is sensible
                 num_padded_samples = num_samples
            else:
                 num_padded_samples = math.ceil(num_samples / num_latents_K_prep) * num_latents_K_prep
            
            # Split key for random vector generation
            key_random_vec, _ = jax.random.split(key) if key is not None else (None, None)
            assert key_random_vec is not None, "Key required for num_samples > 0 in random search"

            random_vectors_shape = (*latents_prepared.shape[:-2], # (*B)
                                    num_latents_K_prep if num_latents_K_prep > 0 else 1, # K_prep or 1 if K_prep=0
                                    num_padded_samples // (num_latents_K_prep if num_latents_K_prep > 0 else 1), # samples per K_prep
                                    latents_prepared.shape[-1]) # H
            
            random_vectors = jax.random.normal(key_random_vec, random_vectors_shape)
            
            # Expand latents_prepared for broadcasting with random_vectors
            # latents_prepared is (*B, K_prep, H) -> (*B, K_prep, 1, H)
            expanded_latents_prepared = latents_prepared[..., None, :] if num_latents_K_prep > 0 else jnp.zeros((*latents_prepared.shape[:-2],0,1,latents_prepared.shape[-1]))


            random_latents = expanded_latents_prepared + scale * random_vectors
            # Reshape to (*B, K_prep * (samples_per_K_prep), H)
            random_latents = random_latents.reshape(*random_latents.shape[:-3], -1, random_latents.shape[-1])
            # Truncate to exact num_samples if num_padded_samples > num_samples
            random_latents = random_latents[..., :num_samples, :] 
            
            # Concatenate. Resulting latents_all_candidates: (*B, K_final, H)
            latents_all_candidates = jnp.concatenate([latents_prepared, random_latents], axis=-2)
        else:
            latents_all_candidates = latents_prepared

        # Flatten input/output for decoding likelihood. These are from `pairs` (*B, N_pairs_for_eval, ...)
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)
        # input_seq/output_seq: (*B, N_pairs_for_eval, SeqLen)

        # We want to evaluate each of K_final candidate latents against all N_pairs_for_eval.
        # latents_all_candidates: (*B, K_final, H)
        # input_seq: (*B, N_pairs_for_eval, SeqLen)
        # Need to make shapes compatible for vmap/scan over decoder.
        # Target for decoder: latents (*B, N_pairs_for_eval, K_final, H) or similar broadcastable form.
        
        # Option 1: vmap over K_final, then sum/mean over N_pairs_for_eval.
        # This is what log_probs_fn and its vmap in _get_gradient_ascent_context does.
        
        # For batched decoding if K_final is large (scan_batch_size logic):
        # This part assumes latents are (*B, N_pairs_for_eval, K_final_batched, H)
        # The original code: `latents = latents[..., None, :, :].repeat(output_seq.shape[-2], axis=-3)`
        # This assumes `latents` input is (*B, K_final, H) and `output_seq` is (*B, N_pairs_for_eval, SeqLen)
        # Resulting `latents_repeated`: (*B, N_pairs_for_eval, K_final, H)
        # This is achieved by:
        # latents_all_candidates is (*B, K_final, H)
        # target_shape for repeat: (*B, K_final, N_pairs_for_eval, H) then transpose
        num_k_final = latents_all_candidates.shape[-2]
        num_n_pairs = output_seq.shape[-2]

        # Expand K_final latents to be used for each of N_pairs_for_eval
        # (*B, K_final, H) -> (*B, K_final, 1, H)
        latents_expanded_k = latents_all_candidates[..., None, :] 
        # Tile along new dim: (*B, K_final, N_pairs_for_eval, H)
        latents_tiled_for_n = jnp.tile(latents_expanded_k, 
                                    (*([1]*latents_expanded_k.ndim[:-2]), num_n_pairs, 1) )
        # Transpose to (*B, N_pairs_for_eval, K_final, H) for easier batching over K_final with vmapped decoder.
        # batch_dims are all leading dimensions up to N_pairs_for_eval.
        # Example: if latents_tiled_for_n is (B, K, N, H), transpose to (B, N, K, H)
        # Current shape is (*batch_dims_of_B, K_final, N_pairs_for_eval, H)
        # Need to find axes for K_final and N_pairs_for_eval
        axes = list(range(latents_tiled_for_n.ndim))
        k_axis_idx = latents_tiled_for_n.ndim - 3
        n_axis_idx = latents_tiled_for_n.ndim - 2
        axes[k_axis_idx], axes[n_axis_idx] = axes[n_axis_idx], axes[k_axis_idx] # Swap K and N axes
        latents_for_decoder_calls = jnp.transpose(latents_tiled_for_n, axes=axes)
        # Now latents_for_decoder_calls is (*batch_dims_of_B, N_pairs_for_eval, K_final, H)

        # `input_seq` is (*B, N_pairs_for_eval, SeqLen_in)
        # `output_seq` is (*B, N_pairs_for_eval, SeqLen_out)
        # `latents_for_decoder_calls` is (*B, N_pairs_for_eval, K_final, H)
        # Decoder call needs to be vmapped over K_final dim.
        # So, `decoder(input_seq_n, output_seq_n, latents_k_for_n, ...)`
        # where _n means indexed by N_pairs_for_eval, _k by K_final.
        # The jax.vmap(decoder, in_axes=(None, None, -2, None), out_axes=-2) assumes
        # latents are (*B, N_pairs_for_eval, K_final_batch, H), and input/output are broadcast.
        # This matches `latents_for_decoder_calls` if K_final is K_final_batch.

        # Batching logic for K_final dimension (axis=-2 of latents_for_decoder_calls after N_pairs_for_eval dim)
        # This is complex; assuming the original batching logic for K (scan_batch_size) is sound.
        # The key change is that `latents` now refers to `latents_all_candidates`
        # And the repeat/transpose logic correctly prepares `latents_for_decoder_calls`.

        # Simplified: use the value_and_grad_log_probs_fn structure from _get_gradient_ascent_context
        # but only need values.
        # Define log_probs_fn_for_search (similar to one in _get_gradient_ascent_context)
        def log_probs_fn_search(candidate_latent_k, inp_seq_all_n, out_seq_all_n, decoder_inst):
            # candidate_latent_k: (*B, H) - one of K_final latents
            # inp_seq_all_n: (*B, N_pairs_for_eval, SeqLen)
            # out_seq_all_n: (*B, N_pairs_for_eval, SeqLen)
            # Tile candidate_latent_k for N_pairs_for_eval: (*B, H) -> (*B, N_pairs_for_eval, H)
            num_n_p = inp_seq_all_n.shape[-2]
            leading_dims_cand = candidate_latent_k.shape[:-1]
            cand_expanded = jnp.expand_dims(candidate_latent_k, axis=len(leading_dims_cand))
            tile_reps = [1]*cand_expanded.ndim
            tile_reps[len(leading_dims_cand)] = num_n_p
            latents_k_for_all_n = jnp.tile(cand_expanded, tile_reps)

            row_logits, col_logits, grid_logits = decoder_inst(inp_seq_all_n, out_seq_all_n, latents_k_for_all_n, dropout_eval=True)
            # _compute_log_probs sums over N_pairs_for_eval dim, returns (*B,) for this k-th latent
            return self._compute_log_probs(row_logits, col_logits, grid_logits, out_seq_all_n)

        # Vmap over K_final dimension of latents_all_candidates
        # latents_all_candidates: (*B, K_final, H)
        # input_seq/output_seq are broadcasted (or handled by None in_axes)
        # log_probs will be (*B, K_final)
        batch_dims_count = len(pairs.shape[:-4]) # Number of leading batch dimensions B*
        vmap_axes_latent = batch_dims_count # This is the K_final dimension
        log_probs = jax.vmap(log_probs_fn_search, 
                             in_axes=(vmap_axes_latent, # Corresponds to K_final dim of latents_all_candidates
                                      None, None, None), # Broadcast input_seq, output_seq, decoder
                             out_axes=vmap_axes_latent   # Output log_probs also has K_final at this axis
                             )(latents_all_candidates, input_seq, output_seq, self.decoder)


        # latents_all_candidates is (*B, K_final, H)
        # log_probs is (*B, K_final)
        best_context, second_best_context = self._select_best_and_second_best_latents(log_probs, latents_all_candidates)
        # Returns (*B,H)

        return best_context, second_best_context

    def _get_gradient_ascent_context(
        self,
        latents_to_optimize_from: chex.Array, # (*B, K, H) e.g. K=N_examples or K=1 if CA
        pairs: chex.Array, # (*B, N_pairs_for_eval, R,C,2)
        grid_shapes: chex.Array, # (*B, N_pairs_for_eval, 2,2)
        key: Optional[chex.PRNGKey], # For random_perturbation in _prepare_latents_before_search
        num_steps: int,
        lr: float,
        # ... other args ...
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        random_perturbation: Optional[dict] = None,
        stop_gradient_latent_move: bool = True,
        **kwargs,
    ) -> tuple[chex.Array, chex.Array]:
        
        # latents_prepared: (*B, K_prepared, H)
        latents_prepared = self._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents_to_optimize_from, random_perturbation, key
        )

        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)
        # input_seq/output_seq: (*B, N_pairs_for_eval, SeqLen)

        # log_probs_fn is defined to take one latent (*B,H) and evaluate against all N_pairs_for_eval
        def log_probs_fn_ga(latent_k_opt, inp_seq_all_n, out_seq_all_n, decoder_inst):
            num_n_p = inp_seq_all_n.shape[-2]
            leading_dims_k_opt = latent_k_opt.shape[:-1]
            k_opt_expanded = jnp.expand_dims(latent_k_opt, axis=len(leading_dims_k_opt))
            tile_reps = [1]*k_opt_expanded.ndim
            tile_reps[len(leading_dims_k_opt)] = num_n_p
            latents_k_opt_for_all_n = jnp.tile(k_opt_expanded, tile_reps)
            
            row_logits, col_logits, grid_logits = decoder_inst(inp_seq_all_n, out_seq_all_n, latents_k_opt_for_all_n, dropout_eval=True)
            return self._compute_log_probs(row_logits, col_logits, grid_logits, out_seq_all_n)

        # Vmap value_and_grad over the K_prepared dimension of latents_prepared
        batch_dims_count = len(pairs.shape[:-4])
        k_prep_axis = batch_dims_count 

        # value_and_grad_log_probs_fn_vmapped: processes all K_prepared latents
        # Input latents will be K_prepared dim of latents_prepared.
        # Output log_probs: (*B, K_prepared), grads: (*B, K_prepared, H)
        value_and_grad_log_probs_fn_vmapped = jax.vmap(
            jax.value_and_grad(log_probs_fn_ga), 
            in_axes=(k_prep_axis, None, None, None), # Vmap over latent_k_opt (K_prepared dim)
            out_axes=(k_prep_axis, k_prep_axis)      # log_probs and grads also have K_prepared dim
        )
        # The original code had more vmaps for outer batch dimensions. This should be handled by JAX's vmap autobatching.
        # If not, the `for batch_dim in range(...)` loop might be needed if `pairs` has >1 leading batch dim.
        # For now, assume `batch_dims_count` covers all of B*.

        # Optimizer setup (as in original)
        lr_schedule_val = kwargs.get("lr_schedule", False)
        lr_schedule_exponent_val = kwargs.get("lr_schedule_exponent", 0.5)
        optimizer_name = kwargs.get("optimizer", "sgd") # optimizer arg was shadowed
        optimizer_kwargs_val = kwargs.get("optimizer_kwargs", None)

        current_lr = lr
        if lr_schedule_val:
            current_lr = optax.cosine_decay_schedule(lr, num_steps, exponent=lr_schedule_exponent_val)
        
        optax_optimizer_chain = [optax.clip_by_global_norm(1.0)]
        if optimizer_name == "sgd":
            optax_optimizer_chain.append(optax.sgd(learning_rate=current_lr, **(optimizer_kwargs_val or {})))
        elif optimizer_name == "adam":
            optax_optimizer_chain.append(optax.adam(learning_rate=current_lr, eps_root=1e-8, **(optimizer_kwargs_val or {})))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        optimizer_combined = optax.chain(*optax_optimizer_chain)
        
        # opt_state is for latents_prepared of shape (*B, K_prepared, H)
        opt_state = optimizer_combined.init(latents_prepared) 

        # Scan loop for gradient ascent steps
        optimized_latents_over_steps = []
        log_probs_over_steps = []
        
        current_optimized_latents = latents_prepared

        for _ in range(num_steps):
            log_probs_k, grads_k = value_and_grad_log_probs_fn_vmapped(
                current_optimized_latents, input_seq, output_seq, self.decoder
            )
            # grads_k shape: (*B, K_prepared, H)
            if stop_gradient_latent_move:
                grads_k = jax.lax.stop_gradient(grads_k)
            
            updates_k, opt_state = optimizer_combined.update(-grads_k, opt_state, current_optimized_latents)
            current_optimized_latents += updates_k
            
            optimized_latents_over_steps.append(current_optimized_latents)
            log_probs_over_steps.append(log_probs_k)

        # Collate results from all steps
        # all_latents_collated: (*B, K_prepared, num_steps, H) -> then reshape
        all_latents_collated = jnp.stack(optimized_latents_over_steps, axis=batch_dims_count + 1) # Stack along step dim
        # all_log_probs_collated: (*B, K_prepared, num_steps) -> then reshape
        all_log_probs_collated = jnp.stack(log_probs_over_steps, axis=batch_dims_count + 1)

        # Reshape to flatten K_prepared and num_steps:
        # final_candidate_latents: (*B, K_prepared * num_steps, H)
        final_candidate_latents_shape = (*latents_prepared.shape[:batch_dims_count], -1, latents_prepared.shape[-1])
        final_candidate_latents = jnp.reshape(all_latents_collated, final_candidate_latents_shape)
        
        # final_log_probs: (*B, K_prepared * num_steps)
        final_log_probs_shape = (*log_probs_k.shape[:batch_dims_count], -1)
        final_log_probs = jnp.reshape(all_log_probs_collated, final_log_probs_shape)

        # Also include the initial latents_prepared in the selection pool
        # Need their log_probs too
        initial_log_probs_k, _ = value_and_grad_log_probs_fn_vmapped(
            latents_prepared, input_seq, output_seq, self.decoder
        ) # (*B, K_prepared)

        # Combine initial with optimized
        combined_latents_for_selection = jnp.concatenate([latents_prepared, final_candidate_latents], axis=batch_dims_count)
        combined_log_probs_for_selection = jnp.concatenate([initial_log_probs_k, final_log_probs], axis=batch_dims_count)
        
        best_context, second_best_context = self._select_best_and_second_best_latents(
            combined_log_probs_for_selection, combined_latents_for_selection
        )
        return best_context, second_best_context


    @classmethod
    def _prepare_latents_before_search(
        cls,
        include_mean_latent: bool,
        include_all_latents: bool,
        latents: chex.Array, # (*B, K, H) - K initial candidates
        random_perturbation: Optional[dict] = None,
        key: Optional[chex.PRNGKey] = None, # For random_perturbation
    ) -> chex.Array:
        prep_latents_list = []
        if include_all_latents: # Use all K initial candidates
            prep_latents_list.append(latents)
        
        if include_mean_latent: # Add mean of K initial candidates
            if latents.shape[-2] > 0 : # Check if K > 0
                mean_latent = latents.mean(axis=-2, keepdims=True) # (*B, 1, H)
                prep_latents_list.append(mean_latent)
            # If K=0 and only mean is requested, this might be an issue. Assume K>=1 if mean needed.

        if not prep_latents_list: # If neither all nor mean, and K>0, this is like include_all_latents=True
            if latents.shape[-2] > 0: # K > 0
                 prep_latents_list.append(latents)
            else: # K=0, no latents to start from (problematic)
                 # Return empty or a zero vector of expected H dim if possible
                 # This case should ideally be guarded by upstream logic.
                 # For now, assume H can be inferred or is fixed.
                 # This path signifies an issue if reached with latents.shape[-2]==0.
                 # Create a dummy zero latent if nothing else, to avoid concat error with random_perturbation
                 # Requires knowing H, which we can get from latents.shape[-1] IF K>0.
                 # If K=0, latents could be (*B,0,H).
                 if latents.ndim > 1 and latents.shape[-1] > 0: # H is known
                    dummy_H = latents.shape[-1]
                    dummy_shape = (*latents.shape[:-2], 0, dummy_H) # (*B,0,H)
                    prep_latents = jnp.zeros(dummy_shape, dtype=latents.dtype) # Still has K=0
                 else: # Cannot infer H, this is a problem.
                    raise ValueError("Cannot prepare latents: input `latents` has K=0 and H cannot be inferred.")


        if prep_latents_list:
            prep_latents = jnp.concatenate(prep_latents_list, axis=-2) # Concatenate along K dim
        elif not random_perturbation: # No initial latents and no random perturbation -> error or empty
            # This should not happen if logic is correct. If latents has K=0, prep_latents is K=0.
            # If random_perturbation is also None, then we have no candidates.
            # Fallback to the dummy shape if created.
            pass # prep_latents might be the K=0 dummy here

        if random_perturbation is not None:
            assert key is not None, "'key' argument required for random perturbation."
            for arg in ["num_samples", "scale"]:
                assert arg in random_perturbation, f"'{arg}' argument required."
            
            num_rand_samples = random_perturbation["num_samples"]
            scale_rand = random_perturbation["scale"]

            # Base for perturbation: mean of original K latents, or zero if K=0
            if latents.shape[-2] > 0:
                perturb_base = latents.mean(axis=-2, keepdims=True) # (*B, 1, H)
            else: # K=0, perturb around zero vector
                dummy_H_pert = latents.shape[-1] if latents.ndim >1 and latents.shape[-1] > 0 else cls.decoder.config.hidden_size # Fallback H
                perturb_base_shape = (*latents.shape[:-2], 1, dummy_H_pert)
                perturb_base = jnp.zeros(perturb_base_shape, dtype=latents.dtype)


            random_vectors = jax.random.normal(key, (*perturb_base.shape[:-2], num_rand_samples, perturb_base.shape[-1]))
            perturbed_random_latents = perturb_base + scale_rand * random_vectors # (*B, num_rand_samples, H)
            
            if prep_latents.shape[-2] > 0 : # If prep_latents had some candidates
                prep_latents = jnp.concatenate([prep_latents, perturbed_random_latents], axis=-2)
            else: # prep_latents was empty (K=0), so it just becomes the random ones
                prep_latents = perturbed_random_latents
        
        if prep_latents.shape[-2] == 0:
            raise ValueError("No latents prepared for search/optimization. Check include_mean/all_latents and random_perturbation settings.")
            
        return prep_latents


    @classmethod
    def _flatten_input_output_for_decoding(
        cls, pairs: chex.Array, grid_shapes: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        # pairs: (*B,N,R,C,2), grid_shapes: (*B,N,2,2)
        # Output: input_seq (*B,N,SeqLen), output_seq (*B,N,SeqLen)
        # where SeqLen = 2 (for shape) + R*C (for grid)
        
        # Make sure leading batch dims are preserved correctly.
        # pairs.shape[:-3] gives (*B,N)
        # pairs.shape[-3] is R, pairs.shape[-2] is C.
        # So R*C is pairs.shape[-3]*pairs.shape[-2]
        # However, R and C can vary per example if not padded to max_R, max_C.
        # The Reshape `(*pairs.shape[:-3], -1, 2)` assumes R,C are fixed for all in batch.
        # This is typical. SeqLen becomes max_R * max_C.
        
        flattened_pairs = jnp.reshape(pairs, (*pairs.shape[:-3], -1, 2))
        # grid_shapes[..., 0] is input shapes part (*B,N,2)
        # flattened_pairs[..., 0] is input grid part (*B,N,max_R*max_C)
        input_seq = jnp.concatenate([grid_shapes[..., 0, :], flattened_pairs[..., 0]], axis=-1)
        output_seq = jnp.concatenate([grid_shapes[..., 1, :], flattened_pairs[..., 1]], axis=-1)
        return input_seq, output_seq

    @classmethod
    def _select_best_and_second_best_latents(
        cls, log_probs: chex.Array, latents: chex.Array # log_probs (*B,K), latents (*B,K,H)
    ) -> tuple[chex.Array, chex.Array]: # Returns (*B,H), (*B,H)
        # Argsort along K dimension (axis=-1 for log_probs, axis=-2 for latents if K is -2)
        # Assuming K is the last dim of log_probs, and second to last of latents.
        k_dim_log_probs = -1
        k_dim_latents = -2 # K is at latents.shape[-2]

        sorted_log_probs_indices = jnp.argsort(log_probs, axis=k_dim_log_probs, descending=True) # (*B,K) indices
        
        # Need to expand sorted_log_probs_indices for take_along_axis on latents
        # Target: index for K dim of latents. Shape: (*B, 1, 1) to get (*B,1,H)
        # sorted_log_probs_indices[..., 0:1] gives (*B,1) - indices of best K
        # Add extra dim for H: sorted_log_probs_indices[..., 0:1, None] -> (*B,1,1)
        
        best_indices_expanded = sorted_log_probs_indices[..., 0:1, None]
        best_context = jnp.take_along_axis(
            latents, best_indices_expanded, axis=k_dim_latents # gather along K dim
        ).squeeze(axis=k_dim_latents) # Squeeze the K dim (which is now size 1)
        
        if sorted_log_probs_indices.shape[k_dim_log_probs] > 1: # If more than one candidate
            second_best_indices_expanded = sorted_log_probs_indices[..., 1:2, None]
            second_best_context = jnp.take_along_axis(
                latents, second_best_indices_expanded, axis=k_dim_latents
            ).squeeze(axis=k_dim_latents)
        else:
            second_best_context = best_context # Fallback if only one latent
        return best_context, second_best_context

    def _compute_log_probs(
        self,
        row_logits: chex.Array, # (*B, N_p, MaxR_vocab)
        col_logits: chex.Array, # (*B, N_p, MaxC_vocab)
        grid_logits: chex.Array, # (*B, N_p, SeqLen, Vocab)
        output_seq: chex.Array, # (*B, N_p, SeqLen_total)
        grid_log_prob_weight: float = 1.0,
    ) -> chex.Array: # Returns (*B,) by summing over N_p
        
        max_cols = self.decoder.config.max_cols
        # output_seq[...,0] is num_rows, output_seq[...,1] is num_cols. Shapes (*B, N_p)
        num_rows, num_cols = output_seq[..., 0].astype(jnp.int32), output_seq[..., 1].astype(jnp.int32)
        
        row_all_log_probs = jax.nn.log_softmax(row_logits, axis=-1)
        # Index is num_rows-1. Shape (*B,N_p,1) for index to get (*B,N_p) log_probs
        row_log_probs = jnp.take_along_axis(row_all_log_probs, num_rows[..., None] - 1, axis=-1).squeeze(axis=-1)
        
        col_all_log_probs = jax.nn.log_softmax(col_logits, axis=-1)
        col_log_probs = jnp.take_along_axis(col_all_log_probs, num_cols[..., None] - 1, axis=-1).squeeze(axis=-1)
        
        last_non_padded_logits = self._get_last_non_padded_logits(grid_logits, num_cols[..., None, None])
        # Ensure slicing on grid_logits is safe
        if max_cols > 0 and grid_logits.shape[-2] > max_cols: # SeqLen > max_cols
            grid_logits = grid_logits.at[..., max_cols::max_cols, :].set(last_non_padded_logits)

        grid_all_log_probs = jax.nn.log_softmax(grid_logits, axis=-1)
        # output_seq[..., 2:] are grid tokens. Shape (*B, N_p, SeqLen_grid)
        # Need [..., None] for take_along_axis. Result (*B, N_p, SeqLen_grid)
        grid_token_log_probs = jnp.take_along_axis(grid_all_log_probs, output_seq[..., 2:, None].astype(jnp.int32), axis=-1).squeeze(axis=-1)
        
        # Normalized mean over sequence for grid_token_log_probs
        # num_rows/cols are (*B,N_p). grid_token_log_probs is (*B,N_p,SeqLen_grid)
        avg_grid_log_probs = self._normalized_mean_over_sequence(grid_token_log_probs, num_rows, num_cols)

        # log_probs_per_pair: (*B, N_p)
        log_probs_per_pair = row_log_probs + col_log_probs + grid_log_prob_weight * avg_grid_log_probs
        
        # Sum over N_p pairs. Result (*B,). Original code had this.
        total_log_probs = jnp.sum(log_probs_per_pair, axis=-1) 
        return total_log_probs

    def _get_last_non_padded_logits(self, grid_logits: chex.Array, num_cols: chex.Array) -> chex.Array:
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        if max_rows <= 1: # No rows to copy from if only one row or fewer
            # Return empty array of compatible shape
            # grid_logits: (*dims, SeqLen, Vocab). SeqLen for last_non_padded should be 0.
            empty_shape = (*grid_logits.shape[:-2], 0, grid_logits.shape[-1])
            return jnp.zeros(empty_shape, dtype=grid_logits.dtype)

        last_non_padded_logits_list = []
        num_cols_int = num_cols.astype(jnp.int32)

        for i in range(1, max_rows): # For target row i (1-idxed), copy from prev row (i-1) (0-idxed)
            # Index for the end of the (i-1)-th row in the flattened sequence part of grid_logits
            # This index must be relative to grid_logits's sequence dim.
            # Example: (*B, N, SeqLen, Vocab). Index is for SeqLen dim.
            # num_cols_int could be (*B,N,1,1). Need to ensure it broadcasts for index calculation.
            # indices_for_prev_row_end: (*B,N,1,1) if num_cols_int has these dims.
            indices_for_prev_row_end = (i - 1) * max_cols + (num_cols_int - 1)
            
            # Safety clip for indices
            seq_len_of_grid_logits = grid_logits.shape[-2]
            safe_indices = jnp.clip(indices_for_prev_row_end, 0, seq_len_of_grid_logits - 1)

            end_of_row_logits = jnp.take_along_axis(
                grid_logits, safe_indices, axis=-2 # Select along sequence dimension
            )
            last_non_padded_logits_list.append(end_of_row_logits)
            
        return jnp.concatenate(last_non_padded_logits_list, axis=-2) # Concatenate along seq_dim


# Main block from original file (for testing, if needed)
if __name__ == "__main__":
    # ... (Keep your __main__ block as is for local testing) ...
    # Example:
    from src.models.utils import TransformerLayerConfig
