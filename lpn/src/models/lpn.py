from typing import Literal, Optional, Union 
import math
from functools import partial

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.tree_util import tree_map
import optax
# from jax.debug import print as jax_print # Uncomment for debugging

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
        use_cross_attention: bool = False, 
        **mode_kwargs,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        
        assert pairs.shape[-4] > 1, f"Number of pairs should be greater than 1, got {pairs.shape[-4]}."
        num_program_pairs = pairs.shape[-4] 

        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)

        if latents_logvar is not None:
            key_sample_latents = self.make_rng("latents")
            base_latents, prior_kl_loss, kl_metrics = self._sample_latents(latents_mu, latents_logvar, key_sample_latents)
            pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu, latents_logvar).mean()
            kl_metrics["pairwise_kl"] = pairwise_kl_loss
        else:
            base_latents, prior_kl_loss, pairwise_kl_loss, kl_metrics = latents_mu, None, None, {}

        if mode_kwargs.get("remove_encoder_latents", False):
            key_init_latents = self.make_rng("latents_init")
            base_latents = jax.random.normal(key_init_latents, base_latents.shape)

        current_context_source_latents = base_latents
        
        if use_cross_attention:
            H = base_latents.shape[-1]
            sqrt_dh = jnp.sqrt(float(H))
            query = base_latents.mean(axis=-2, keepdims=True) 
            key_val_attn = base_latents 
            attn_scores = jnp.einsum('...qh,...kh->...qk', query, key_val_attn) / sqrt_dh
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            attended_context = jnp.einsum('...qk,...kh->...qh', attn_weights, key_val_attn).squeeze(axis=-2)

            leading_dims_shape = attended_context.shape[:-1] 
            tile_repeats = [1] * len(leading_dims_shape) + [num_program_pairs, 1] 
            current_context_source_latents = jnp.tile(attended_context[..., None, :], tile_repeats)

        leave_one_out_source_latents = make_leave_one_out(current_context_source_latents, axis=-2)

        if mode == "mean":
            context_for_loss = leave_one_out_source_latents.mean(axis=-2)
            loss, metrics = self._loss_from_pair_and_context(context_for_loss, pairs, grid_shapes, dropout_eval)
        elif mode == "all":
            loss, metrics = jax.vmap(
                self._loss_from_pair_and_context, in_axes=(-2, None, None, None), out_axes=-1
            )(leave_one_out_source_latents, pairs, grid_shapes, dropout_eval)
            context_for_logging = current_context_source_latents
        elif mode == "random_search" or mode == "gradient_ascent":
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
            
            if mode == "random_search":
                for arg in ["num_samples", "scale"]: assert arg in mode_kwargs
                key_rs = self.make_rng("random_search")
                context_for_loss, _ = self._get_random_search_context(
                    leave_one_out_source_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_rs, **mode_kwargs
                )
            else: 
                for arg in ["num_steps", "lr"]: assert arg in mode_kwargs
                key_ga_name = "gradient_ascent_random_perturbation" 
                key_ga = self.make_rng(key_ga_name) if mode_kwargs.get("random_perturbation") else None
                context_for_loss, _ = self._get_gradient_ascent_context(
                    leave_one_out_source_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key_ga, **mode_kwargs
                )
            loss, metrics = self._loss_from_pair_and_context(context_for_loss, pairs, grid_shapes, dropout_eval)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        final_context_for_metrics = context_for_logging if mode == "all" else context_for_loss
        loo_base_latents = make_leave_one_out(base_latents, axis=-2)
        leave_one_out_final_contexts = make_leave_one_out(final_context_for_metrics, axis=-2)
        
        cosine_between_contexts = jnp.einsum("...h,...nh->...n", final_context_for_metrics, leave_one_out_final_contexts) / (
            norm(final_context_for_metrics, axis=-1)[..., None] * norm(leave_one_out_final_contexts, axis=-1) + 1e-5
        )
        cosine_between_base_latents = jnp.einsum("...h,...nh->...n", base_latents, loo_base_latents) / (
            norm(base_latents, axis=-1)[..., None] * norm(loo_base_latents, axis=-1) + 1e-5
        )
        
        dist_ctx_lat = norm(final_context_for_metrics - base_latents, axis=-1)

        metrics.update(
            latents_norm=norm(base_latents, axis=-1),
            context_norm=norm(final_context_for_metrics, axis=-1),
            distance_context_latents=dist_ctx_lat,
            distance_between_contexts=norm(final_context_for_metrics[..., None, :] - leave_one_out_final_contexts, axis=-1),
            cosine_between_contexts=cosine_between_contexts,
            distance_between_latents=norm(base_latents[..., None, :] - loo_base_latents, axis=-1),
            cosine_between_latents=cosine_between_base_latents,
        )
        
        loss, metrics = tree_map(jnp.mean, (loss, metrics))
        metrics.update(kl_metrics)
        if prior_kl_loss is not None:
            if prior_kl_coeff is None: raise ValueError("Prior KL coeff required for VI.")
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
        if num_pairs > 1:
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
        self, context: chex.Array, pairs: chex.Array, grid_shapes: chex.Array, dropout_eval: bool,
    ) -> tuple[chex.Array, dict]:
        config = self.decoder.config
        input_seq, output_seq = LPN._flatten_input_output_for_decoding(pairs, grid_shapes)

        context_for_decoder = context
        if context.ndim == input_seq.ndim - 1 and \
           context.shape[:-1] == input_seq.shape[:-2] and \
           hasattr(self.decoder.config, 'hidden_size') and \
           context.shape[-1] == self.decoder.config.hidden_size:
            
            num_program_pairs_in_seq = input_seq.shape[-2]
            axis_for_new_N_dim = context.ndim -1 
            context_expanded = jnp.expand_dims(context, axis=axis_for_new_N_dim) 
            
            tile_repeats_list = [1] * context_expanded.ndim
            tile_repeats_list[axis_for_new_N_dim] = num_program_pairs_in_seq
            context_for_decoder = jnp.tile(context_expanded, tile_repeats_list)
        
        row_logits, col_logits, grid_logits = self.decoder(input_seq, output_seq, context_for_decoder, dropout_eval)
        
        grid_shapes_row, grid_shapes_col = grid_shapes[..., 0, 1], grid_shapes[..., 1, 1]
        one_hot_grid_shapes_row_labels = jax.nn.one_hot(grid_shapes_row - 1, config.max_rows)
        row_loss = -jnp.sum(jax.nn.log_softmax(row_logits) * one_hot_grid_shapes_row_labels, axis=-1)
        one_hot_grid_shapes_col_labels = jax.nn.one_hot(grid_shapes_col - 1, config.max_cols)
        col_loss = -jnp.sum(jax.nn.log_softmax(col_logits) * one_hot_grid_shapes_col_labels, axis=-1)
        
        last_non_padded_logits = self._get_last_non_padded_logits(
            grid_logits, grid_shapes_col[..., None, None] 
        )
        if config.max_cols > 0 and grid_logits.shape[-2] >= config.max_cols :
            if hasattr(last_non_padded_logits, 'shape') and last_non_padded_logits.shape[-2] > 0 : 
                grid_logits = grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)

        one_hot_grid_labels = jax.nn.one_hot(pairs[..., 1].reshape(*pairs.shape[:-3], -1), config.vocab_size)
        grid_losses = -jnp.sum(jax.nn.log_softmax(grid_logits) * one_hot_grid_labels, axis=-1)
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row, grid_shapes_col)
        loss = row_loss + col_loss + grid_loss 
        metrics = {
            "shape_row_loss": row_loss, "shape_col_loss": col_loss,
            "grid_loss": grid_loss, "total_loss": loss,
        }
        return loss, metrics

    def _normalized_mean_over_sequence(
        self, grid_seq: chex.Array, num_rows: chex.Array, num_cols: chex.Array
    ) -> chex.Array:
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        num_rows_int = num_rows.astype(jnp.int32)
        num_cols_int = num_cols.astype(jnp.int32)
        row_arange_b = jnp.arange(max_rows).reshape( *( (1,)*num_rows_int.ndim + (max_rows,)) )
        col_arange_b = jnp.arange(max_cols).reshape( *( (1,)*num_cols_int.ndim + (max_cols,)) )
        grid_row_mask = row_arange_b < num_rows_int[..., None]
        grid_col_mask = col_arange_b < num_cols_int[..., None]
        grid_pad_mask_2d = grid_row_mask[..., :, None] & grid_col_mask[..., None, :]
        grid_pad_mask = grid_pad_mask_2d.reshape(*grid_pad_mask_2d.shape[:-2], -1)
        grid_seq_masked = jnp.where(grid_pad_mask, grid_seq, 0)
        mean_seq = jnp.sum(grid_seq_masked, axis=-1) / (jnp.sum(grid_pad_mask, axis=-1) + 1e-7)
        return mean_seq
    
    def generate_output(
        self,
        pairs: chex.Array,        
        grid_shapes: chex.Array,  
        input: chex.Array,        
        input_grid_shape: chex.Array, 
        key: Optional[chex.PRNGKey],
        dropout_eval: bool,
        mode: Literal["mean", "first", "random_search", "gradient_ascent"],
        return_two_best: bool = False,
        use_cross_attention: bool = False, 
        **mode_kwargs,
    ) -> Union[tuple[chex.Array, chex.Array, dict], tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
        input_grid = input 

        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)

        _key_for_sampling, _key_for_search_or_ga, _key_for_latent_init = None, None, None
        if key is not None: 
            key_parts = jax.random.split(key, 3)
            _key_for_sampling, _key_for_search_or_ga, _key_for_latent_init = key_parts[0], key_parts[1], key_parts[2]

        if latents_logvar is not None:
            assert _key_for_sampling is not None, "Key required for VI sampling in generate_output."
            example_latents, *_ = self._sample_latents(latents_mu, latents_logvar, _key_for_sampling)
        else:
            example_latents = latents_mu

        if mode_kwargs.get("remove_encoder_latents", False):
            assert _key_for_latent_init is not None, "Key required for remove_encoder_latents."
            example_latents = jax.random.normal(_key_for_latent_init, example_latents.shape)

        source_latents_for_gen_modes = example_latents

        print(f"1st generate output: Latents shape: {example_latents.shape}")

        if use_cross_attention:
            H = example_latents.shape[-1]
            sqrt_dh = jnp.sqrt(float(H))
            query = example_latents.mean(axis=-2, keepdims=True)
            key_val_attn = example_latents
            attn_scores = jnp.einsum('...qh,...kh->...qk', query, key_val_attn) / sqrt_dh
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            attended_context_gen = jnp.einsum('...qk,...kh->...qh', attn_weights, key_val_attn).squeeze(axis=-2)
            source_latents_for_gen_modes = attended_context_gen[..., None, :] 
        
        print(f"2nd generate output, after crossatt: Latents shape: {source_latents_for_gen_modes.shape}")

        if mode == "mean":
            final_gen_context = source_latents_for_gen_modes.mean(axis=-2) 
            first_context, second_context = final_gen_context, final_gen_context

            print(f"3rd generate output, after mean: Latents shape: {first_context.shape}")

        elif mode == "first":
            final_gen_context = source_latents_for_gen_modes[..., 0, :] 
            first_context, second_context = final_gen_context, final_gen_context
        elif mode == "random_search":
            assert _key_for_search_or_ga is not None, "Key required for random_search in generate_output."
            for arg in ["num_samples", "scale"]: assert arg in mode_kwargs
            first_context, second_context = self._get_random_search_context(
                source_latents_for_gen_modes, pairs, grid_shapes, _key_for_search_or_ga, **mode_kwargs
            )
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]: assert arg in mode_kwargs
            first_context, second_context = self._get_gradient_ascent_context(
                source_latents_for_gen_modes, pairs, grid_shapes, _key_for_search_or_ga, **mode_kwargs
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        info = {"context": first_context} 

        if return_two_best:
            contexts_to_generate = jnp.stack([first_context, second_context], axis=0)
            output_grids_stacked, output_shapes_stacked = jax.vmap(
                partial(self._generate_output_from_context, 
                        input_grid=input_grid, 
                        input_grid_shape=input_grid_shape, 
                        dropout_eval=dropout_eval)
            )(contexts_to_generate)
            
            first_output_grids, second_output_grids = output_grids_stacked[0], output_grids_stacked[1]
            first_output_shapes, second_output_shapes = output_shapes_stacked[0], output_shapes_stacked[1]
            return first_output_grids, first_output_shapes, second_output_grids, second_output_shapes, info
        else:
            output_grids, output_shapes = self._generate_output_from_context(
                first_context, input_grid, input_grid_shape, dropout_eval
            )
            return output_grids, output_shapes, info

    def _generate_output_from_context(
        self, context: chex.Array, input: chex.Array, input_grid_shape: chex.Array, dropout_eval: bool
    ) -> tuple[chex.Array, chex.Array]:
        flattened_input = jnp.reshape(input, (*input.shape[:-2], -1))
        input_seq = jnp.concatenate([input_grid_shape, flattened_input], axis=-1)
        output_seq = jnp.zeros_like(input_seq).at[..., :2].set(1)  # Initialize the grid shape tokens to 1.

        def grid_shape_step(output_seq: chex.Array, row: bool) -> chex.Array:
            row_logits, col_logits, _ = self.decoder(input_seq, output_seq, context, dropout_eval)
            if row:
                logits = row_logits
            else:
                logits = col_logits
            # +1 to shift the tokens to [1, max_rows] or [1, max_cols]
            new_token = jnp.argmax(logits, axis=-1).astype(output_seq.dtype) + 1
            output_seq = output_seq.at[..., int(not row)].set(new_token)
            return output_seq

        # First predict the number of rows and then the number of columns.
        output_seq = grid_shape_step(output_seq, row=True)
        output_seq = grid_shape_step(output_seq, row=False)
        output_shapes = output_seq[..., :2]
        max_cols = self.decoder.config.max_cols

        def one_step(decoder: DecoderTransformer, output_seq: chex.Array, i: int) -> tuple[chex.Array, None]:
            *_, grid_logits = decoder(input_seq, output_seq, context, dropout_eval)
            # If we are at the beginning of a new row, the index of the logits to predict the next token is
            # the index of the last non-padded token of the previous row.
            logits_index = jnp.where(
                (i % max_cols == 0) & (i > 0),
                (i // max_cols - 1) * max_cols + output_shapes[..., 1].astype(jnp.int32),
                i,
            )
            logits = jnp.take_along_axis(grid_logits, logits_index[..., None, None], axis=-2).squeeze(axis=-2)
            new_token = jnp.argmax(logits, axis=-1).astype(output_seq.dtype)
            output_seq = output_seq.at[..., 2 + i].set(new_token)  # +2 to skip the grid shapes
            return output_seq, None

        # Then predict the grid values.
        output_seq, _ = nn.scan(
            one_step,
            variable_broadcast="params",
            variable_carry="output_seq",
            split_rngs={"params": False},
        )(self.decoder, output_seq, jnp.arange(self.decoder.config.max_len))
        output_grids = jnp.reshape(output_seq[..., 2:], (*input.shape[:-2], *input.shape[-2:]))

        return output_grids, output_shapes

    @staticmethod
    def _flatten_input_output_for_decoding(
        pairs: chex.Array, grid_shapes: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        flattened_pairs = jnp.reshape(pairs, (*pairs.shape[:-3], -1, 2))
        input_seq = jnp.concatenate([grid_shapes[..., 0, :], flattened_pairs[..., 0]], axis=-1)
        output_seq = jnp.concatenate([grid_shapes[..., 1, :], flattened_pairs[..., 1]], axis=-1)
        return input_seq, output_seq

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
            print(f"Preparing latents for search, Latents shape: {latents.shape}")
            mean_latent = latents
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

    @staticmethod
    def _select_best_and_second_best_latents(
        log_probs: chex.Array, # Shape: (*batch_dims, K)
        latents: chex.Array    # Shape: (*batch_dims, K, H)
    ) -> tuple[chex.Array, chex.Array]:
        k_dim_log_probs = log_probs.ndim - 1
        k_dim_latents = latents.ndim - 2

        assert log_probs.shape[:-1] == latents.shape[:-2], \
            f"Batch dimensions mismatch: log_probs {log_probs.shape[:-1]}, latents {latents.shape[:-2]}"

        sorted_indices = jnp.argsort(log_probs, axis=k_dim_log_probs, descending=True)
        
        best_k_index_slice = jax.lax.slice_in_dim(sorted_indices, start_index=0, limit_index=1, axis=k_dim_log_probs)
        best_k_index_expanded = best_k_index_slice[..., None] 
        
        best_ctx = jnp.take_along_axis(latents, best_k_index_expanded, axis=k_dim_latents).squeeze(axis=k_dim_latents)
        
        second_best_ctx = best_ctx 
        if sorted_indices.shape[k_dim_log_probs] > 1:
            second_k_index_slice = jax.lax.slice_in_dim(sorted_indices, start_index=1, limit_index=2, axis=k_dim_log_probs)
            second_k_index_expanded = second_k_index_slice[..., None]
            second_best_ctx = jnp.take_along_axis(latents, second_k_index_expanded, axis=k_dim_latents).squeeze(axis=k_dim_latents)
            
        return best_ctx, second_best_ctx

    def _compute_log_probs(
        self, row_logits, col_logits, grid_logits, output_seq, grid_log_prob_weight: float = 1.0,
        use_product_score: bool = False 
    ) -> chex.Array: 
        max_cols = self.decoder.config.max_cols
        num_rows, num_cols = output_seq[..., 0].astype(jnp.int32), output_seq[..., 1].astype(jnp.int32)
        
        row_all_lp = jax.nn.log_softmax(row_logits, axis=-1)
        row_lp = jnp.take_along_axis(row_all_lp, num_rows[..., None] - 1, axis=-1).squeeze(axis=-1)
        
        col_all_lp = jax.nn.log_softmax(col_logits, axis=-1)
        col_lp = jnp.take_along_axis(col_all_lp, num_cols[..., None] - 1, axis=-1).squeeze(axis=-1)
        
        last_non_pad_lgts = self._get_last_non_padded_logits(grid_logits, num_cols[..., None, None])
        if max_cols > 0 and grid_logits.shape[-2] >= max_cols and \
           hasattr(last_non_pad_lgts, 'shape') and last_non_pad_lgts.shape[-2] > 0:
            grid_logits = grid_logits.at[..., max_cols::max_cols, :].set(last_non_pad_lgts)

        grid_all_lp = jax.nn.log_softmax(grid_logits, axis=-1)
        grid_tok_lp = jnp.take_along_axis(grid_all_lp, output_seq[..., 2:, None].astype(jnp.int32), axis=-1).squeeze(axis=-1)
        avg_grid_lp = self._normalized_mean_over_sequence(grid_tok_lp, num_rows, num_cols)

        lp_per_pair = row_lp + col_lp + grid_log_prob_weight * avg_grid_lp
        
        n_p_axis = -1 
        if lp_per_pair.ndim == 0: return lp_per_pair

        if use_product_score: 
	        total_lp = jnp.log(jnp.clip(jnp.exp(lp_per_pair).prod(axis=n_p_axis), a_min=1e-10))
        else:
    	    total_lp = jnp.sum(lp_per_pair, axis=n_p_axis) 
        return total_lp

    def _get_last_non_padded_logits(self, grid_logits: chex.Array, num_cols: chex.Array) -> chex.Array:
        # VMAP version
        max_rows_cfg, max_cols_cfg = self.decoder.config.max_rows, self.decoder.config.max_cols

        if max_rows_cfg <= 1:
            return jnp.zeros((*grid_logits.shape[:-2], 0, grid_logits.shape[-1]), dtype=grid_logits.dtype)

        num_cols_int = num_cols.astype(jnp.int32) 
        prev_row_indices_0_indexed = jnp.arange(0, max_rows_cfg - 1) 


        def get_one_last_logit(prev_row_idx_scalar):
            index_to_gather = prev_row_idx_scalar * max_cols_cfg + (num_cols_int - 1)
            seq_len_of_grid_logits = grid_logits.shape[-2]
            safe_index_to_gather = jnp.clip(index_to_gather, 0, seq_len_of_grid_logits - 1)
            
            seq_dim_axis = grid_logits.ndim - 2
            one_logit_vector = jnp.take_along_axis(grid_logits, safe_index_to_gather, axis=seq_dim_axis)
            return one_logit_vector

        all_logit_vectors_vmapped = jax.vmap(get_one_last_logit, in_axes=0)(prev_row_indices_0_indexed)
        # Shape: (L, *batch_dims_grid, 1, VocabSize)
        
        # Squeeze the singleton dimension (axis=-2 for the (1) dim from num_cols_int's original trailing singletons)
        all_logit_vectors_squeezed = all_logit_vectors_vmapped.squeeze(axis=-2) 
        # Shape: (L, *batch_dims_grid, VocabSize)
        
        num_grid_batch_dims = grid_logits.ndim - 2 # Number of dims like B, N before Seq, Vocab
        
        # Transpose to move L to be the sequence dimension for concatenation: (*batch_dims_grid, L, VocabSize)
        current_axes_order = list(range(all_logit_vectors_squeezed.ndim))
        
        if num_grid_batch_dims == 0: 
            perm = current_axes_order 
        elif num_grid_batch_dims > 0 :
            batch_dim_axes = current_axes_order[1 : num_grid_batch_dims+1] 
            l_axis = [current_axes_order[0]] 
            # Any remaining axes (e.g., Vocab)
            remaining_axes = current_axes_order[num_grid_batch_dims+1:] 
            perm = batch_dim_axes + l_axis + remaining_axes
        else: 
            perm = current_axes_order

        gathered_logits_final_shape = jnp.transpose(all_logit_vectors_squeezed, axes=perm)
        return gathered_logits_final_shape
    
    def _get_random_search_context(
        self, latents_to_search_from, pairs_for_eval, grid_shapes_for_eval, key,
        num_samples: int, scale: float, scan_batch_size: Optional[int] = None, 
        include_mean_latent: bool = True, include_all_latents: bool = False, **kwargs
    ) -> tuple[chex.Array, chex.Array]:

        perturb_dict = {"num_samples": num_samples, "scale": scale} if num_samples > 0 else None
        all_candidate_latents = LPN._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents_to_search_from, 
            random_perturbation=perturb_dict, key=key
        )
        
        input_seq_eval, output_seq_eval = LPN._flatten_input_output_for_decoding(
            pairs_for_eval, grid_shapes_for_eval
        )

        def log_probs_fn_search_local(candidate_k_latent, inp_seq_Neval, out_seq_Neval, decoder_instance):
            num_N_eval = inp_seq_Neval.shape[-2] 
            
            axis_to_expand = candidate_k_latent.ndim -1 
            cand_exp_local = jnp.expand_dims(candidate_k_latent, axis=axis_to_expand)
            
            tile_reps_local = [1]*cand_exp_local.ndim
            tile_reps_local[axis_to_expand] = num_N_eval
            latents_k_for_Neval = jnp.tile(cand_exp_local, tile_reps_local)

            r_logits, c_logits, g_logits = decoder_instance(
                inp_seq_Neval, out_seq_Neval, latents_k_for_Neval, dropout_eval=True
            )
            log_probs_val = self._compute_log_probs(r_logits, c_logits, g_logits, out_seq_Neval)
            return jnp.sum(log_probs_val) 
        
        vmap_axis_for_K_dim = latents_to_search_from.ndim - 2 # K is before H

        log_probs_all_K = jax.vmap(
            log_probs_fn_search_local, 
            in_axes=(vmap_axis_for_K_dim, None, None, None),
            out_axes=vmap_axis_for_K_dim 
        )(all_candidate_latents, input_seq_eval, output_seq_eval, self.decoder)
        
        best_ctx, second_best_ctx = LPN._select_best_and_second_best_latents(
            log_probs_all_K, all_candidate_latents
        )
        return best_ctx, second_best_ctx

    def _get_gradient_ascent_context(
        self,
        latents: chex.Array,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        key: Optional[chex.PRNGKey],
        num_steps: int,
        lr: float,
        lr_schedule: bool = False,
        lr_schedule_exponent: float = 0.5,
        accumulate_gradients_decoder_pairs: bool = False,
        scan_gradients_latents: bool = False,
        optimizer: Literal["sgd", "adam"] = "sgd",
        optimizer_kwargs: Optional[dict] = None,
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        random_perturbation: Optional[dict] = None,
        stop_gradient_latent_move: bool = True,
        **kwargs,
    ) -> tuple[chex.Array, chex.Array]:
        """Returns the best two contexts using a gradient ascent algorithm.

        Args:
            latents: latents from the encoder. Shape (*B, N, H).
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, N, 2, 2). Expects grid shapes values
                to be in [1, max_rows] and [1, max_cols].
            num_steps: number of gradient ascent steps.
            lr: learning rate for the gradient ascent.
            lr_schedule: if true, uses a cosine learning rate schedule, default to false.
            lr_schedule_exponent: exponent for the cosine learning rate schedule, default to 0.5.
            accumulate_gradients_decoder_pairs: if true, accumulates the gradients over the pairs, default to
                false.
            scan_gradients_latents: if true, scans the gradients over the latents, otherwise, use vmap,
                default to false.
            optimizer: optimizer to use for the gradient ascent. Can be "sgd" or "adam", default to "sgd".
            optimizer_kwargs: additional keyword arguments for the optimizer (e.g. b1, b2, eps for adam).
            include_mean_latent: if true (default to true), includes the mean latent in the latents from which
                to start the gradient ascent.
            include_all_latents: if true (default to false), includes all the pair latents in the latents from
                which to start the gradient ascent.
            random_perturbation: dictionary of random perturbation arguments. If not None, the following
                arguments are required:
                - num_samples: number of random samples to generate around the mean latent.
                - scale: Gaussian scale of the random perturbations.
            stop_gradient_latent_move: if true (default to true), do not propagate the loss gradient through
                the latent modification from the gradient ascent.

        Returns:
            best_context: best context. Shape (*B, H).
            second_best_context: second best context. Shape (*B, H).
        """
        latents = self._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents, random_perturbation, key
        )

        # Flatten input/output for decoding likelihood
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        def log_probs_fn(
            latents: chex.Array, input_seq: chex.Array, output_seq: chex.Array, decoder: DecoderTransformer
        ) -> chex.Array:
            # Use the same latent for all pairs of the same task.
            latents = latents[..., None, :].repeat(output_seq.shape[-2], axis=-2)
            row_logits, col_logits, grid_logits = decoder(input_seq, output_seq, latents, dropout_eval=True)
            log_probs = self._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)
            return log_probs

        value_and_grad_log_probs_fn = jax.vmap(
            jax.value_and_grad(log_probs_fn), in_axes=(-2, None, None, None), out_axes=(-1, -2)
        )
        # Add vmaps for batch dimensions
        for batch_dim in range(input_seq[..., 0, 0].ndim):
            value_and_grad_log_probs_fn = jax.vmap(value_and_grad_log_probs_fn, in_axes=(0, 0, 0, None))

        vmap_log_probs_fn = jax.vmap(log_probs_fn, in_axes=(-2, None, None, None), out_axes=-1)

        if accumulate_gradients_decoder_pairs:

            def wrap_value_and_grad(value_and_grad_log_probs):
                def wrapped(latents, input_seq, output_seq, decoder):
                    def body_fn(decoder, carry, seqs):
                        log_probs, grads = carry
                        log_probs_i, grads_i = value_and_grad_log_probs(
                            latents, seqs[0][..., None, :], seqs[1][..., None, :], decoder
                        )
                        return (log_probs + log_probs_i, grads + grads_i), None

                    init_carry = (jnp.zeros_like(latents[..., 0]), jnp.zeros_like(latents))
                    (log_probs, grads), _ = nn.scan(
                        body_fn,
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                    )(decoder, init_carry, (input_seq, output_seq))

                    return log_probs, grads

                return wrapped

            def wrap_log_prob(log_probs_fn):
                def wrapped(latents, input_seq, output_seq, decoder):
                    log_probs, _ = nn.scan(
                        lambda decoder, log_prob, seqs: (
                            log_prob
                            + log_probs_fn(latents, seqs[0][..., None, :], seqs[1][..., None, :], decoder),
                            None,
                        ),
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                    )(decoder, jnp.zeros_like(latents[..., 0]), (input_seq, output_seq))
                    return log_probs

                return wrapped

            value_and_grad_log_probs_fn = wrap_value_and_grad(value_and_grad_log_probs_fn)
            vmap_log_probs_fn = wrap_log_prob(vmap_log_probs_fn)

        if scan_gradients_latents:

            def wrap_value_and_grad(value_and_grad_log_probs):
                def wrapped(latents, input_seq, output_seq, decoder):
                    _, (log_probs, grads) = nn.scan(
                        lambda decoder, _, latent: (
                            _,
                            value_and_grad_log_probs(latent[..., None, :], input_seq, output_seq, decoder),
                        ),
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                        out_axes=(-1, -2),
                    )(decoder, None, latents)
                    return jnp.squeeze(log_probs, axis=-2), jnp.squeeze(grads, axis=-3)

                return wrapped

            def wrap_log_prob(log_probs_fn):
                def wrapped(latents, input_seq, output_seq, decoder):
                    _, log_probs = nn.scan(
                        lambda decoder, _, latent: (
                            _,
                            log_probs_fn(latent[..., None, :], input_seq, output_seq, decoder),
                        ),
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                        out_axes=-1,
                    )(decoder, None, latents)
                    return jnp.squeeze(log_probs, axis=-2)

                return wrapped

            value_and_grad_log_probs_fn = wrap_value_and_grad(value_and_grad_log_probs_fn)
            vmap_log_probs_fn = wrap_log_prob(vmap_log_probs_fn)

        if lr_schedule:
            lr = optax.cosine_decay_schedule(lr, num_steps, exponent=lr_schedule_exponent)
        if optimizer == "sgd":
            optimizer: optax.GradientTransformation = optax.chain(
                optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr, **(optimizer_kwargs or {}))
            )
        elif optimizer == "adam":
            optimizer: optax.GradientTransformation = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=lr, eps_root=1e-8, **(optimizer_kwargs or {})),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        opt_state = optimizer.init(latents)

        def update_latents(decoder, carry, _):
            latents, opt_state = carry
            log_probs, grads = value_and_grad_log_probs_fn(latents, input_seq, output_seq, decoder)
            assert grads.shape == latents.shape
            if stop_gradient_latent_move:
                grads = jax.lax.stop_gradient(grads)
            updates, opt_state = optimizer.update(-grads, opt_state)
            latents += updates
            return (latents, opt_state), (latents, log_probs)

        (last_latents, _), (all_latents, all_log_probs) = nn.scan(
            update_latents,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=num_steps,
            out_axes=(-2, -1),
        )(self.decoder, (latents, opt_state), None)

        # Concatenate original latents to all_latents and flatten all the latents.
        latents = jnp.concatenate([latents[..., None, :], all_latents], axis=-2).reshape(
            *latents.shape[:-2], -1, latents.shape[-1]
        )
        # Get all log_probs
        last_log_probs = vmap_log_probs_fn(last_latents, input_seq, output_seq, self.decoder)

        log_probs = jnp.concatenate([all_log_probs, last_log_probs[..., None]], axis=-1).reshape(
            *last_log_probs.shape[:-1], -1
        )

        best_context, second_best_context = self._select_best_and_second_best_latents(log_probs, latents)

        return best_context, second_best_context

# Main block
if __name__ == "__main__":
    from src.models.utils import TransformerLayerConfig 