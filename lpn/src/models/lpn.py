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
        
        if use_cross_attention:
            H = example_latents.shape[-1]
            sqrt_dh = jnp.sqrt(float(H))
            query = example_latents.mean(axis=-2, keepdims=True)
            key_val_attn = example_latents
            attn_scores = jnp.einsum('...qh,...kh->...qk', query, key_val_attn) / sqrt_dh
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            attended_context_gen = jnp.einsum('...qk,...kh->...qh', attn_weights, key_val_attn).squeeze(axis=-2)
            source_latents_for_gen_modes = attended_context_gen[..., None, :] 
        
        if mode == "mean":
            final_gen_context = source_latents_for_gen_modes.mean(axis=-2) 
            first_context, second_context = final_gen_context, final_gen_context
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
        self, context: chex.Array, input_grid: chex.Array, input_grid_shape: chex.Array, dropout_eval: bool
    ) -> tuple[chex.Array, chex.Array]:
        batch_dims = input_grid.shape[:-2] 
        H_dim = context.shape[-1]

        flattened_input_grid = jnp.reshape(input_grid, (*batch_dims, -1))
        input_seq_single = jnp.concatenate([input_grid_shape, flattened_input_grid], axis=-1)
        output_seq_single_init = jnp.zeros_like(input_seq_single).at[..., :2].set(1)

        s_len = input_seq_single.shape[-1]
        
        input_seq_N1 = jnp.reshape(input_seq_single, (*batch_dims, 1, s_len))
        context_N1 = jnp.reshape(context, (*batch_dims, 1, H_dim))
        current_output_seq_N1 = jnp.reshape(output_seq_single_init, (*batch_dims, 1, s_len))

        def grid_shape_step_fn(output_seq_step_N1: chex.Array, row_flag: bool) -> chex.Array:
            pred_row_logits_N1, pred_col_logits_N1, _ = self.decoder(
                input_seq_N1, output_seq_step_N1, context_N1, dropout_eval
            )
            target_logits_N1 = pred_row_logits_N1 if row_flag else pred_col_logits_N1
            axis_to_squeeze_N = len(batch_dims) 
            target_logits_single = target_logits_N1.squeeze(axis=axis_to_squeeze_N) 
            
            new_token_val_single = jnp.argmax(target_logits_single, axis=-1).astype(output_seq_single_init.dtype) + 1
            new_token_val_expanded = new_token_val_single[..., None]

            token_idx = 0 if row_flag else 1
            return output_seq_step_N1.at[..., token_idx].set(new_token_val_expanded)

        current_output_seq_N1 = grid_shape_step_fn(current_output_seq_N1, row_flag=True)
        current_output_seq_N1 = grid_shape_step_fn(current_output_seq_N1, row_flag=False)
        
        axis_to_squeeze_N_shape = len(batch_dims)
        output_shapes_predicted = current_output_seq_N1[..., :2].squeeze(axis=axis_to_squeeze_N_shape)
        max_cols_cfg = self.decoder.config.max_cols

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
            sel_logits_N1 = jnp.take_along_axis(pred_grid_lgts_N1, idx_for_take_N1, axis=len(batch_dims) + 1) 
            
            axis_N_sqz = len(batch_dims)
            axis_S_token_sqz = len(batch_dims) + 1 
            sel_logits_single = sel_logits_N1.squeeze(axis=(axis_N_sqz, axis_S_token_sqz))
            
            new_grid_token_val_single = jnp.argmax(sel_logits_single, axis=-1).astype(output_seq_single_init.dtype)
            new_grid_token_val_expanded = new_grid_token_val_single[..., None]
            
            updated_loop_output_seq_N1 = loop_carry_output_seq_N1.at[..., 2 + grid_token_idx].set(new_grid_token_val_expanded)
            return updated_loop_output_seq_N1, None

        final_gen_output_seq_N1, _ = nn.scan(
            scan_step_fn, 
            variable_broadcast=["params", "batch_stats"], 
            split_rngs={"params": False}, 
        )(current_output_seq_N1, jnp.arange(self.decoder.config.max_len))

        axis_to_squeeze_N_final = len(batch_dims)
        final_gen_output_seq_single = final_gen_output_seq_N1.squeeze(axis=axis_to_squeeze_N_final)
        output_grids_final = jnp.reshape(final_gen_output_seq_single[..., 2:], input_grid.shape)

        return output_grids_final, output_shapes_predicted

    @staticmethod
    def _flatten_input_output_for_decoding(
        pairs: chex.Array, grid_shapes: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        flattened_pairs = jnp.reshape(pairs, (*pairs.shape[:-3], -1, 2))
        input_seq = jnp.concatenate([grid_shapes[..., 0, :], flattened_pairs[..., 0]], axis=-1)
        output_seq = jnp.concatenate([grid_shapes[..., 1, :], flattened_pairs[..., 1]], axis=-1)
        return input_seq, output_seq

    @staticmethod
    def _prepare_latents_before_search(
        include_mean_latent: bool, include_all_latents: bool,
        latents: chex.Array, random_perturbation: Optional[dict] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        prep_latents_list = []
        H_dim = latents.shape[-1] if latents.ndim > 0 and latents.shape[-1] > 0 else 0

        if latents.shape[-2] > 0 : 
            if include_all_latents: prep_latents_list.append(latents)
            if include_mean_latent: 
                mean_latent = latents.mean(axis=-2, keepdims=True)
                prep_latents_list.append(mean_latent)
            if not prep_latents_list and latents.shape[-2] > 0: 
                 prep_latents_list.append(latents)
        
        if prep_latents_list:
            current_prep_latents = jnp.concatenate(prep_latents_list, axis=-2)
        else: 
            if H_dim == 0 and random_perturbation is None: 
                raise ValueError("Cannot prepare latents: H_dim is 0 and no random_perturbation.")
            current_prep_latents = jnp.zeros((*latents.shape[:-2], 0, H_dim if H_dim > 0 else 1 ), dtype=latents.dtype)

        if random_perturbation is not None:
            assert key is not None, "Key required for random perturbation."
            num_rand_samples = random_perturbation["num_samples"]
            scale_rand = random_perturbation["scale"]

            if latents.shape[-2] > 0:
                perturb_base = latents.mean(axis=-2, keepdims=True)
            else: 
                if H_dim == 0: raise ValueError("Cannot determine H for zero-perturbation base when K=0.")
                perturb_base = jnp.zeros((*latents.shape[:-2], 1, H_dim), dtype=latents.dtype)

            random_vectors = jax.random.normal(key, (*perturb_base.shape[:-2], num_rand_samples, perturb_base.shape[-1]))
            perturbed_random_latents = perturb_base + scale_rand * random_vectors
            
            if current_prep_latents.shape[-2] > 0:
                 current_prep_latents = jnp.concatenate([current_prep_latents, perturbed_random_latents], axis=-2)
            else: 
                 current_prep_latents = perturbed_random_latents
        
        if current_prep_latents.shape[-2] == 0: 
            raise ValueError("No latents prepared for search. Final K=0.")
        return current_prep_latents

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

        # --- Start jax.debug.print block ---
        # from jax.debug import print as jax_print # Make sure this is imported at module level
        # jax_print("--- _get_last_non_padded_logits ---")
        # jax_print("grid_logits shape: {}", grid_logits.shape)
        # jax_print("num_cols_int shape: {}", num_cols_int.shape)
        # jax_print("prev_row_indices_0_indexed shape: {}", prev_row_indices_0_indexed.shape)
        # --- End jax.debug.print block ---

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
        self, latents_to_optimize_from, pairs_for_eval, grid_shapes_for_eval, key,
        num_steps: int, lr: float, lr_schedule: bool = False, lr_schedule_exponent: float = 0.5,
        optimizer: Literal["sgd", "adam"] = "sgd", optimizer_kwargs: Optional[dict] = None,
        include_mean_latent: bool = True, include_all_latents: bool = False,
        random_perturbation: Optional[dict] = None, stop_gradient_latent_move: bool = True, **kwargs
    ) -> tuple[chex.Array, chex.Array]:

        latents_prepared = LPN._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents_to_optimize_from, 
            random_perturbation, key
        )

        input_seq_eval, output_seq_eval = LPN._flatten_input_output_for_decoding(
            pairs_for_eval, grid_shapes_for_eval
        )

        def log_probs_fn_ga_local(one_k_latent, inp_Neval, out_Neval, decoder_inst):
            num_N_eval = inp_Neval.shape[-2]
            axis_to_expand_ga = one_k_latent.ndim -1
            k_exp_ga = jnp.expand_dims(one_k_latent, axis=axis_to_expand_ga)
            
            tile_reps_ga = [1]*k_exp_ga.ndim
            tile_reps_ga[axis_to_expand_ga] = num_N_eval
            latents_k_for_Neval = jnp.tile(k_exp_ga, tile_reps_ga)

            r_lg, c_lg, g_lg = decoder_inst(inp_Neval, out_Neval, latents_k_for_Neval, dropout_eval=True)
            log_probs_val = self._compute_log_probs(r_lg, c_lg, g_lg, out_Neval, 
                                           use_product_score=kwargs.get("use_product_score",False))
            return jnp.sum(log_probs_val) 


        vmap_axis_K_prep_ga = latents_prepared.ndim - 2 # K is before H

        value_and_grad_vmapped = jax.vmap(
            jax.value_and_grad(log_probs_fn_ga_local),
            in_axes=(vmap_axis_K_prep_ga, None, None, None),
            out_axes=(vmap_axis_K_prep_ga, vmap_axis_K_prep_ga)
        )

        current_lr_val = optax.cosine_decay_schedule(lr, num_steps, exponent=lr_schedule_exponent) if lr_schedule else lr
        opt_chain_list = [optax.clip_by_global_norm(1.0)]
        if optimizer == "sgd": opt_chain_list.append(optax.sgd(current_lr_val, **(optimizer_kwargs or {})))
        elif optimizer == "adam": opt_chain_list.append(optax.adam(current_lr_val, eps_root=1e-8, **(optimizer_kwargs or {})))
        else: raise ValueError(f"Unsupported optimizer: {optimizer}")
        optax_opt = optax.chain(*opt_chain_list)
        
        opt_state = optax_opt.init(latents_prepared)

        ga_latents = latents_prepared
        # Compute initial log_probs for t=0 for history (value part of value_and_grad)
        initial_log_probs_k = value_and_grad_vmapped(ga_latents, input_seq_eval, output_seq_eval, self.decoder)[0]
        history_latents = [ga_latents]
        history_log_probs = [initial_log_probs_k]


        for _ in range(num_steps):
            log_probs_val_k, grads_val_k = value_and_grad_vmapped(ga_latents, input_seq_eval, output_seq_eval, self.decoder)
            if stop_gradient_latent_move: grads_val_k = jax.lax.stop_gradient(grads_val_k)
            updates_val_k, opt_state = optax_opt.update(-grads_val_k, opt_state, ga_latents)
            ga_latents = ga_latents + updates_val_k 
            history_latents.append(ga_latents)
            # Store log_probs of the *new* latents (value part of value_and_grad for next iter, or recompute)
            current_step_log_probs_k = value_and_grad_vmapped(ga_latents, input_seq_eval, output_seq_eval, self.decoder)[0]
            history_log_probs.append(current_step_log_probs_k)
            
        stack_axis = latents_prepared.ndim - 1 # Stack along a new dimension after K, before H

        all_versions_latents = jnp.stack(history_latents, axis=stack_axis) 
        # Shape: (*B, K, num_steps+1, H)
        # Reshape to (*B, K*(num_steps+1), H)
        final_shape_latents = (*latents_prepared.shape[:-2], -1, latents_prepared.shape[-1])
        collated_candidate_latents = jnp.reshape(all_versions_latents, final_shape_latents)

        all_versions_log_probs = jnp.stack(history_log_probs, axis=stack_axis)
        # Shape: (*B, K, num_steps+1)
        # Reshape to (*B, K*(num_steps+1))
        final_shape_log_probs = (*history_log_probs[0].shape[:-1], -1) # Batch dims from log_probs, then K*steps
        collated_log_probs = jnp.reshape(all_versions_log_probs, final_shape_log_probs)
        
        best_ctx, second_best_ctx = LPN._select_best_and_second_best_latents(
            collated_log_probs, collated_candidate_latents
        )
        return best_ctx, second_best_ctx

# Main block
if __name__ == "__main__":
    from src.models.utils import TransformerLayerConfig 

    batch_size = 4; mini_batch_size_N = 3; max_rows_val = 5; max_cols_val = 5
    vocab_size_val = 10; hidden_size_H = 96 

    encoder_config_test = EncoderTransformerConfig(
        vocab_size=vocab_size_val, max_rows=max_rows_val, max_cols=max_cols_val,
        transformer_layer=TransformerLayerConfig(dropout_rate=0.0, hidden_size=hidden_size_H, num_heads=4, mlp_size=128),
        variational=True, output_size=hidden_size_H 
    )
    decoder_config_test = DecoderTransformerConfig(
        vocab_size=vocab_size_val, max_rows=max_rows_val, max_cols=max_cols_val,
        transformer_layer=TransformerLayerConfig(dropout_rate=0.0, hidden_size=hidden_size_H, num_heads=4, mlp_size=128),
        hidden_size=hidden_size_H, 
    )
    encoder_test_module = EncoderTransformer(encoder_config_test)
    decoder_test_module = DecoderTransformer(decoder_config_test)
    lpn_test_model = LPN(encoder=encoder_test_module, decoder=decoder_test_module)
    key_main = jax.random.PRNGKey(0)
    key_pairs, key_shapes, key_init_master, key_dropout_master, key_gen_main = jax.random.split(key_main, 5)

    test_pairs = jax.random.randint(key_pairs, (batch_size, mini_batch_size_N, max_rows_val, max_cols_val, 2), 0, vocab_size_val)
    test_grid_shapes = jax.random.randint(key_shapes, (batch_size, mini_batch_size_N, 2, 2), 1, min(max_rows_val, max_cols_val) + 1)
    
    print("Initializing LPN model...")
    init_rng_keys = ['params', 'latents', 'latents_init', 'random_search', 'gradient_ascent_random_perturbation', 'dropout']
    init_rng_values = jax.random.split(key_init_master, len(init_rng_keys))
    init_rngs = {name: val for name, val in zip(init_rng_keys, init_rng_values)}
    
    variables_lpn = lpn_test_model.init(
        init_rngs, test_pairs, test_grid_shapes, 
        dropout_eval=False, mode="mean", use_cross_attention=False, 
        prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
    )
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables_lpn["params"]))
    print(f"LPN Number of parameters: {num_params:,}")

    apply_rng_keys = ['dropout', 'latents', 'latents_init', 'random_search', 'gradient_ascent_random_perturbation']
    apply_rng_values = jax.random.split(key_dropout_master, len(apply_rng_keys))
    apply_rngs = {name: val for name, val in zip(apply_rng_keys, apply_rng_values)}

    print("\nTesting __call__ with use_cross_attention=True, mode='mean'")
    loss_ca_mean, _ = lpn_test_model.apply(
        variables_lpn, test_pairs, test_grid_shapes,
        dropout_eval=False, mode="mean", use_cross_attention=True,
        rngs=apply_rngs, prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
    )
    print(f"CA Mean Loss: {loss_ca_mean}")

    print("\nTesting generate_output with use_cross_attention=True, mode='first'")
    gen_input_grid = test_pairs[:, 0, ..., 0] 
    gen_input_grid_shape = test_grid_shapes[:, 0, 0, :] 

    gen_out_grids, gen_out_shapes, _ = lpn_test_model.apply(
        variables_lpn, method=lpn_test_model.generate_output, 
        pairs=test_pairs, grid_shapes=test_grid_shapes, 
        input=gen_input_grid, input_grid_shape=gen_input_grid_shape, 
        key=key_gen_main, dropout_eval=True, mode="first", 
        use_cross_attention=True, return_two_best=False,
        rngs=apply_rngs 
    )
    print(f"Generated output grids shape (CA, first): {gen_out_grids.shape}")
    print(f"Generated output shapes shape (CA, first): {gen_out_shapes.shape}")

    @partial(jax.jit, static_argnames=["dropout_eval", "mode", "use_cross_attention"])
    def apply_lpn_jitted(variables, p, gs, drp_eval, m, use_ca, rngs_dict, prior_c, pairwise_c):
        return lpn_test_model.apply(variables, p, gs, dropout_eval=drp_eval, mode=m, use_cross_attention=use_ca,
                                    rngs=rngs_dict, prior_kl_coeff=prior_c, pairwise_kl_coeff=pairwise_c)

    print("\nOriginal Mean Loss (CA=False):") 
    loss_orig_mean, _ = apply_lpn_jitted(
        variables_lpn, test_pairs, test_grid_shapes,
        drp_eval=False, m="mean", use_ca=False, 
        rngs_dict=apply_rngs, prior_c=1e-4, pairwise_c=1e-4,
    )
    print(f"Original Mean Loss: {loss_orig_mean}")
    print("--- End of __main__ example ---")