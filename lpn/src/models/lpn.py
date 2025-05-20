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
            # print(f"LPN __call__: using cross attention: {use_cross_attention}") 
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
                key_ga = self.make_rng("gradient_ascent_random_perturbation") if mode_kwargs.get("random_perturbation") else None
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
            if last_non_padded_logits.shape[-2] > 0 : 
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
            assert _key_for_search_or_ga is not None, "Key required for gradient_ascent random_perturbation."
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
            # Squeeze the N=1 dimension (which is at axis `len(batch_dims)`)
            # If batch_dims is (B1,B2), N=1 is at axis 2. target_logits_N1 (B1,B2,1,V). Squeeze axis 2.
            axis_to_squeeze_N = len(batch_dims) 
            target_logits_single = target_logits_N1.squeeze(axis=axis_to_squeeze_N) 
            
            new_token_val_single = jnp.argmax(target_logits_single, axis=-1).astype(output_seq_single_init.dtype) + 1
            # new_token_val_single is (*batch_dims). Slice is (*batch_dims, 1). Expand last dim.
            new_token_val_expanded = new_token_val_single[..., None]

            token_idx = 0 if row_flag else 1
            return output_seq_step_N1.at[..., token_idx].set(new_token_val_expanded)

        current_output_seq_N1 = grid_shape_step_fn(current_output_seq_N1, row_flag=True)
        current_output_seq_N1 = grid_shape_step_fn(current_output_seq_N1, row_flag=False)
        
        # Squeeze N=1 dim for output_shapes_predicted
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
            sel_logits_N1 = jnp.take_along_axis(pred_grid_lgts_N1, idx_for_take_N1, axis=-2) # seq_dim is -2 for (*batch,N,S,V)
            
            # Squeeze N=1 dim (axis=len(batch_dims)) and token_seq_dim (axis=len(batch_dims)+1)
            axis_N_sqz = len(batch_dims)
            axis_S_token_sqz = len(batch_dims) + 1
            sel_logits_single = sel_logits_N1.squeeze(axis=(axis_N_sqz, axis_S_token_sqz))
            
            new_grid_token_val_single = jnp.argmax(sel_logits_single, axis=-1).astype(output_seq_single_init.dtype)
            new_grid_token_val_expanded = new_grid_token_val_single[..., None]
            
            updated_loop_output_seq_N1 = loop_carry_output_seq_N1.at[..., 2 + grid_token_idx].set(new_grid_token_val_expanded)
            return updated_loop_output_seq_N1, None

        final_gen_output_seq_N1, _ = nn.scan(
            scan_step_fn, variable_broadcast="params", split_rngs={"params": False},
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
        # Default H for perturbation if latents K=0. Needs better handling if decoder available.
        # For now, assume latents.shape[-1] is valid or an error will occur.
        default_H_if_K_is_zero = latents.shape[-1] if latents.ndim > 0 and latents.shape[-1] > 0 else 0


        if latents.shape[-2] > 0 : 
            if include_all_latents: prep_latents_list.append(latents)
            if include_mean_latent: 
                mean_latent = latents.mean(axis=-2, keepdims=True)
                prep_latents_list.append(mean_latent)
            if not prep_latents_list: prep_latents_list.append(latents)
        
        current_prep_latents = jnp.concatenate(prep_latents_list, axis=-2) if prep_latents_list else \
                               jnp.zeros((*latents.shape[:-2], 0, default_H_if_K_is_zero ), dtype=latents.dtype)

        if random_perturbation is not None:
            assert key is not None, "Key required for random perturbation."
            num_rand_samples = random_perturbation["num_samples"]
            scale_rand = random_perturbation["scale"]

            if latents.shape[-2] > 0:
                perturb_base = latents.mean(axis=-2, keepdims=True)
            else: # K=0, perturb around zero vector.
                  # Infer H from latents if possible (e.g. shape is (B,0,H)), else error or use default_H_if_K_is_zero
                H_for_zeros = latents.shape[-1] if latents.ndim > 1 and latents.shape[-1] > 0 else default_H_if_K_is_zero
                if H_for_zeros == 0: raise ValueError("Cannot determine H for zero-perturbation base.")
                perturb_base = jnp.zeros((*latents.shape[:-2], 1, H_for_zeros), dtype=latents.dtype)


            random_vectors = jax.random.normal(key, (*perturb_base.shape[:-2], num_rand_samples, perturb_base.shape[-1]))
            perturbed_random_latents = perturb_base + scale_rand * random_vectors
            
            current_prep_latents = jnp.concatenate([current_prep_latents, perturbed_random_latents], axis=-2) if current_prep_latents.shape[-2] > 0 else perturbed_random_latents
        
        if current_prep_latents.shape[-2] == 0:
            raise ValueError("No latents prepared for search. Check config.")
        return current_prep_latents

    @staticmethod
    def _select_best_and_second_best_latents(
        log_probs: chex.Array, latents: chex.Array 
    ) -> tuple[chex.Array, chex.Array]:
        # log_probs: (*batch_dims_lp, K)
        # latents: (*batch_dims_l, K, H)
        # Ensure batch_dims_lp == batch_dims_l
        
        k_dim_log_probs = -1 
        # k_dim_latents depends on how many batch_dims latents has. It's the dim before H.
        k_dim_latents = latents.ndim - 2

        sorted_indices = jnp.argsort(log_probs, axis=k_dim_log_probs, descending=True)
        
        # Expand indices for take_along_axis to select full H vectors
        # Index shape for take_along_axis needs to be broadcastable with latents,
        # selecting along k_dim_latents.
        # E.g., if latents is (B,K,H), indices (B,1) -> expanded to (B,1,1) for axis=1.
        best_idx_slice = sorted_indices[..., 0:1] # (*batch_dims_lp, 1)
        # Add dummy dims to match rank of latents up to H dim, for H selection.
        # Number of dummy dims = latents.ndim - best_idx_slice.ndim - 1 (for H)
        num_dummy_dims_for_H = latents.ndim - best_idx_slice.ndim -1
        best_idx_expanded = best_idx_slice
        for _ in range(num_dummy_dims_for_H): # Should be 0 if batch_dims match
            best_idx_expanded = best_idx_expanded[..., None]
        best_idx_expanded = best_idx_expanded[..., None] # Final dim for H

        best_ctx = jnp.take_along_axis(latents, best_idx_expanded, axis=k_dim_latents).squeeze(axis=k_dim_latents)
        
        second_best_ctx = best_ctx 
        if sorted_indices.shape[k_dim_log_probs] > 1:
            second_idx_slice = sorted_indices[..., 1:2]
            second_idx_expanded = second_idx_slice
            for _ in range(num_dummy_dims_for_H):
                second_idx_expanded = second_idx_expanded[..., None]
            second_idx_expanded = second_idx_expanded[..., None]
            second_best_ctx = jnp.take_along_axis(latents, second_idx_expanded, axis=k_dim_latents).squeeze(axis=k_dim_latents)
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
        if max_cols > 0 and grid_logits.shape[-2] >= max_cols and last_non_pad_lgts.shape[-2] > 0:
            grid_logits = grid_logits.at[..., max_cols::max_cols, :].set(last_non_pad_lgts)

        grid_all_lp = jax.nn.log_softmax(grid_logits, axis=-1)
        grid_tok_lp = jnp.take_along_axis(grid_all_lp, output_seq[..., 2:, None].astype(jnp.int32), axis=-1).squeeze(axis=-1)
        avg_grid_lp = self._normalized_mean_over_sequence(grid_tok_lp, num_rows, num_cols)

        lp_per_pair = row_lp + col_lp + grid_log_prob_weight * avg_grid_lp
        
        # Sum over N_p (pairs for eval dimension, which is axis -1 of lp_per_pair after batch dims)
        # If lp_per_pair is (*B_outer, N_p), sum over N_p.
        # B_outer could be empty if not batched further.
        n_p_axis = -1 if lp_per_pair.ndim > 0 else 0 # Handle scalar case (not expected)
        if lp_per_pair.ndim == 0: # Scalar output, no sum needed
             return lp_per_pair

        if use_product_score: 
	        total_lp = jnp.log(jnp.clip(jnp.exp(lp_per_pair).prod(axis=n_p_axis), a_min=1e-10))
        else:
    	    total_lp = jnp.sum(lp_per_pair, axis=n_p_axis) 
        return total_lp

    def _get_last_non_padded_logits(self, grid_logits: chex.Array, num_cols: chex.Array) -> chex.Array:
        max_rows_cfg, max_cols_cfg = self.decoder.config.max_rows, self.decoder.config.max_cols

        if max_rows_cfg <= 1:
            # Handles cases where no previous rows exist to copy from.
            # Shape: (*batch_dims_of_grid_logits, 0, vocab_size)
            return jnp.zeros((*grid_logits.shape[:-2], 0, grid_logits.shape[-1]), dtype=grid_logits.dtype)

        num_cols_int = num_cols.astype(jnp.int32) # Shape e.g. (*B, N, 1, 1)

        # `i_values` represents the target row index (1-indexed) for which we are finding the EOL logit of the *previous* row.
        # The loop was `for i in range(1, max_rows_cfg)`.
        # These `i` values determine which previous row's end-of-line logit to pick.
        # The logit picked at unrolled loop step `i` is for filling the start of row `i` (0-indexed: row `i`).
        # So we need `max_rows_cfg - 1` such logits if `max_rows_cfg > 1`.
        
        # i_values: (max_rows_cfg - 1), e.g., [1, 2, ..., max_rows_cfg-1]
        # This `i` corresponds to the `i` in the original loop.
        i_values = jnp.arange(1, max_rows_cfg) # Shape: (max_rows_cfg-1,)

        # Expand i_values to be broadcastable with num_cols_int and grid_logits batch dims.
        # num_cols_int: (*B, N, 1, 1) or similar. Let's assume grid_logits is (*B_dims, Seq, Vocab)
        # and num_cols_int is (*B_dims, 1, 1) effectively for broadcasting.
        # We want indices to be calculated for each item in B_dims and for each i_value.
        # Target shape for `indices_to_gather`: (*B_dims, max_rows_cfg-1, 1)
        
        # Make i_values broadcastable: (1, ..., 1, max_rows_cfg-1, 1) to align with num_cols_int for broadcasting
        # num_cols_int could be (*batch_dims, 1, 1) where *batch_dims matches grid_logits.
        # i_values needs to be reshaped to allow broadcasting with num_cols_int.
        # Example: num_cols_int (B,N,1,1), i_values (R-1,). Reshape i_values to (1,1,R-1,1) for broadcasting.
        # Or, simpler: calculate indices assuming broadcasting works element-wise for i_values.
        
        # indices: shape will be `num_cols_int.shape` with an added dim for `i_values` if `i_values` is broadcast correctly.
        # `index = max_cols_cfg * i - (max_cols_cfg - num_cols_int)`
        # `i` is effectively (max_rows_cfg-1,). `num_cols_int` is (*B,N,1,1).
        # After broadcasting i_values, `indices` will be (*B,N,max_rows_cfg-1,1)
        indices = max_cols_cfg * i_values[..., None] - (max_cols_cfg - num_cols_int)
        
        # Clip indices
        seq_len_of_grid_logits = grid_logits.shape[-2]
        safe_indices = jnp.clip(indices, 0, seq_len_of_grid_logits - 1)
        # safe_indices is now, e.g., (*B, N, max_rows_cfg-1, 1)

        # grid_logits is (*B, N, Seq, Vocab)
        # We want to gather `max_rows_cfg-1` logit vectors.
        # `jnp.take_along_axis` will gather along axis -2 (Seq dim).
        # Input `grid_logits`:    (...,      Seq,          Vocab)
        # Input `safe_indices`:   (..., num_indices, 1)  (num_indices = max_rows_cfg-1)
        # Output should be:       (..., num_indices, Vocab)
        
        # Ensure safe_indices has a final singleton dim if it's just for indexing, not slicing.
        # The formula already makes it (*B,N,max_rows_cfg-1,1) so it's fine.
        
        gathered_logits = jnp.take_along_axis(grid_logits, safe_indices, axis=-2)
        # gathered_logits should be (*B, N, max_rows_cfg-1, Vocab)
        
        # The original code concatenates these along axis=-2.
        # This `gathered_logits` already has them effectively concatenated along the `max_rows_cfg-1` dimension.
        # If the usage `grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)`
        # expects `last_non_padded_logits` to have a flat sequence dimension that matches the number of
        # elements being set (which is `max_rows_cfg - 1` if `max_cols::max_cols` selects that many start-of-row positions),
        # then the shape `(*B, N, max_rows_cfg-1, Vocab)` is correct.
        
        return gathered_logits
        
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
            # candidate_k_latent: (*B_cand, H)
            # inp_seq_Neval: (*B_inp, N_eval, SeqLen)
            # Assume B_cand == B_inp. Tile candidate_k_latent for N_eval.
            # (*B, H) -> (*B, 1, H) -> (*B, N_eval, H)
            lead_dims_cand_local = candidate_k_latent.shape[:-1]
            cand_exp_local = jnp.expand_dims(candidate_k_latent, axis=len(lead_dims_cand_local))
            
            tile_reps_local = [1]*cand_exp_local.ndim
            tile_reps_local[len(lead_dims_cand_local)] = num_N_eval
            latents_k_for_Neval = jnp.tile(cand_exp_local, tile_reps_local)

            r_logits, c_logits, g_logits = decoder_instance(
                inp_seq_Neval, out_seq_Neval, latents_k_for_Neval, dropout_eval=True
            )
            return self._compute_log_probs(r_logits, c_logits, g_logits, out_seq_Neval)

        # Vmap over K_candidates dim. all_candidate_latents shape: (*B_common, K_cand, H)
        # inp_seq_eval shape: (*B_common, N_eval, Seq)
        # We need to determine the axis of K_cand.
        # If all_candidate_latents has leading batch_dims matching inp_seq_eval, K_cand is at len(batch_dims).
        
        # Determine number of shared batch dimensions
        num_shared_batch_dims = 0
        for d1, d2 in zip(all_candidate_latents.shape, input_seq_eval.shape):
            if d1 == d2:
                num_shared_batch_dims += 1
            else:
                break
        
        vmap_axis_for_K_dim = num_shared_batch_dims # K_cand is the first differing dimension

        log_probs_all_K = jax.vmap(
            log_probs_fn_search_local, 
            in_axes=(vmap_axis_for_K_dim, # Axis of K in all_candidate_latents
                     None,               # Broadcast input_seq_eval
                     None,               # Broadcast output_seq_eval
                     None),              # Broadcast decoder instance
            out_axes=vmap_axis_for_K_dim # Output has K at the same axis position
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
            lead_dims_k = one_k_latent.shape[:-1]
            k_exp = jnp.expand_dims(one_k_latent, axis=len(lead_dims_k))
            tile_reps_ga = [1]*k_exp.ndim; tile_reps_ga[len(lead_dims_k)] = num_N_eval
            latents_k_for_Neval = jnp.tile(k_exp, tile_reps_ga)
            r_lg, c_lg, g_lg = decoder_inst(inp_Neval, out_Neval, latents_k_for_Neval, dropout_eval=True)
            return self._compute_log_probs(r_lg, c_lg, g_lg, out_Neval, 
                                           use_product_score=kwargs.get("use_product_score",False))

        num_shared_batch_dims_ga = 0
        for d1, d2 in zip(latents_prepared.shape, input_seq_eval.shape):
            if d1 == d2: num_shared_batch_dims_ga += 1
            else: break
        vmap_axis_K_prep_ga = num_shared_batch_dims_ga

        value_and_grad_vmapped = jax.vmap(
            jax.value_and_grad(log_probs_fn_ga_local),
            in_axes=(vmap_axis_K_prep_ga, None, None, None),
            out_axes=(vmap_axis_K_prep_ga, vmap_axis_K_prep_ga)
        )

        current_lr_val = optax.cosine_decay_schedule(lr, num_steps, exponent=lr_schedule_exponent) if lr_schedule else lr
        opt_chain = [optax.clip_by_global_norm(1.0)]
        if optimizer == "sgd": opt_chain.append(optax.sgd(current_lr_val, **(optimizer_kwargs or {})))
        elif optimizer == "adam": opt_chain.append(optax.adam(current_lr_val, eps_root=1e-8, **(optimizer_kwargs or {})))
        else: raise ValueError(f"Unsupported optimizer: {optimizer}")
        optax_opt = optax.chain(*opt_chain)
        
        opt_state = optax_opt.init(latents_prepared)

        ga_latents = latents_prepared
        # Get initial log_probs for t=0
        initial_log_probs_k, _ = value_and_grad_vmapped(ga_latents, input_seq_eval, output_seq_eval, self.decoder)
        history_latents = [ga_latents]
        history_log_probs = [initial_log_probs_k]


        for i_step in range(num_steps):
            # Use log_probs from previous step if already computed, else compute.
            # For step 0, grads are from initial_log_probs_k computation.
            # For subsequent, compute fresh.
            log_probs_val_k, grads_val_k = value_and_grad_vmapped(ga_latents, input_seq_eval, output_seq_eval, self.decoder)
            if stop_gradient_latent_move: grads_val_k = jax.lax.stop_gradient(grads_val_k)
            
            updates_val_k, opt_state = optax_opt.update(-grads_val_k, opt_state, ga_latents)
            ga_latents = ga_latents + updates_val_k # Ensure new array for history
            history_latents.append(ga_latents)
            # Compute log_probs for the new latents to store.
            current_step_log_probs_k, _ = value_and_grad_vmapped(ga_latents, input_seq_eval, output_seq_eval, self.decoder)
            history_log_probs.append(current_step_log_probs_k)
            
        all_versions_latents = jnp.stack(history_latents, axis=vmap_axis_K_prep_ga + 1) 
        num_batch_dims_lp = len(latents_prepared.shape[:-2])
        final_shape_latents = (*latents_prepared.shape[:num_batch_dims_lp], -1, latents_prepared.shape[-1])
        collated_candidate_latents = jnp.reshape(all_versions_latents, final_shape_latents)

        all_versions_log_probs = jnp.stack(history_log_probs, axis=vmap_axis_K_prep_ga + 1)
        final_shape_log_probs = (*history_log_probs[0].shape[:num_batch_dims_lp], -1)
        collated_log_probs = jnp.reshape(all_versions_log_probs, final_shape_log_probs)
        
        best_ctx, second_best_ctx = LPN._select_best_and_second_best_latents(
            collated_log_probs, collated_candidate_latents
        )
        return best_ctx, second_best_ctx

# Main block from original file (for testing, if needed)
if __name__ == "__main__":
    from src.models.utils import TransformerLayerConfig

    batch_size = 4
    mini_batch_size_N = 3 
    max_rows_val = 5
    max_cols_val = 5
    vocab_size_val = 10
    hidden_size_H = 96 

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
    key_pairs, key_shapes, key_init, key_dropout, key_gen_main = jax.random.split(key_main, 5) # Renamed key_gen

    test_pairs = jax.random.randint(
        key_pairs, (batch_size, mini_batch_size_N, max_rows_val, max_cols_val, 2),
        minval=0, maxval=vocab_size_val,
    )
    test_grid_shapes = jax.random.randint(
        key_shapes, (batch_size, mini_batch_size_N, 2, 2),
        minval=1, maxval=min(max_rows_val, max_cols_val) + 1,
    )
    
    print("Initializing LPN model...")
    variables_lpn = lpn_test_model.init(
        key_init, test_pairs, test_grid_shapes, 
        dropout_eval=False, mode="mean", use_cross_attention=False, 
        prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
    )
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables_lpn["params"]))
    print(f"LPN Number of parameters: {num_params:,}")

    # Define RNG keys needed by make_rng inside the model for different purposes
    apply_rngs = {"dropout": key_dropout, "latents": jax.random.fold_in(key_dropout,1), 
                  "latents_init":jax.random.fold_in(key_dropout,2), 
                  "random_search":jax.random.fold_in(key_dropout,3), 
                  "gradient_ascent_random_perturbation":jax.random.fold_in(key_dropout,4)}


    print("\nTesting __call__ with use_cross_attention=True, mode='mean'")
    loss_ca_mean, _ = lpn_test_model.apply(
        variables_lpn, test_pairs, test_grid_shapes,
        dropout_eval=False, mode="mean", use_cross_attention=True,
        rngs=apply_rngs, 
        prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
    )
    print(f"CA Mean Loss: {loss_ca_mean}")

    print("\nTesting generate_output with use_cross_attention=True, mode='first'")
    gen_input_grid = test_pairs[:, 0, ..., 0] 
    gen_input_grid_shape = test_grid_shapes[:, 0, 0, :] 

    gen_out_grids, gen_out_shapes, _ = lpn_test_model.apply(
        variables_lpn,
        method=lpn_test_model.generate_output, 
        pairs=test_pairs, 
        grid_shapes=test_grid_shapes, 
        input=gen_input_grid, 
        input_grid_shape=gen_input_grid_shape, 
        key=key_gen_main, # Master key for generate_output
        dropout_eval=True, 
        mode="first", 
        use_cross_attention=True,
        return_two_best=False,
        rngs=apply_rngs 
    )
    print(f"Generated output grids shape (CA, first): {gen_out_grids.shape}")
    print(f"Generated output shapes shape (CA, first): {gen_out_shapes.shape}")

    print("\n--- Original __main__ tests (for reference, use_cross_attention=False) ---")
    print("Original Mean Loss (CA=False):")
    loss_orig_mean, _ = jax.jit(lpn_test_model.apply, static_argnames=["dropout_eval", "mode", "use_cross_attention"])(
        variables_lpn, test_pairs, test_grid_shapes,
        dropout_eval=False, mode="mean", use_cross_attention=False, 
        rngs=apply_rngs,
        prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4,
    )
    print(f"Original Mean Loss: {loss_orig_mean}")
    print("--- End of __main__ example ---")
