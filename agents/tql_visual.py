import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
import math

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, LogParam, Param, SignedExpParam, SinhParam


def _policy_noise_batch_shape(observations):
    """Batch prefix for policy noise (must match batch of encoded features, not H/W).

    Pixel batches are ``(..., H, W, C)`` (ndim >= 4); noise shape prefix is ``shape[:-3]``.
    Vector observations are ``(..., d)`` with ndim < 4; prefix is ``shape[:-1]``.
    """
    if observations.ndim >= 4:
        return observations.shape[:-3]
    return observations.shape[:-1]


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward layers."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False, return_attention_weights: bool = False):
        # Pre-LayerNorm architecture (more stable)
        # Multi-head self-attention with pre-norm
        attn_input = nn.LayerNorm()(x)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f'hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).'
            )
        head_dim = self.hidden_dim // self.num_heads

        query = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name='query'
        )(attn_input)
        key = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name='key'
        )(attn_input)
        value = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name='value'
        )(attn_input)

        scale = jnp.sqrt(head_dim).astype(attn_input.dtype)
        logits = jnp.einsum('...qhd,...khd->...hqk', query, key) / scale
        attention_weights = nn.softmax(logits, axis=-1)

        attn_output = jnp.einsum('...hqk,...khd->...qhd', attention_weights, value)
        attn_output = attn_output.reshape(*attn_input.shape[:-1], self.hidden_dim)
        attn_output = nn.Dense(self.hidden_dim, name='out')(attn_output)
        attn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = x + attn_out
        
        # Feed-forward network with pre-norm
        mlp_input = nn.LayerNorm()(x)
        mlp_dim = self.hidden_dim * self.mlp_ratio
        mlp_out = nn.Dense(mlp_dim)(mlp_input)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_out)
        mlp_out = nn.Dense(self.hidden_dim)(mlp_out)
        mlp_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_out)
        
        x = x + mlp_out
        if return_attention_weights:
            return x, attention_weights
        return x


class TransformerValue(nn.Module):
    """Transformer-based value function with continuous state and action inputs."""
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    num_ensembles: int = 2
    encoder: Any = None
    use_state_action_separation: bool = True  # Whether to treat state/action as separate token sequences
    use_modality_embeddings: bool = False  # Whether to use learnable modality embeddings for state/action tokens
    
    @nn.compact
    def __call__(self, observations, actions, training: bool = False, return_attention_weights: bool = False):
        """
        Args:
            observations: shape (..., obs_dim)
            actions: shape (..., action_dim)
            
        Returns:
            Q-values: shape (..., num_ensembles)
        """
        # Apply encoder if provided (for visual inputs)
        if self.encoder is not None:
            observations = self.encoder(observations)
        
        batch_shape = observations.shape[:-1]
        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]
        
        if self.use_state_action_separation:
            # Treat each dimension as a separate token
            # This allows the transformer to attend between different state/action dimensions

            # Reshape to treat each dimension as a token: (..., obs_dim, 1)
            obs_expanded = observations[..., None]  # (..., obs_dim, 1)
            action_expanded = actions[..., None]    # (..., action_dim, 1)

            # Project each dimension to hidden_dim
            obs_tokens = nn.Dense(self.hidden_dim, name='obs_projection')(obs_expanded)
            action_tokens = nn.Dense(self.hidden_dim, name='action_projection')(action_expanded)
            
            # Optionally add learnable modality embeddings
            if self.use_modality_embeddings:
                # Create learnable modality embeddings
                state_type_embed = self.param(
                    'state_type_embed',
                    nn.initializers.normal(stddev=0.02),
                    (1, self.hidden_dim)
                )
                action_type_embed = self.param(
                    'action_type_embed',
                    nn.initializers.normal(stddev=0.02),
                    (1, self.hidden_dim)
                )
                
                # Broadcast modality embeddings to match batch dimensions
                state_shape = (1,) * len(batch_shape) + state_type_embed.shape
                state_type_embed_broadcast = jnp.reshape(state_type_embed, state_shape)
                state_type_embed_broadcast = jnp.tile(state_type_embed_broadcast, batch_shape + (obs_dim, 1))
                
                action_shape = (1,) * len(batch_shape) + action_type_embed.shape
                action_type_embed_broadcast = jnp.reshape(action_type_embed, action_shape)
                action_type_embed_broadcast = jnp.tile(action_type_embed_broadcast, batch_shape + (action_dim, 1))
                
                # Add modality embeddings
                obs_tokens = obs_tokens + state_type_embed_broadcast
                action_tokens = action_tokens + action_type_embed_broadcast
            
            # Concatenate along sequence dimension
            tokens = jnp.concatenate([obs_tokens, action_tokens], axis=-2)
            # Shape: (..., obs_dim + action_dim, hidden_dim)
            
            seq_len = obs_dim + action_dim
            
        else:
            # Simpler approach: concatenate observations and actions, then project
            combined = jnp.concatenate([observations, actions], axis=-1)
            # Shape: (..., obs_dim + action_dim)
            
            # Project to hidden_dim and add sequence dimension
            tokens = nn.Dense(self.hidden_dim, name='input_projection')(combined)
            tokens = tokens[..., None, :]  # Add sequence dimension
            # Shape: (..., 1, hidden_dim)
            
            seq_len = 1
        
        # Add learnable positional embeddings
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (seq_len, self.hidden_dim)
        )
        
        # Broadcast positional embeddings to match batch dimensions
        # tokens shape: (..., seq_len, hidden_dim)
        # pos_embedding shape: (seq_len, hidden_dim)
        # We need to broadcast pos_embedding to (..., seq_len, hidden_dim)
        pos_shape = (1,) * len(batch_shape) + pos_embedding.shape
        pos_embedding_broadcast = jnp.reshape(pos_embedding, pos_shape)
        tokens = tokens + pos_embedding_broadcast
        
        # Optional: Add learnable CLS token for aggregation
        # (alternative to mean pooling)
        cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, self.hidden_dim)
        )
        cls_shape = (1,) * len(batch_shape) + cls_token.shape
        cls_token_broadcast = jnp.reshape(cls_token, cls_shape)
        cls_token_broadcast = jnp.tile(cls_token_broadcast, batch_shape + (1, 1))
        tokens = jnp.concatenate([cls_token_broadcast, tokens], axis=-2)
        # Shape: (..., seq_len + 1, hidden_dim)
        
        # Apply transformer blocks
        layer_attention_weights = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}'
            )
            if return_attention_weights:
                tokens, attn_weights = block(tokens, training=training, return_attention_weights=True)
                layer_attention_weights.append(attn_weights)
            else:
                tokens = block(tokens, training=training)
        
        # Final layer norm (standard in transformers)
        tokens = nn.LayerNorm(name='final_ln')(tokens)
        
        # Extract CLS token for prediction (first token)
        cls_output = tokens[..., 0, :]
        
        # Alternative: Global average pooling (uncomment if not using CLS token)
        # pooled = jnp.mean(tokens, axis=-2)
        
        # Output heads for ensemble
        q_values = []
        for i in range(self.num_ensembles):
            # Add a small MLP head for each ensemble member
            q = nn.Dense(self.hidden_dim // 2, name=f'q_head_{i}_hidden')(cls_output)
            q = nn.relu(q)
            q = nn.Dense(1, name=f'q_head_{i}_output')(q)
            q_values.append(q)
        
        q_values = jnp.concatenate(q_values, axis=-1)
        # Shape: (..., num_ensembles)
        if return_attention_weights:
            stacked_weights = (
                jnp.stack(layer_attention_weights, axis=0)
                if layer_attention_weights
                else None
            )
            return q_values, stacked_weights
        return q_values




class TQLTempMultiL2Agent(flax.struct.PyTreeNode):
    """FQL agent with transformer critic and flow matching actors."""
    
    rng: Any
    network: Any
    init_critic_params: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng, step=None):
        """Compute the FQL critic loss (like SAC, includes temperature loss)."""
        rng, sample_rng, dropout_rng = jax.random.split(rng, 3)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        
        next_qs = self.network.select('target_critic')(
            batch['next_observations'], 
            actions=next_actions,
            training=False
        )
        # next_qs shape: (batch_size, num_ensembles)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=-1)  # (batch_size,)
        else:
            next_q = next_qs.mean(axis=-1)  # (batch_size,)
        
        
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        
        use_attention_entropy_loss = bool(self.config.get('use_attention_entropy_loss', True))
        q, attn_weights = self.network.select('critic')(
            batch['observations'],
            actions=batch['actions'],
            params=grad_params,
            training=True,
            rngs={'dropout': dropout_rng},
            return_attention_weights=True,
        )

         # Q loss
        q_loss = jnp.square(q - target_q[:, None]).mean()

        # Attention entropy regularization and temperature learning (like SAC's alpha)
        # attn_weights shape: (num_layers, batch, num_heads, q, k)
        attn_entropy = -jnp.sum(
            attn_weights * jnp.log(jnp.clip(attn_weights, a_min=1e-9)),
            axis=-1,
        )  # (num_layers, batch, num_heads, q)

        # Compute 2D mean entropy per layer: (num_layers, 2)
        attn_entropy_cls_mean = attn_entropy[..., 0].mean(axis=(1, 2))  # (num_layers,)
        attn_entropy_other_mean = attn_entropy[..., 1:].mean(axis=(1, 2, 3))  # (num_layers,)
        attn_entropy_layer_mean = jnp.stack(
            [attn_entropy_cls_mean, attn_entropy_other_mean],
            axis=-1,
        )  # (num_layers, 2)

        if use_attention_entropy_loss:
            # Learn a separate temperature per transformer layer and token group (CLS vs other).
            # temperature shape: (num_layers, 2)
            temperature = self.network.select('attention_entropy_temperature')(params=grad_params)
            # Clamp temperature to a valid range
            temp_min = self.config['attention_entropy_temperature_min']
            temp_max = self.config['attention_entropy_temperature_max']
            temperature_clipped = jnp.clip(temperature, temp_min, temp_max)
            temperature_no_grad = jax.lax.stop_gradient(temperature_clipped)

            attention_entropy_loss = (-temperature_no_grad * attn_entropy_layer_mean).mean()
            attn_entropy_layer_mean_no_grad = jax.lax.stop_gradient(attn_entropy_layer_mean)

            target_entropy = self.config['attention_entropy_target']
            # Required shape: (num_layers, 2) == (layer, {cls, other})
            target_entropy = jnp.asarray(target_entropy, dtype=attn_entropy_layer_mean_no_grad.dtype)

            temperature_loss = (temperature * (attn_entropy_layer_mean_no_grad - target_entropy)).mean()

        else:
            attention_entropy_loss = jnp.asarray(0.0, dtype=q.dtype)
            temperature_loss = jnp.asarray(0.0, dtype=q.dtype)
       
        # Optional: L2 regularization towards critic initialization (stabilizes training).
        critic_reg_method = self.config['critic_reg_method']
        if grad_params is not None and critic_reg_method == 'l2_init' and self.config['critic_l2_init_coef'] > 0:
            critic_params = grad_params['modules_critic']
            init_params = self.init_critic_params
            # Exclude LayerNorm params from L2-init regularization (stability "glue").
            # This includes unnamed LayerNorms (e.g. 'LayerNorm_0') and the named final LN ('final_ln').
            items, _ = jax.tree_util.tree_flatten_with_path(critic_params)
            init_items, _ = jax.tree_util.tree_flatten_with_path(init_params)

            def _is_layernorm(path):
                for kk in path:
                    if isinstance(kk, jax.tree_util.DictKey) and isinstance(kk.key, str):
                        name = kk.key
                        if name.startswith('LayerNorm') or name == 'final_ln':
                            return True
                return False

            diffs = []
            denom = 0
            for (path, p), (_, p0) in zip(items, init_items, strict=True):
                if _is_layernorm(path):
                    continue
                diffs.append(jnp.sum(jnp.square(p - p0)))
                denom += int(p.size)
            l2_init_sum = (
                jnp.sum(jnp.stack(diffs))
                if diffs
                else jnp.asarray(0.0, dtype=jnp.float32)
            )
            l2_init_mean = (
                l2_init_sum / jnp.asarray(denom, dtype=l2_init_sum.dtype)
                if denom > 0
                else jnp.asarray(0.0, dtype=jnp.float32)
            )
            l2_init_loss = self.config['critic_l2_init_coef'] * l2_init_mean
        else:
            l2_init_sum = jnp.asarray(0.0, dtype=jnp.float32)
            l2_init_mean = jnp.asarray(0.0, dtype=jnp.float32)
            l2_init_loss = jnp.asarray(0.0, dtype=jnp.float32)

        critic_loss = q_loss + attention_entropy_loss + temperature_loss + l2_init_loss
        
        info = {
            'critic_loss': critic_loss,
            'q_loss': q_loss,
            'attention_entropy_loss': attention_entropy_loss,
            'temperature_loss': temperature_loss,
            'critic_l2_init_sum': l2_init_sum,
            # 'critic_l2_init_mean': l2_init_mean,
            'critic_l2_init_loss': l2_init_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
        }
        
        if use_attention_entropy_loss:
            # Report per-layer diagnostics as scalars (loggers often don't like arrays).
            for i in range(int(temperature.shape[0])):
                info[f'attention_entropy_coef_layer_{i}_cls'] = temperature[i, 0]
                info[f'attention_entropy_coef_layer_{i}_other'] = temperature[i, 1]
        # Always report attention entropy means if we have them (both enabled and disabled modes).
        for i in range(int(attn_entropy_layer_mean.shape[0])):
            info[f'attention_entropy_mean_layer_{i}_cls'] = attn_entropy_layer_mean[i, 0]
            info[f'attention_entropy_mean_layer_{i}_other'] = attn_entropy_layer_mean[i, 1]
        
        return critic_loss, info
    
    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        
        # BC flow loss
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        
        pred = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        if self.config["extraction"] == "implicit":
            return bc_flow_loss, {'bc_flow_loss': bc_flow_loss}
        
        # Distillation loss
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
        
        # Q loss
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'], 
            actions=actor_actions,
            training=False
        )
        # qs shape: (batch_size, num_ensembles)
        q = jnp.mean(qs, axis=-1)  # (batch_size,)
        
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
        
        # Total loss
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
        
        # Additional metrics for logging
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=None):
        """Compute the total loss (including temperature loss like SAC).
        
        Args:
            batch: Training batch.
            grad_params: Gradient parameters.
            rng: Random number generator.
            step: Current training step (optional, kept for compatibility but not used).
        """
        info = {}
        rng = rng if rng is not None else self.rng
        
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, step=step)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        loss = critic_loss + actor_loss
        return loss, info
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params
    
    @jax.jit
    def update(self, batch, step=None):
        """Update the agent and return a new agent with information dictionary.
        
        Args:
            batch: Training batch.
            step: Current training step (optional, kept for compatibility but not used).
        """
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)
        
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        critic_reg_method = self.config['critic_reg_method']
        rr_every = int(self.config['critic_random_replace_every'])
        rr_fraction = float(self.config['critic_random_replace_fraction'])
        rr_partial_reset = bool(self.config.get('random_replace_partial_reset', False))
        rr_drop_scope = str(self.config.get('random_replace_drop_scope', 'all_except_red'))
        if rr_drop_scope not in ('all_except_red', 'q_head_only'):
            raise ValueError(
                f"Unknown random_replace_drop_scope={rr_drop_scope!r}. "
                "Expected one of: 'all_except_red', 'q_head_only'."
            )
        if critic_reg_method == 'random_replace':
            step_i = step if step is not None else new_network.step
            step_i = jnp.asarray(step_i, dtype=jnp.int32)
            do_rr = (step_i % rr_every) == 0 
            frac = jnp.asarray(rr_fraction, dtype=jnp.float32)

            def _apply(cp):
                # Key generation lives inside the branch so it doesn't run on non-reset steps.
                rr_key = jax.random.fold_in(rng, step_i)

                # Random-replace mask over critic params, with a configurable scope.
                #
                # Scopes (random_replace_drop_scope):
                # - 'all_except_red' (default): drop everything except LayerNorm + embeddings + q_head_*
                # - 'q_head_only': drop only the final Q head MLPs (q_head_*_hidden and q_head_*_output)
                items, treedef = jax.tree_util.tree_flatten_with_path(cp)
                init_items, _ = jax.tree_util.tree_flatten_with_path(self.init_critic_params)
                keys = jax.random.split(rr_key, len(items))

                def _path_has_prefix(path, prefix: str):
                    for kk in path:
                        if isinstance(kk, jax.tree_util.DictKey) and isinstance(kk.key, str):
                            if kk.key.startswith(prefix):
                                return True
                    return False

                def _is_red_zone(path):
                    for kk in path:
                        if isinstance(kk, jax.tree_util.DictKey) and isinstance(kk.key, str):
                            name = kk.key
                            if name.startswith('LayerNorm') or name == 'final_ln':
                                return True
                            if name in ('pos_embedding', 'cls_token'):
                                return True
                    return False

                def _should_drop(path):
                    if rr_drop_scope == 'all_except_red':
                        # Drop everything except red-zone + q_head (kept stable).
                        if _is_red_zone(path):
                            return False
                        if _path_has_prefix(path, 'q_head_'):
                            return False
                        return True
                    if rr_drop_scope == 'q_head_only':
                        # Drop only the Q heads (last hidden + output per ensemble member).
                        return _path_has_prefix(path, 'q_head_')
                    return False

                new_leaves = []
                mask_rates = []
                for k, (path, p), (_, p0) in zip(keys, items, init_items, strict=True):
                    if not _should_drop(path):
                        new_leaves.append(p)
                        continue

                    # Elementwise mask over the remaining (non-protected) params.
                    m = jax.random.bernoulli(k, p=frac, shape=p.shape)
                    target = (0.5 * p0 + 0.5 * p) if rr_partial_reset else p0
                    new_leaves.append(jnp.where(m, target, p))
                    mask_rates.append(jnp.mean(m.astype(jnp.float32)))

                new_cp = jax.tree_util.tree_unflatten(treedef, new_leaves)
                
                mean_mask_rate = (
                    jnp.mean(jnp.stack(mask_rates))
                    if mask_rates
                    else jnp.asarray(0.0, dtype=jnp.float32)
                )
                return new_cp, mean_mask_rate

            (new_critic_params, mean_mask_rate) = jax.lax.cond(
                do_rr,
                _apply,
                lambda cp: (cp, jnp.asarray(0.0, dtype=jnp.float32)),
                new_network.params['modules_critic'],
            )
            new_network.params['modules_critic'] = new_critic_params
            info['critic/random_replace_applied'] = do_rr.astype(jnp.float32)
            info['critic/random_replace_fraction'] = frac
            info['critic/random_replace_mask_rate'] = mean_mask_rate
            info['critic/random_replace_partial_reset'] = jnp.asarray(
                1.0 if rr_partial_reset else 0.0, dtype=jnp.float32
            )
        else:
            info['critic/random_replace_applied'] = jnp.asarray(0.0, dtype=jnp.float32)
            info['critic/random_replace_partial_reset'] = jnp.asarray(0.0, dtype=jnp.float32)
        
        self.target_update(new_network, 'critic')
        
        return self.replace(network=new_network, rng=new_rng), info
        
    
    @jax.jit
    def _sample_actions_batch(self, observations, action_seed):
        """Internal method for batched observations."""

        if 'implicit' in self.config["extraction"]:
            orig_observations = observations
            if self.config['encoder'] is not None:
                observations = self.network.select('actor_bc_flow_encoder')(observations)
            action_seed, noise_seed = jax.random.split(action_seed)

            # Sample `num_samples` noises and propagate them through the flow.
            _bs = _policy_noise_batch_shape(observations)
            actions = jax.random.normal(
                action_seed,
                (
                    *_bs,
                    self.config['num_samples'],
                    self.config['action_dim'],
                ),
            )
            n_observations = jnp.repeat(jnp.expand_dims(observations, 1), self.config['num_samples'], axis=1)
            n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 1), self.config['num_samples'], axis=1)
            for i in range(self.config['flow_steps']):
                t = jnp.full((*_bs, self.config['num_samples'], 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(n_observations, actions, t, is_encoded=True)
                actions = actions + vels / self.config['flow_steps']
            actions = jnp.clip(actions, -1, 1)

            # Pick the action with the highest Q-value.
            q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=-1)
            actions = actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]

        
        else:
            noises = jax.random.normal(
                action_seed,
                (*_policy_noise_batch_shape(observations), self.config['action_dim']),
            )
            actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)
        return actions
    
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the one-step policy."""
        # Handle seed
        action_seed = seed if seed is not None else jax.random.PRNGKey(0)

        use_encoder = self.config.get('encoder') is not None
        squeeze_batch = False
        if observations.ndim == 1:
            observations = observations[None, :]
            squeeze_batch = True
        elif use_encoder and observations.ndim == 3:
            # Single image observation (H, W, C).
            observations = observations[None, ...]
            squeeze_batch = True

        actions = self._sample_actions_batch(observations, action_seed)
        if squeeze_batch:
            return actions[0]
        return actions
    
    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.
        
        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Validate target entropy specification only if the feature is enabled.
        # Required shape: (num_layers, 2) == (layer, {cls, other}).
        use_attention_entropy_loss = bool(config.get('use_attention_entropy_loss', True))
        num_layers = int(config['num_layers'])
        if use_attention_entropy_loss:
            target_entropy = config.get('attention_entropy_target', 0.0)
            if target_entropy is None:
                raise ValueError(
                    "attention_entropy_target cannot be None; provide a (num_layers, 2) target."
                )
            ok = (
                isinstance(target_entropy, (list, tuple))
                and len(target_entropy) == num_layers
                and all(isinstance(te, (list, tuple)) and len(te) == 2 for te in target_entropy)
            )
            if not ok:
                raise ValueError(
                    "attention_entropy_target must be a (num_layers, 2) nested list/tuple "
                    "[(cls, other) per layer]. Example for num_layers=2: ((3.0, 2.5), (1.0, 0.5))."
                )
        
        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
        
        # Define critic with transformer
        critic_def = TransformerValue(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            num_ensembles=config['num_ensembles'],
            encoder=encoders.get('critic'),
            use_state_action_separation=config.get('use_state_action_separation', True),
            use_modality_embeddings=config.get('use_modality_embeddings', False),
        )
        
        # Define actors with original flow matching networks
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )
        
        # Define learnable attention entropy temperature.
        # temp_type options:
        # - 'exp'   : alpha = exp(log_value)  (always positive, exp-like)
        # - 'linear': alpha = value          (can be negative)
        # - 'sign'  : alpha = sign(raw)*(exp(|raw|)-1) (signed exp-like)
        # - 'sinh'  : alpha = sinh(raw)      (signed exp-like tails)
        attention_temp_init = config.get('attention_entropy_temperature_init', 1.0)
        temp_shape = (config['num_layers'], 2)
        if config['temp_type'] == 'exp':
            attention_temp_def = LogParam(init_value=attention_temp_init, shape=temp_shape)
        elif config['temp_type'] == 'linear':
            attention_temp_def = Param(init_value=attention_temp_init, shape=temp_shape)
        elif config['temp_type'] == 'sign':
            attention_temp_def = SignedExpParam(init_value=attention_temp_init, shape=temp_shape)
        elif config['temp_type'] == 'sinh':
            attention_temp_def = SinhParam(init_value=attention_temp_init, shape=temp_shape)
        else:
            raise ValueError("temp_type must be one of: 'exp', 'linear', 'sign', 'sinh'")
        
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
            attention_entropy_temperature=(attention_temp_def, ()),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        
        network_def = ModuleDict(networks)

        # Initialize with dropout RNG for transformer critic
        init_rng_dict = {'params': init_rng, 'dropout': jax.random.PRNGKey(0)}
        network_params = network_def.init(init_rng_dict, **network_args)['params']




        def make_optimizer(learning_rate, grad_clip=None, optimizer_type='adam', 
                          weight_decay=0.0, warmup_steps=0, total_steps=None, params=None):
            optimizer_chain = []
            if grad_clip is not None:
                optimizer_chain.append(optax.clip_by_global_norm(grad_clip))
            
            # Warmup cosine decay schedule (applies to all optimizers)
            if warmup_steps > 0:
                decay_steps = total_steps - warmup_steps
                lr_schedule = optax.warmup_cosine_decay_schedule(
                    init_value=learning_rate * 0.01,
                    peak_value=learning_rate,
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps,
                    end_value=learning_rate * 0.1
                )
            else:
                lr_schedule = learning_rate
            
            if optimizer_type == 'muon':
                # Use Muon optimizer from local implementation
                optimizer_chain.append(optax.contrib.muon(learning_rate=lr_schedule))
            elif optimizer_type == 'adamw':
                # Weight decay mask (only apply to "main weights"):
                def should_apply_wd(path, value):
                    path_str = '/'.join(str(k.key if hasattr(k, 'key') else k) for k in path).lower()
                    
                    # Exclude input projections to preserve state/action signal
                    if 'projection' in path_str:
                        return False
                        
                    # Standard exclusions
                    if any(x in path_str for x in ['bias', 'layernorm', 'final_ln', 'pos_embedding', 'cls_token', 'state_type_embed', 'action_type_embed']):
                        return False
                    if value.ndim <= 1:
                        return False
                    
                    if 'modules_target_' in path_str:
                        return False
                    if 'modules_attention_entropy_temperature' in path_str:
                        return False
                        
                    return True

                mask = jax.tree_util.tree_map_with_path(should_apply_wd, params)

                optimizer_chain.append(
                    optax.adamw(
                        learning_rate=lr_schedule,
                        weight_decay=weight_decay,
                        mask=mask,
                    )
                )
            elif optimizer_type == 'adam':
                optimizer_chain.append(optax.adam(learning_rate=lr_schedule))
            else:
                raise ValueError(f"Optimizer type {optimizer_type} not supported")

            return optax.chain(*optimizer_chain)

        # Get learning rates (backward compatibility: use lr if critic_lr/actor_lr not present)
        critic_lr = config.get('critic_lr', config.get('lr', 3e-4))
        actor_lr = config.get('actor_lr', config.get('lr', 3e-4))
        
        # Create separate optimizers for critic and actor
        def make_single_optimizer(lr):
            return make_optimizer(
                lr,
                config['critic_grad_clip'] if config['critic_grad_clip'] > 0 else None,
                optimizer_type=config['optimizer'],
                weight_decay=config.get('adamw_weight_decay', 0.0),
                warmup_steps=config.get('warmup_steps', 0),
                total_steps=config.get('train_steps', 1000000),
                params=network_params
            )
        
        critic_optimizer = make_single_optimizer(critic_lr)
        actor_optimizer = make_single_optimizer(actor_lr)
        
        # Create parameter masks for multi_transform
        def partition_fn(path, value):
            path_str = '/'.join(str(k.key if hasattr(k, 'key') else k) for k in path)
            path_lower = path_str.lower()
            # Critic parameters
            if 'modules_critic' in path_lower or 'modules_attention_entropy_temperature' in path_lower:
                return 'critic'
            # Actor parameters (everything else that's not target)
            elif 'modules_actor' in path_lower:
                return 'actor'
            # Default to actor for any other parameters
            else:
                return 'actor'
        
        param_labels = jax.tree_util.tree_map_with_path(partition_fn, network_params)
        
        # Create multi-transform optimizer
        network_tx = optax.multi_transform(
            {'critic': critic_optimizer, 'actor': actor_optimizer},
            param_labels
        )
    


        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        # Copy to target networks
        params = network.params
        params['modules_target_critic'] = copy.deepcopy(params['modules_critic'])
        # Save critic initialization for optional L2-init regularization.
        init_critic_params = copy.deepcopy(params['modules_critic'])
        
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        
        return cls(
            rng,
            network=network,
            init_critic_params=init_critic_params,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='tql_temp_multi_l2',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            
            # Training hyperparameters
            critic_lr=3e-4,  # Learning rate for critic
            actor_lr=3e-4,  # Learning rate for actor
            critic_grad_clip=0.0, 
            
            # Critic regularization method selector (explicit; no auto mode):
            # - 'none'
            # - 'l2_init'
            # - 'random_replace' (periodic partial reset-to-init)
            critic_reg_method='none',
            critic_l2_init_coef=0.0,
            critic_random_replace_every=0,
            critic_random_replace_fraction=0.0,
            # If True and critic_reg_method=='random_replace', masked weights become
            # 0.5 * init + 0.5 * current (instead of a full reset to init).
            random_replace_partial_reset=False,
            # Which part of the critic to apply random-replace to:
            # - 'all_except_red': drop everything except LayerNorm + embeddings + q_head_*
            # - 'q_head_only'   : drop only the Q head MLPs (q_head_*_hidden and q_head_*_output)
            random_replace_drop_scope='all_except_red',
            batch_size=256,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=10.0,  # Distillation coefficient
            flow_steps=10,
            normalize_q_loss=False,
            extraction='fql',
            num_samples=16,
            
            # Optimizer settings
            optimizer='adamw',  # 'adam', 'adamw', or 'muon'
            adamw_weight_decay=0.001,  # Weight decay for AdamW (applied to all params except LayerNorm, embeddings, bias)
            warmup_steps=10000,  # Number of warmup steps for all optimizers (0 = no warmup)
            # Attention entropy temperature and target (learned automatically like SAC)
            use_attention_entropy_loss=True,
            temp_type='exp',  # 'exp' | 'linear' | 'sign' | 'sinh'
            attention_entropy_temperature_init=1.0,  # Initial temperature value (exp(0) = 1.0, but can be set to any positive value)
            attention_entropy_temperature_min=-100.0,  # Minimum temperature value
            attention_entropy_temperature_max=100.0,  # Maximum temperature value
            # Target entropy for attention. Required shape: (num_layers, 2) == (layer, {cls, other}).
            attention_entropy_target=((0.0, 0.0), (0.0, 0.0)),  # default matches num_layers=2
            
            # Actor network (original MLP)
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            
            # Critic transformer architecture
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            mlp_ratio=4,
            dropout_rate=0.0,
            num_ensembles=2,
            use_state_action_separation=True,  # Whether to treat each dimension as a token
            use_modality_embeddings=True,  # Whether to use learnable modality embeddings for state/action tokens
            
            # Encoder
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name
        )
    )
    return config