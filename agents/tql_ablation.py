import copy
from typing import Any, Sequence, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    LogParam,
    SignedExpParam,
)

def l2_normalize(x, eps):
    return x / (jnp.linalg.norm(x, ord=2) + eps)


def _spectral_norm(kernel_2d, power_iters, eps):
    u = jnp.ones((kernel_2d.shape[1],), dtype=kernel_2d.dtype)
    u = l2_normalize(u, eps)

    def body(_, u):
        v = jax.lax.stop_gradient(kernel_2d @ u)
        v = l2_normalize(v, eps)
        u = jax.lax.stop_gradient(kernel_2d.T @ v)
        u = l2_normalize(u, eps)
        return u

    u = jax.lax.fori_loop(0, power_iters, body, u)
    v = jax.lax.stop_gradient(kernel_2d @ u)
    v = l2_normalize(v, eps)
    sigma = jnp.einsum('i,ij,j->', v, kernel_2d, u)
    return sigma


class SigmaDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    sigma_init: float = 1.0
    power_iters: int = 5
    eps: float = 1e-6
    precision: Any = None

    @nn.compact
    def __call__(self, inputs):
        if self.power_iters < 1:
            raise ValueError('power_iters must be >= 1')

        kernel = self.param(
            'kernel',
            self.kernel_init,
            (inputs.shape[-1], self.features),
            self.param_dtype,
        )
        sigma_param = self.param(
            'sigma',
            init_fn=lambda _key: jnp.full((1,), self.sigma_init, dtype=self.param_dtype),
        )
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None

        kernel_f32 = kernel.astype(jnp.float32)
        sigma = _spectral_norm(kernel_f32, self.power_iters, self.eps)
        sigma = jnp.maximum(sigma, self.eps)
        kernel_hat = (sigma_param.astype(jnp.float32) / sigma) * kernel_f32

        inputs = inputs.astype(self.dtype)
        kernel_hat = kernel_hat.astype(self.dtype)
        y = jax.lax.dot_general(
            inputs,
            kernel_hat,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y = y + jnp.reshape(bias.astype(self.dtype), (1,) * (y.ndim - 1) + (-1,))
        return y


class SigmaDenseGeneral(nn.Module):
    features: Union[int, Sequence[int]]
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    sigma_init: float = 1.0
    power_iters: int = 5
    eps: float = 1e-6
    precision: Any = None

    @nn.compact
    def __call__(self, inputs):
        if self.power_iters < 1:
            raise ValueError('power_iters must be >= 1')

        features = (self.features,) if isinstance(self.features, int) else tuple(self.features)
        in_features = inputs.shape[-1]

        kernel = self.param(
            'kernel',
            self.kernel_init,
            (in_features, *features),
            self.param_dtype,
        )
        sigma_param = self.param(
            'sigma',
            init_fn=lambda _key: jnp.full((1,), self.sigma_init, dtype=self.param_dtype),
        )
        if self.use_bias:
            bias = self.param('bias', self.bias_init, features, self.param_dtype)
        else:
            bias = None

        kernel_f32 = kernel.astype(jnp.float32)
        kernel_2d = jnp.reshape(kernel_f32, (in_features, -1))
        sigma = _spectral_norm(kernel_2d, self.power_iters, self.eps)
        sigma = jnp.maximum(sigma, self.eps)
        kernel_hat = (sigma_param.astype(jnp.float32) / sigma) * kernel_f32

        inputs = inputs.astype(self.dtype)
        kernel_hat = kernel_hat.astype(self.dtype)
        y = jax.lax.dot_general(
            inputs,
            kernel_hat,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y = y + jnp.reshape(bias.astype(self.dtype), (1,) * (y.ndim - bias.ndim) + bias.shape)
        return y

def _dense_layers(use_spectral_norm: bool):
    if use_spectral_norm:
        return SigmaDense, SigmaDenseGeneral
    return nn.Dense, nn.DenseGeneral

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward layers."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    ffn_type: str = 'gelu'
    gated_mlp_ratio: float = 8 / 3
    dropout_rate: float = 0.1
    use_spectral_norm: bool = True
    use_rms_norm: bool = False
    rms_norm_eps: float = 1e-6
    use_post_norm: bool = False
    use_qk_norm: bool = False
    attn_logits_softcap: float = 0.0
    attn_dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False, return_attention_weights: bool = False):
        x = x.astype(self.dtype)
        # Pre-LayerNorm architecture (more stable)
        # Multi-head self-attention with pre-norm
        # LayerNorm always in fp32 for numerical stability
        if self.use_rms_norm:
            attn_input = nn.RMSNorm(
                epsilon=self.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                use_scale=True,
                reduction_axes=-1,
                feature_axes=-1,
            )(x)
        else:
            attn_input = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f'hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).'
            )
        head_dim = self.hidden_dim // self.num_heads

        dense_cls, dense_general_cls = _dense_layers(self.use_spectral_norm)

        query = dense_general_cls(
            features=(self.num_heads, head_dim),
            name='query',
            dtype=self.dtype,
        )(attn_input)
        key = dense_general_cls(
            features=(self.num_heads, head_dim),
            name='key',
            dtype=self.dtype,
        )(attn_input)
        value = dense_general_cls(
            features=(self.num_heads, head_dim),
            name='value',
            dtype=self.dtype,
        )(attn_input)
        if self.use_qk_norm:
            query = nn.RMSNorm(
                name='q_norm',
                epsilon=self.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                use_scale=True,
                reduction_axes=-1,
                feature_axes=-1,
            )(query)
            key = nn.RMSNorm(
                name='k_norm',
                epsilon=self.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                use_scale=True,
                reduction_axes=-1,
                feature_axes=-1,
            )(key)

        scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
        # For bf16: compute QK^T matmul in fp32 for stability (σReparam), then keep in fp32 for softmax
        # For fp32: no casting needed, everything stays fp32
        if self.dtype == jnp.bfloat16:
            query_f32 = query.astype(jnp.float32)
            key_f32 = key.astype(jnp.float32)
            logits = jnp.einsum('...qhd,...khd->...hqk', query_f32, key_f32) / scale
        else:
            logits = jnp.einsum('...qhd,...khd->...hqk', query, key) / scale
        if self.attn_logits_softcap > 0.0:
            sc = jnp.asarray(self.attn_logits_softcap, dtype=jnp.float32)
            logits = sc * jnp.tanh(logits.astype(jnp.float32) / sc)
        attention_weights = nn.softmax(logits.astype(jnp.float32), axis=-1)
        if self.attn_dropout_rate > 0.0:
            attention_weights = nn.Dropout(
                rate=self.attn_dropout_rate,
                deterministic=not training,
            )(attention_weights)
        attention_weights = attention_weights.astype(self.dtype)

        attn_output = jnp.einsum('...hqk,...khd->...qhd', attention_weights, value)
        attn_output = attn_output.reshape(*attn_input.shape[:-1], self.hidden_dim)
        attn_output = dense_cls(self.hidden_dim, name='out', dtype=self.dtype)(attn_output)
        attn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = x + attn_out
        if self.use_post_norm:
            if self.use_rms_norm:
                x = nn.RMSNorm(
                    name='post_attn_norm',
                    epsilon=self.rms_norm_eps,
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    use_scale=True,
                    reduction_axes=-1,
                    feature_axes=-1,
                )(x)
            else:
                x = nn.LayerNorm(name='post_attn_norm', dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)

        # Feed-forward network with pre-norm
        # LayerNorm always in fp32 for numerical stability
        if self.use_rms_norm:
            mlp_input = nn.RMSNorm(
                epsilon=self.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                use_scale=True,
                reduction_axes=-1,
                feature_axes=-1,
            )(x)
        else:
            mlp_input = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
        if self.ffn_type == 'gelu':
            mlp_dim = self.hidden_dim * self.mlp_ratio
            mlp_out = dense_cls(mlp_dim, dtype=self.dtype)(mlp_input)
            mlp_out = nn.gelu(mlp_out)
        else:
            mlp_dim = int(self.hidden_dim * self.gated_mlp_ratio)
            mlp_dim = (mlp_dim + 7) // 8 * 8
            gate = dense_cls(mlp_dim, dtype=self.dtype)(mlp_input)
            up = dense_cls(mlp_dim, dtype=self.dtype)(mlp_input)
            if self.ffn_type == 'swiglu':
                mlp_out = jax.nn.silu(gate) * up
            elif self.ffn_type == 'geglu':
                mlp_out = nn.gelu(gate) * up
            else:
                raise ValueError(
                    f"Unknown ffn_type={self.ffn_type!r}. Expected 'gelu'|'swiglu'|'geglu'."
                )
        mlp_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_out)
        mlp_out = dense_cls(self.hidden_dim, dtype=self.dtype)(mlp_out)
        mlp_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_out)

        x = x + mlp_out
        if self.use_post_norm:
            if self.use_rms_norm:
                x = nn.RMSNorm(
                    name='post_mlp_norm',
                    epsilon=self.rms_norm_eps,
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    use_scale=True,
                    reduction_axes=-1,
                    feature_axes=-1,
                )(x)
            else:
                x = nn.LayerNorm(name='post_mlp_norm', dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
        if return_attention_weights:
            return x, attention_weights.astype(jnp.float32)
        return x


class TransformerValue(nn.Module):
    """Transformer-based value function with continuous state and action inputs."""
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    ffn_type: str = 'gelu'
    gated_mlp_ratio: float = 8 / 3
    dropout_rate: float = 0.1
    num_ensembles: int = 2
    encoder: Any = None
    use_state_action_separation: bool = True  # Whether to treat state/action as separate token sequences
    use_modality_embeddings: bool = False  # Whether to use learnable modality embeddings for state/action tokens
    use_spectral_norm: bool = True
    use_rms_norm: bool = False
    rms_norm_eps: float = 1e-6
    use_post_norm: bool = False
    use_qk_norm: bool = False
    attn_logits_softcap: float = 0.0
    attn_dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

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

        observations = observations.astype(self.dtype)
        actions = actions.astype(self.dtype)

        dense_cls, _ = _dense_layers(self.use_spectral_norm)

        batch_shape = observations.shape[:-1]
        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        # Reshape to treat each dimension as a token: (..., obs_dim, 1)
        obs_expanded = observations[..., None]  # (..., obs_dim, 1)
        action_expanded = actions[..., None]    # (..., action_dim, 1)

        # Project each dimension to hidden_dim
        obs_tokens = dense_cls(self.hidden_dim, name='obs_projection', dtype=self.dtype)(obs_expanded)
        action_tokens = dense_cls(self.hidden_dim, name='action_projection', dtype=self.dtype)(action_expanded)

        # Optionally add learnable modality embeddings (fp32 for stability)
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
            obs_tokens = obs_tokens + state_type_embed_broadcast.astype(self.dtype)
            action_tokens = action_tokens + action_type_embed_broadcast.astype(self.dtype)

        # Concatenate along sequence dimension
        tokens = jnp.concatenate([obs_tokens, action_tokens], axis=-2)
        # Shape: (..., obs_dim + action_dim, hidden_dim)
        seq_len = obs_dim + action_dim

        # Add learnable positional embeddings (fp32 for stability)
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
        tokens = tokens + pos_embedding_broadcast.astype(self.dtype)

        # Optional: Add learnable CLS token for aggregation (kept in fp32 for stability)
        # (alternative to mean pooling)
        cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, self.hidden_dim)
        )
        cls_shape = (1,) * len(batch_shape) + cls_token.shape
        cls_token_broadcast = jnp.reshape(cls_token, cls_shape)
        cls_token_broadcast = jnp.tile(cls_token_broadcast, batch_shape + (1, 1))
        tokens = jnp.concatenate([cls_token_broadcast.astype(self.dtype), tokens], axis=-2)
        # Shape: (..., seq_len + 1, hidden_dim)

        # Apply transformer blocks
        layer_attention_weights = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                ffn_type=self.ffn_type,
                gated_mlp_ratio=self.gated_mlp_ratio,
                dropout_rate=self.dropout_rate,
                use_spectral_norm=self.use_spectral_norm,
                use_rms_norm=self.use_rms_norm,
                rms_norm_eps=self.rms_norm_eps,
                use_post_norm=self.use_post_norm,
                use_qk_norm=self.use_qk_norm,
                attn_logits_softcap=self.attn_logits_softcap,
                attn_dropout_rate=self.attn_dropout_rate,
                dtype=self.dtype,
                name=f'transformer_block_{i}'
            )
            if return_attention_weights:
                tokens, attn_weights = block(tokens, training=training, return_attention_weights=True)
                layer_attention_weights.append(attn_weights)
            else:
                tokens = block(tokens, training=training)

        # Final layer norm (always in fp32 for stability)
        if self.use_rms_norm:
            tokens = nn.RMSNorm(
                name='final_ln',
                epsilon=self.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                use_scale=True,
                reduction_axes=-1,
                feature_axes=-1,
            )(tokens)
        else:
            tokens = nn.LayerNorm(name='final_ln', dtype=jnp.float32)(tokens.astype(jnp.float32)).astype(self.dtype)

        # Extract CLS token for prediction (first token)
        cls_output = tokens[..., 0, :]

        # Alternative: Global average pooling (uncomment if not using CLS token)
        # pooled = jnp.mean(tokens, axis=-2)

        # Output heads for ensemble
        q_values = []
        for i in range(self.num_ensembles):
            # Add a small MLP head for each ensemble member
            q = dense_cls(self.hidden_dim // 2, name=f'q_head_{i}_hidden', dtype=self.dtype)(cls_output)
            q = nn.relu(q)
            q = dense_cls(1, name=f'q_head_{i}_output', dtype=self.dtype)(q)
            q_values.append(q)

        q_values = jnp.concatenate(q_values, axis=-1)
        # Cast output to fp32 for loss computation
        q_values = q_values.astype(jnp.float32)
        # Shape: (..., num_ensembles)
        if return_attention_weights:
            stacked_weights = (
                jnp.stack(layer_attention_weights, axis=0)
                if layer_attention_weights
                else None
            )
            return q_values, stacked_weights
        return q_values




class TQLAblationAgent(flax.struct.PyTreeNode):
    """FQL agent with transformer critic and flow matching actors."""
    
    rng: Any
    network: Any
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
        
        use_attention_entropy_loss = bool(self.config['use_attention_entropy_loss'])
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
       
       
        critic_loss = q_loss + attention_entropy_loss + temperature_loss
        
        info = {
            'critic_loss': critic_loss,
            'q_loss': q_loss,
            'attention_entropy_loss': attention_entropy_loss,
            'temperature_loss': temperature_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
        }
        if use_attention_entropy_loss:
            for i in range(int(temperature.shape[0])):
                info[f'attention_entropy_coef_layer_{i}_cls'] = temperature[i, 0]
                info[f'attention_entropy_coef_layer_{i}_other'] = temperature[i, 1]
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
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    @jax.jit
    def _sample_actions_batch(self, observations, action_seed):
        """Internal method for batched observations."""

        if 'implicit' in self.config["extraction"]:
            orig_observations = observations
            if self.config['encoder'] is not None:
                observations = self.network.select('actor_flow_encoder')(observations)
            action_seed, noise_seed = jax.random.split(action_seed)

            # Sample `num_samples` noises and propagate them through the flow.
            actions = jax.random.normal(
                action_seed,
                (
                    *observations.shape[:-1],
                    self.config['num_samples'],
                    self.config['action_dim'],
                ),
            )
            n_observations = jnp.repeat(jnp.expand_dims(observations, 1), self.config['num_samples'], axis=1)
            n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 1), self.config['num_samples'], axis=1)
            for i in range(self.config['flow_steps']):
                t = jnp.full((*observations.shape[:-1], self.config['num_samples'], 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(n_observations, actions, t, is_encoded=True)
                actions = actions + vels / self.config['flow_steps']
            actions = jnp.clip(actions, -1, 1)

            # Pick the action with the highest Q-value.
            q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=-1)
            actions = actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            noises = jax.random.normal(
                action_seed,
                (*observations.shape[:-1], self.config['action_dim']),
            )
            actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)
        return actions
    
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the one-step policy."""
        # Handle seed
        action_seed = seed if seed is not None else jax.random.PRNGKey(0)
        
        # Check if we need to add batch dimension (this check happens at trace time)
        if observations.ndim == 1:
            observations = observations[None, :]
            actions = self._sample_actions_batch(observations, action_seed)
            return actions[0]
        
        return self._sample_actions_batch(observations, action_seed)
    
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
        if config['use_attention_entropy_loss']:
            target_entropy = config['attention_entropy_target']
            if target_entropy is None:
                raise ValueError(
                    "attention_entropy_target cannot be None; provide a (num_layers, 2) target."
                )
            ok = (
                isinstance(target_entropy, (list, tuple))
                and len(target_entropy) == config['num_layers']
                and all(isinstance(te, (list, tuple)) and len(te) == 2 for te in target_entropy)
            )
            if not ok:
                raise ValueError(
                    "attention_entropy_target must be a (num_layers, 2) nested list/tuple "
                    "[(cls, other) per layer]. Example for num_layers=2: ((3.0, 2.5), (1.0, 0.5))."
                )

        # Determine dtype for activations (bf16 or fp32)
        use_bf16 = config.get('use_bf16', False)
        dtype = jnp.bfloat16 if use_bf16 else jnp.float32

        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module(dtype=dtype)
            encoders['actor_bc_flow'] = encoder_module(dtype=dtype)
            encoders['actor_onestep_flow'] = encoder_module(dtype=dtype)
        else:
            encoders['critic'] = None
            encoders['actor_bc_flow'] = None
            encoders['actor_onestep_flow'] = None

        # Define critic with transformer
        critic_def = TransformerValue(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            ffn_type=config['ffn_type'],
            gated_mlp_ratio=config['gated_mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            num_ensembles=config['num_ensembles'],
            encoder=encoders['critic'],
            use_modality_embeddings=config['use_modality_embeddings'],
            use_spectral_norm=config['use_spectral_norm'],
            use_rms_norm=config['use_rms_norm'],
            rms_norm_eps=config['rms_norm_eps'],
            use_post_norm=config['use_post_norm'],
            use_qk_norm=config['use_qk_norm'],
            attn_logits_softcap=config['attn_logits_softcap'],
            attn_dropout_rate=config['attn_dropout_rate'],
            dtype=dtype,
        )

        # Define actors with original flow matching networks
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders['actor_bc_flow'],
            dtype=dtype,
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders['actor_onestep_flow'],
            dtype=dtype,
        )
        
        # Define learnable attention entropy temperature.
        # temp_type options:
        # - 'exp'   : alpha = exp(log_value)  (always positive, exp-like)
        # - 'sign'  : alpha = sign(raw)*(exp(|raw|)-1) (signed exp-like)
        attention_temp_init = config['attention_entropy_temperature_init']
        temp_shape = (config['num_layers'], 2)
        if config['temp_type'] == 'exp':
            attention_temp_def = LogParam(init_value=attention_temp_init, shape=temp_shape)
        elif config['temp_type'] == 'sign':
            attention_temp_def = SignedExpParam(init_value=attention_temp_init, shape=temp_shape)
        else:
            raise ValueError("temp_type must be one of: 'exp', 'linear', 'sign', 'sinh'")

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
            attention_entropy_temperature=(attention_temp_def, ()),
        )
        if encoders['actor_bc_flow'] is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable
            network_info['actor_bc_flow_encoder'] = (encoders['actor_bc_flow'], (ex_observations,))
        
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
        critic_lr = config['critic_lr']
        actor_lr = config['actor_lr']

        # Create separate optimizers for critic and actor
        def make_single_optimizer(lr):
            return make_optimizer(
                lr,
                config['critic_grad_clip'] if config['critic_grad_clip'] > 0 else None,
                optimizer_type=config['optimizer'],
                weight_decay=config['adamw_weight_decay'],
                warmup_steps=config['warmup_steps'],
                total_steps=config['train_steps'],
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
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='tql_ablation',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            # Training hyperparameters
            critic_lr=1e-4,  # Learning rate for critic
            actor_lr=5e-4,  # Learning rate for actor
            critic_grad_clip=0.0, 
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
            adamw_weight_decay=0.01,  # Weight decay for AdamW (applied to all params except LayerNorm, embeddings, bias)
            warmup_steps=10000,  # Number of warmup steps for all optimizers (0 = no warmup)
            # Mixed precision (bf16) - activations in bf16, weights/optimizer in fp32
            use_bf16=False,  # Enable bfloat16 mixed precision training
            # Attention entropy temperature and target (learned automatically like SAC)
            use_attention_entropy_loss=True,
            temp_type='exp',  # 'exp' | 'sign'
            attention_entropy_temperature_init=1.0,  # Initial temperature value (exp(0) = 1.0, but can be set to any positive value)
            attention_entropy_temperature_min=-100.0,  # Minimum temperature value
            attention_entropy_temperature_max=100.0,  # Maximum temperature value
            # Target entropy for attention. Required shape: (num_layers, 2) == (layer, {cls, other}).
            attention_entropy_target=((0.0, 0.0), (0.0, 0.0)),  # default matches num_layers=2
            # Actor network (original MLP)
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            # Critic transformer architecture
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            mlp_ratio=4,
            ffn_type='gelu',
            gated_mlp_ratio=8 / 3,
            dropout_rate=0.0,
            num_ensembles=2,
            use_modality_embeddings=True,  # Whether to use learnable modality embeddings for state/action tokens
            use_spectral_norm=False,
            use_rms_norm=False,
            rms_norm_eps=1e-6,
            use_post_norm=False,
            use_qk_norm=False,
            attn_logits_softcap=0.0,
            attn_dropout_rate=0.0,

            # Encoder
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name
        )
    )
    return config
