import copy
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, LogParam, SignedExpParam


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward layers."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False, return_attention_weights: bool = False):
        attn_input = nn.LayerNorm()(x)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f'hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).'
            )
        head_dim = self.hidden_dim // self.num_heads

        query = nn.DenseGeneral(features=(self.num_heads, head_dim), name='query')(attn_input)
        key   = nn.DenseGeneral(features=(self.num_heads, head_dim), name='key')(attn_input)
        value = nn.DenseGeneral(features=(self.num_heads, head_dim), name='value')(attn_input)

        scale = jnp.sqrt(head_dim).astype(attn_input.dtype)
        logits = jnp.einsum('...qhd,...khd->...hqk', query, key) / scale
        attention_weights = nn.softmax(logits, axis=-1)

        attn_output = jnp.einsum('...hqk,...khd->...qhd', attention_weights, value)
        attn_output = attn_output.reshape(*attn_input.shape[:-1], self.hidden_dim)
        attn_output = nn.Dense(self.hidden_dim, name='out')(attn_output)
        attn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = x + attn_out

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
    """Transformer-based value function with current (obs, act) and optional history tokens."""
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    num_ensembles: int = 2
    encoder: Any = None
    use_modality_embeddings: bool = False
    tokenization_mode: str = 'per_modality'  # 'per_modality' | 'per_dim' | 'linear_projected_per_dim'
    obs_token_dim: int = 32
    action_token_dim: int = 8
    num_action_tokens: int = 1  # how many tokens to use for the current action (split evenly across action_dim)

    @nn.compact
    def __call__(
        self,
        observations,
        actions,
        history_observations=None,
        training: bool = False,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            observations: (..., obs_dim)
            actions: (..., act_dim)
            history_observations: (..., hist_length, obs_dim) or None

        Returns:
            Q-values: (..., num_ensembles)
        """

        if self.encoder is not None:
            observations, history_observations = self.encoder(observations, history_observations)

        if observations.ndim >= 3:
            batch_shape = observations.shape[:-2]  # (B,) for patch tokens (B, P, D)
        else:
            batch_shape = observations.shape[:-1]
        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]
        action_chunk_size = actions.shape[-2]
        # if action_chunk_size!=25:
        #     import IPython
        #     IPython.embed()
        # assert action_chunk_size == 25 # TODO: this is temporary for debugging
        use_history = history_observations is not None

        # Shared projection layers (reused for history and current tokens)
        # For linear_projected_per_dim mode, we first project obs/action vectors
        # to smaller latent vectors, then tokenize each latent dimension.
        obs_proj = nn.Dense(self.hidden_dim, name='obs_projection')
        act_proj = nn.Dense(self.hidden_dim, name='action_projection')
        obs_down_proj = nn.Dense(self.obs_token_dim, name='obs_down_projection')
        act_down_proj = nn.Dense(self.action_token_dim, name='action_down_projection')

        # Modality embeddings (optional) 
        if self.use_modality_embeddings:
            state_type_embed = self.param(
                'state_type_embed',
                nn.initializers.normal(stddev=0.02),
                (1, self.hidden_dim),
            )
            action_type_embed = self.param(
                'action_type_embed',
                nn.initializers.normal(stddev=0.02),
                (1, self.hidden_dim),
            )

        def _project_obs(obs_slice):
            """Project observations to tokens based on tokenization mode."""
            if self.tokenization_mode == 'per_modality':
                tokens = obs_proj(obs_slice)[..., None, :]  # (..., 1, hidden_dim)
                token_count = 1
            elif self.tokenization_mode == 'per_dim':
                tokens = obs_proj(obs_slice[..., None])  # (..., obs_dim, hidden_dim)
                token_count = obs_dim
            elif self.tokenization_mode == 'linear_projected_per_dim':
                obs_latent = obs_down_proj(obs_slice)  # (..., obs_token_dim)
                tokens = obs_proj(obs_latent[..., None])  # (..., obs_token_dim, hidden_dim)
                token_count = self.obs_token_dim
            else:
                raise ValueError(
                    f"Unsupported tokenization_mode: {self.tokenization_mode}. "
                    "Use one of {'per_modality', 'per_dim', 'linear_projected_per_dim'}."
                )
            if self.use_modality_embeddings:
                embed = jnp.reshape(state_type_embed, (1,) * len(batch_shape) + (1, self.hidden_dim))
                embed = jnp.broadcast_to(embed, batch_shape + (token_count, self.hidden_dim))
                tokens = tokens + embed
            return tokens

        def _project_act(act_slice):
            """Project actions to tokens based on tokenization mode.

            If num_action_tokens > 1, the action is split evenly into num_action_tokens
            sub-vectors and each is projected to its own token, regardless of
            tokenization_mode.  This is controlled by the --agent.num_action_tokens flag.
            """
            nat = self.num_action_tokens
            if nat > 1:
                # Split flat action into nat sub-vectors, project each to a token
                sub_dim = act_slice.shape[-1] // nat
                act_split = act_slice.reshape(*batch_shape, nat, sub_dim)  # (..., nat, sub_dim)
                tokens = act_proj(act_split)  # (..., nat, hidden_dim)
                token_count = nat
            elif self.tokenization_mode == 'per_modality':
                tokens = act_proj(act_slice)[..., None, :]  # (..., 1, hidden_dim)
                token_count = 1
            elif self.tokenization_mode == 'per_dim':
                tokens = act_proj(act_slice[..., None])  # (..., act_dim, hidden_dim)
                token_count = action_dim
            elif self.tokenization_mode == 'linear_projected_per_dim':
                act_latent = act_down_proj(act_slice)  # (..., action_token_dim)
                tokens = act_proj(act_latent[..., None])  # (..., action_token_dim, hidden_dim)
                token_count = self.action_token_dim
            else:
                raise ValueError(
                    f"Unsupported tokenization_mode: {self.tokenization_mode}. "
                    "Use one of {'per_modality', 'per_dim', 'linear_projected_per_dim'}."
                )
            if self.use_modality_embeddings:
                embed = jnp.reshape(action_type_embed, (1,) * len(batch_shape) + (1, self.hidden_dim))
                embed = jnp.broadcast_to(embed, batch_shape + (token_count, self.hidden_dim))
                tokens = tokens + embed
            return tokens

        # Build token sequence 
        # Order: [hist_0_obs, hist_0_act, ..., hist_{T-1}_obs, hist_{T-1}_act,
        #         cur_obs_0, ..., cur_obs_{D-1}, cur_act_0, ..., cur_act_{A-1}]
        token_list = []
        if use_history:
            hist_length = history_observations.shape[-2]
            # history_observations: (..., hist_length, obs_dim)
            for k in range(hist_length):
                h_obs = history_observations[..., k, :] # (..., obs_dim)
                token_list.append(_project_obs(h_obs)) # (..., obs_dim, hidden_dim)

        # Current obs and act tokens
        if len(observations.shape) == 3:
            token_length = observations.shape[-2]
            # history_observations: (..., hist_length, obs_dim)
            for k in range(token_length):
                h_obs = observations[..., k, :] # (..., obs_dim)
                token_list.append(_project_obs(h_obs)) # (..., obs_dim, hidden_dim)

        else:
            token_list.append(_project_obs(observations)) # (..., obs_dim, hidden_dim)

        for k in range(actions.shape[1]):
            token_list.append(_project_act(actions[:,k])) # (..., act_dim, hidden_dim)

        # Concatenate along sequence axis
        tokens = jnp.concatenate(token_list, axis=-2)
        # Shape: (..., seq_len, hidden_dim)
        seq_len = tokens.shape[-2]

        # Positional embeddings 
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (seq_len, self.hidden_dim),
        )
        pos_shape = (1,) * len(batch_shape) + pos_embedding.shape
        tokens = tokens + jnp.reshape(pos_embedding, pos_shape)

        # CLS token 
        cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, self.hidden_dim),
        )
        cls_broadcast = jnp.tile(
            jnp.reshape(cls_token, (1,) * len(batch_shape) + (1, self.hidden_dim)),
            batch_shape + (1, 1),
        )
        tokens = jnp.concatenate([cls_broadcast, tokens], axis=-2)
        # Shape: (..., seq_len + 1, hidden_dim)

        # Transformer blocks 
        layer_attention_weights = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}',
            )
            if return_attention_weights:
                tokens, attn_weights = block(tokens, training=training, return_attention_weights=True)
                layer_attention_weights.append(attn_weights)
            else:
                tokens = block(tokens, training=training)

        tokens = nn.LayerNorm(name='final_ln')(tokens)
        cls_output = tokens[..., 0, :]

        # Ensemble Q heads  
        q_values = []
        for i in range(self.num_ensembles):
            q = nn.Dense(self.hidden_dim // 2, name=f'q_head_{i}_hidden')(cls_output)
            q = nn.relu(q)
            q = nn.Dense(1, name=f'q_head_{i}_output')(q)
            q_values.append(q)
        q_values = jnp.concatenate(q_values, axis=-1)

        if return_attention_weights:
            stacked_weights = (
                jnp.stack(layer_attention_weights, axis=0)
                if layer_attention_weights else None
            )
            return q_values, stacked_weights
        return q_values


def _flatten_history(history_observations):
    """Flatten history observations into a single vector for actor conditioning.

    Args:
        history_observations: (..., hist_length, obs_dim)

    Returns:
        (..., hist_length * obs_dim)
    """
    batch_shape = history_observations.shape[:-2]
    return history_observations.reshape(*batch_shape, -1)


class MTQLAgent(flax.struct.PyTreeNode):
    """TQL agent with transformer critic and flow matching actors."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _prepare_actor_obs(self, observations, history_observations=None):
        """Concatenate flattened history observations onto observations for actor input."""
        if history_observations is not None:
            hist_flat = _flatten_history(history_observations)
            return jnp.concatenate([hist_flat, observations], axis=-1)
        return observations

    def critic_loss(self, batch, grad_params, rng, step=None):
        rng, sample_rng, dropout_rng = jax.random.split(rng, 3)

        next_actions = self.sample_actions(
            batch['next_observations'],
            history_observations=batch.get('next_history_observations'),
            seed=sample_rng,
        )
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'],
            actions=next_actions,
            history_observations=batch.get('next_history_observations'),
            training=False,
        )
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=-1)
        else:
            next_q = next_qs.mean(axis=-1)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        use_attention_entropy_loss = bool(self.config['use_attention_entropy_loss'])
        q, attn_weights = self.network.select('critic')(
            batch['observations'],
            actions=batch['actions'],
            history_observations=batch.get('history_observations'),
            params=grad_params,
            training=True,
            rngs={'dropout': dropout_rng},
            return_attention_weights=True,
        )

        q_loss = jnp.square(q - target_q[:, None]).mean()

        attn_entropy = -jnp.sum(
            attn_weights * jnp.log(jnp.clip(attn_weights, a_min=1e-9)),
            axis=-1,
        )
        attn_entropy_cls_mean   = attn_entropy[..., 0].mean(axis=(1, 2))
        attn_entropy_other_mean = attn_entropy[..., 1:].mean(axis=(1, 2, 3))
        attn_entropy_layer_mean = jnp.stack(
            [attn_entropy_cls_mean, attn_entropy_other_mean], axis=-1
        )

        if use_attention_entropy_loss:
            temperature = self.network.select('attention_entropy_temperature')(params=grad_params)
            temp_min = self.config['attention_entropy_temperature_min']
            temp_max = self.config['attention_entropy_temperature_max']
            temperature_clipped = jnp.clip(temperature, temp_min, temp_max)
            temperature_no_grad = jax.lax.stop_gradient(temperature_clipped)

            attention_entropy_loss = (-temperature_no_grad * attn_entropy_layer_mean).mean()
            attn_entropy_layer_mean_no_grad = jax.lax.stop_gradient(attn_entropy_layer_mean)

            target_entropy = jnp.asarray(
                self.config['attention_entropy_target'],
                dtype=attn_entropy_layer_mean_no_grad.dtype,
            )
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
                info[f'attention_entropy_coef_layer_{i}_cls']   = temperature[i, 0]
                info[f'attention_entropy_coef_layer_{i}_other'] = temperature[i, 1]
        for i in range(int(attn_entropy_layer_mean.shape[0])):
            info[f'attention_entropy_mean_layer_{i}_cls']   = attn_entropy_layer_mean[i, 0]
            info[f'attention_entropy_mean_layer_{i}_other'] = attn_entropy_layer_mean[i, 1]

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # Augmented observations for actor (history observations flattened and appended)
        actor_obs = self._prepare_actor_obs(
            batch['observations'],
            batch.get('history_observations'),
        )

        # BC flow loss
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            actor_obs, x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        if self.config['extraction'] == 'implicit':
            return bc_flow_loss, {'bc_flow_loss': bc_flow_loss}

        # Distillation loss
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(actor_obs, noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            actor_obs, noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss
        actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'],
            actions=actor_actions_clipped,
            history_observations=batch.get('history_observations'),
            training=False,
        )
        q = jnp.mean(qs, axis=-1)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam    = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        actions = self.sample_actions(
            batch['observations'],
            history_observations=batch.get('history_observations'),
            seed=rng,
        )
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
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, step=step)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step=None):
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def _sample_actions_batch(self, observations, action_seed, history_observations=None):
        """Internal batched action sampling with optional history."""
        # Build actor observations (history observations flattened in)
        actor_obs = self._prepare_actor_obs(observations, history_observations)

        if 'implicit' in self.config['extraction']:
            orig_observations = observations
            if self.config['encoder'] is not None:
                actor_obs = self.network.select('actor_bc_flow_encoder')(actor_obs)

            action_seed, noise_seed = jax.random.split(action_seed)
            actions = jax.random.normal(
                action_seed,
                (*observations.shape[:-1], self.config['num_samples'], self.config['action_dim']),
            )
            n_actor_obs = jnp.repeat(jnp.expand_dims(actor_obs, 1), self.config['num_samples'], axis=1)
            n_orig_obs = jnp.repeat(jnp.expand_dims(orig_observations, 1), self.config['num_samples'], axis=1)
            n_hist_obs = (
                jnp.repeat(jnp.expand_dims(history_observations, 1), self.config['num_samples'], axis=1)
                if history_observations is not None else None
            )

            for i in range(self.config['flow_steps']):
                t = jnp.full((*observations.shape[:-1], self.config['num_samples'], 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(n_actor_obs, actions, t, is_encoded=True)
                actions = actions + vels / self.config['flow_steps']
            actions = jnp.clip(actions, -1, 1)

            # Q selection uses original (non-actor-augmented) obs and history tokens
            q = self.network.select('critic')(
                n_orig_obs,
                actions=actions,
                history_observations=n_hist_obs,
            ).min(axis=-1)
            actions = actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            noises  = jax.random.normal(action_seed, (*observations.shape[:-1], self.config['action_dim']))
            actions = self.network.select('actor_onestep_flow')(actor_obs, noises)
            actions = jnp.clip(actions, -1, 1)

        return actions

    def sample_actions(
        self,
        observations,
        history_observations=None,
        seed=None,
        temperature=1.0,
    ):
        action_seed = seed if seed is not None else jax.random.PRNGKey(0)
        if observations.ndim == 1:
            observations = observations[None, :]
            if history_observations is not None:
                history_observations = history_observations[None, ...]  # (1, T, obs_dim)
            return self._sample_actions_batch(observations, action_seed, history_observations)[0]
        return self._sample_actions_batch(observations, action_seed, history_observations)

    @jax.jit
    def compute_flow_actions(self, actor_obs, noises):
        """Compute actions from BC flow using Euler method. actor_obs already has history appended."""
        if self.config['encoder'] is not None:
            actor_obs = self.network.select('actor_bc_flow_encoder')(actor_obs)
        actions = noises
        for i in range(self.config['flow_steps']):
            t    = jnp.full((*actor_obs.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(actor_obs, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # History example inputs (None if not using history)
        ex_hist_obs = example_batch.get('history_observations')  # (B, T, obs_dim) or None

        # Actor observations: flatten history observations and concat if present
        if ex_hist_obs is not None:
            hist_flat = _flatten_history(ex_hist_obs)
            ex_actor_obs = jnp.concatenate([ex_observations, hist_flat], axis=-1)
        else:
            ex_actor_obs = ex_observations

        # Validate attention entropy target
        if config['use_attention_entropy_loss']:
            target_entropy = config['attention_entropy_target']
            if target_entropy is None:
                raise ValueError(
                    'attention_entropy_target cannot be None; provide a (num_layers, 2) target.'
                )
            ok = (
                isinstance(target_entropy, (list, tuple))
                and len(target_entropy) == config['num_layers']
                and all(isinstance(te, (list, tuple)) and len(te) == 2 for te in target_entropy)
            )
            if not ok:
                raise ValueError(
                    'attention_entropy_target must be a (num_layers, 2) nested list/tuple. '
                    f'Example for num_layers=2: ((3.0, 2.5), (1.0, 0.5)).'
                )

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
        else:
            encoders['critic'] = None
            encoders['actor_bc_flow'] = None
            encoders['actor_onestep_flow'] = None

        critic_def = TransformerValue(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            num_ensembles=config['num_ensembles'],
            encoder=encoders['critic'],
            use_modality_embeddings=config['use_modality_embeddings'],
            tokenization_mode=config['tokenization_mode'],
            obs_token_dim=config['obs_token_dim'],
            action_token_dim=config['action_token_dim'],
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders['actor_bc_flow'],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders['actor_onestep_flow'],
        )

        temp_shape = (config['num_layers'], 2)
        attention_temp_init = config['attention_entropy_temperature_init']
        if config['temp_type'] == 'exp':
            attention_temp_def = LogParam(init_value=attention_temp_init, shape=temp_shape)
        elif config['temp_type'] == 'sign':
            attention_temp_def = SignedExpParam(init_value=attention_temp_init, shape=temp_shape)
        else:
            raise ValueError("temp_type must be one of: 'exp', 'sign'")

        # Critic init args include history if present
        # critic_init_args = ()
        # critic_init_kwargs = dict(
        #     observations=ex_observations, actions=ex_actions, 
        #     history_observations=ex_hist_obs,
        #     history_actions=ex_hist_act,
        # )


        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions, ex_hist_obs), {}),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions, ex_hist_obs), {}),
            actor_bc_flow=(actor_bc_flow_def, (ex_actor_obs, ex_actions, ex_times), {}),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_actor_obs, ex_actions), {}),
            attention_entropy_temperature=(attention_temp_def, (), {}),
        )

        if encoders['actor_bc_flow'] is not None:
            network_info['actor_bc_flow_encoder'] = (encoders['actor_bc_flow'], (ex_actor_obs,), {})

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_kwargs = {k: v[2] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        init_rng_dict = {'params': init_rng, 'dropout': jax.random.PRNGKey(0)}
        # ModuleDict.init passes **network_kwargs per module if supported,
        # otherwise fall back to positional-only init.
        # network_params = network_def.init(
            # init_rng_dict,
            # **{k: (network_args[k], network_kwargs[k]) for k in network_args},
        # )['params']
        network_params = network_def.init(
            init_rng_dict,
            **network_args,   # critic=(ex_obs, ex_acts, ex_hist_obs, ex_hist_act)
        )['params']
        # NOTE: if your ModuleDict.init signature differs, replace the above with:
        # network_params = network_def.init(init_rng_dict, **network_args)['params']
        # and pass history kwargs only at call time, not during init.

        def make_optimizer(lr, params):
            optimizer_chain = []
            if config['critic_grad_clip'] > 0:
                optimizer_chain.append(optax.clip_by_global_norm(config['critic_grad_clip']))
            if config['warmup_steps'] > 0:
                warmup_steps = min(
                    config['warmup_steps'],
                    max(config['train_steps'] - 1, 0),
                )
                if warmup_steps == 0:
                    lr_schedule = lr
                else:
                    lr_schedule = optax.warmup_cosine_decay_schedule(
                        init_value=lr * 0.01,
                        peak_value=lr,
                        warmup_steps=warmup_steps,
                        decay_steps=max(config['train_steps'], warmup_steps + 1),
                        end_value=lr * 0.1,
                    )
            else:
                lr_schedule = lr

            if config['optimizer'] == 'muon':
                optimizer_chain.append(optax.contrib.muon(learning_rate=lr_schedule))
            elif config['optimizer'] == 'adamw':
                def should_apply_wd(path, value):
                    path_str = '/'.join(str(k.key if hasattr(k, 'key') else k) for k in path).lower()
                    if 'projection' in path_str:
                        return False
                    if any(x in path_str for x in [
                        'bias', 'layernorm', 'final_ln', 'pos_embedding',
                        'cls_token', 'state_type_embed', 'action_type_embed',
                    ]):
                        return False
                    if value.ndim <= 1:
                        return False
                    if 'modules_target_' in path_str:
                        return False
                    if 'modules_attention_entropy_temperature' in path_str:
                        return False
                    return True
                mask = jax.tree_util.tree_map_with_path(should_apply_wd, params)
                optimizer_chain.append(optax.adamw(
                    learning_rate=lr_schedule,
                    weight_decay=config['adamw_weight_decay'],
                    mask=mask,
                ))
            elif config['optimizer'] == 'adam':
                optimizer_chain.append(optax.adam(learning_rate=lr_schedule))
            else:
                raise ValueError(f"Optimizer type {config['optimizer']} not supported")
            return optax.chain(*optimizer_chain)

        critic_optimizer = make_optimizer(config['critic_lr'], network_params)
        actor_optimizer  = make_optimizer(config['actor_lr'],  network_params)

        def partition_fn(path, value):
            path_str = '/'.join(str(k.key if hasattr(k, 'key') else k) for k in path).lower()
            if 'modules_critic' in path_str or 'modules_attention_entropy_temperature' in path_str:
                return 'critic'
            return 'actor'

        param_labels = jax.tree_util.tree_map_with_path(partition_fn, network_params)
        network_tx   = optax.multi_transform(
            {'critic': critic_optimizer, 'actor': actor_optimizer},
            param_labels,
        )
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params['modules_target_critic'] = copy.deepcopy(network.params['modules_critic'])

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='mtql',
        ob_dims=ml_collections.config_dict.placeholder(list),
        action_dim=ml_collections.config_dict.placeholder(int),

        critic_lr=1e-4,
        actor_lr=5e-4,
        critic_grad_clip=0.0,

        batch_size=256,
        discount=0.99,
        tau=0.005,
        q_agg='mean',
        alpha=10.0,
        flow_steps=10,
        normalize_q_loss=False,
        extraction='fql',
        num_samples=16,

        optimizer='adamw',
        adamw_weight_decay=0.01,
        warmup_steps=10000,

        use_attention_entropy_loss=True,
        temp_type='exp',
        attention_entropy_temperature_init=1.0,
        attention_entropy_temperature_min=-100.0,
        attention_entropy_temperature_max=100.0,
        attention_entropy_target=((0.0, 0.0), (0.0, 0.0)),

        actor_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=True,

        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        dropout_rate=0.0,
        num_ensembles=2,
        use_modality_embeddings=True,
        tokenization_mode='per_modality',
        obs_token_dim=32,
        action_token_dim=8,

        encoder=ml_collections.config_dict.placeholder(str),
    ))
    return config
