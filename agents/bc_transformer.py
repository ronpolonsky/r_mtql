"""Transformer Behavioral Cloning agent.

Simple imitation learning: a transformer encodes the current observation plus
a sparse history of (obs, action) pairs and directly regresses to the
demonstrated action via MSE loss.  No Q-learning, no target network.
"""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.mtql import TransformerBlock
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


# ---------------------------------------------------------------------------
# Transformer actor module
# ---------------------------------------------------------------------------

class TransformerActor(nn.Module):
    """Transformer that maps (history, current obs) -> action.

    Token sequence (per_modality tokenization):
        [CLS, hist_obs_0, hist_act_0, ..., hist_obs_{T-1}, hist_act_{T-1}, cur_obs]

    The CLS token output is fed through a small MLP head to produce actions.
    """

    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    action_dim: int = 4
    encoder: Any = None

    @nn.compact
    def __call__(
        self,
        observations,           # (..., obs_dim) or dict
        history_observations=None,  # (..., T, obs_dim) or dict or None
        history_actions=None,       # (..., T, act_dim) or None
        training: bool = False,
    ):
        if self.encoder is not None:
            observations, history_observations = self.encoder(observations, history_observations)

        batch_shape = observations.shape[:-1]
        obs_dim = observations.shape[-1]
        use_history = history_observations is not None

        obs_proj = nn.Dense(self.hidden_dim, name='obs_projection')

        def _tok_obs(x):
            return obs_proj(x)[..., None, :]   # (..., 1, hidden_dim)

        # Build token list
        token_list = []
        if use_history:
            T = history_observations.shape[-2]
            for k in range(T):
                token_list.append(_tok_obs(history_observations[..., k, :]))
        token_list.append(_tok_obs(observations))   # current obs token

        tokens = jnp.concatenate(token_list, axis=-2)  # (..., seq_len, hidden_dim)
        seq_len = tokens.shape[-2]

        # Positional embedding
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (seq_len, self.hidden_dim),
        )
        tokens = tokens + jnp.reshape(pos_embedding, (1,) * len(batch_shape) + pos_embedding.shape)

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

        # Transformer blocks
        for i in range(self.num_layers):
            tokens = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}',
            )(tokens, training=training)

        tokens = nn.LayerNorm(name='final_ln')(tokens)
        cls_out = tokens[..., 0, :]   # (..., hidden_dim)

        # Action head
        x = nn.Dense(self.hidden_dim, name='action_head_hidden')(cls_out)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, name='action_head_output')(x) # TODO: we removed a tanh from here


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BCTransformerAgent(flax.struct.PyTreeNode):
    """Behavioral cloning with a transformer actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _norm_actions(self, actions):
        lo  = jnp.asarray(self.config['action_min'])
        hi  = jnp.asarray(self.config['action_max'])
        return 2.0 * (actions - lo) / (hi - lo + 1e-8) - 1.0

    def _denorm_actions(self, actions):
        lo  = jnp.asarray(self.config['action_min'])
        hi  = jnp.asarray(self.config['action_max'])
        return (actions + 1.0) / 2.0 * (hi - lo + 1e-8) + lo

    def _norm_obs(self, obs):
        if not isinstance(obs, dict) or self.config.get('proprio_min') is None:
            return obs
        lo  = jnp.asarray(self.config['proprio_min'])
        hi  = jnp.asarray(self.config['proprio_max'])
        return {**obs, 'proprio': 2.0 * (obs['proprio'].astype(jnp.float32) - lo) / (hi - lo + 1e-8) - 1.0}

    def _norm_hist_obs(self, hist_obs):
        if hist_obs is None or not isinstance(hist_obs, dict) or self.config.get('proprio_min') is None:
            return hist_obs
        lo  = jnp.asarray(self.config['proprio_min'])
        hi  = jnp.asarray(self.config['proprio_max'])
        return {**hist_obs, 'proprio': 2.0 * (hist_obs['proprio'].astype(jnp.float32) - lo) / (hi - lo + 1e-8) - 1.0}

    # ------------------------------------------------------------------

    @jax.jit
    def update(self, batch, step=None):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            obs      = self._norm_obs(batch['observations'])
            hist_obs = self._norm_hist_obs(batch.get('history_observations'))
            pred_actions = self.network.select('actor')(
                obs,
                history_observations=hist_obs,
                training=True,
                rngs={'dropout': rng},
                params=params,
            )
            norm_targets = self._norm_actions(batch['actions'])
            loss = jnp.mean((pred_actions - norm_targets) ** 2)
            return loss, {'bc_loss': loss, 'mse': loss}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def _sample_actions_batch(self, observations, history_observations, seed):
        obs      = self._norm_obs(observations)
        hist_obs = self._norm_hist_obs(history_observations)
        pred_norm = self.network.select('actor')(obs, history_observations=hist_obs, training=False)
        return jnp.clip(self._denorm_actions(pred_norm), -1, 1)

    @jax.jit
    def sample_actions(
        self,
        observations,
        history_observations=None,
        seed=None,
        temperature=1.0,
    ):
        def _is_unbatched(obs):
            if isinstance(obs, dict):
                return next(iter(obs.values())).ndim == 1
            return obs.ndim == 1

        def _add_batch(obs):
            if isinstance(obs, dict):
                return {k: v[None] for k, v in obs.items()}
            return obs[None]

        if _is_unbatched(observations):
            observations = _add_batch(observations)
            if history_observations is not None:
                history_observations = _add_batch(history_observations)
            return self._sample_actions_batch(observations, history_observations, seed)[0]

        return self._sample_actions_batch(observations, history_observations, seed)

    @jax.jit
    def compute_recon_mse(self, batch, seed=None):
        pred_actions = self._sample_actions_batch(
            batch['observations'],
            batch.get('history_observations'),
            seed,
        )
        return jnp.mean((pred_actions - batch['actions']) ** 2)

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_obs = example_batch['observations']
        action_dim = example_batch['actions'].shape[-1]
        ex_hist_obs = example_batch.get('history_observations')

        enc_name = config.get('encoder', None)
        encoder = encoder_modules[enc_name]() if enc_name else None

        actor_def = TransformerActor(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            action_dim=action_dim,
            encoder=encoder,
        )

        network_info = dict(
            actor=(actor_def, (ex_obs, ex_hist_obs), {}),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        network_params = network_def.init(
            {'params': init_rng, 'dropout': jax.random.PRNGKey(0)},
            **network_args,
        )['params']

        optimizer = optax.adamw(
            learning_rate=config['lr'],
            weight_decay=config['weight_decay'],
        )
        network = TrainState.create(network_def, network_params, tx=optimizer)

        config['action_dim']   = action_dim
        config['action_min']  = example_batch['action_min'].tolist()
        config['action_max']  = example_batch['action_max'].tolist()
        config['proprio_min'] = example_batch['proprio_min'].tolist()
        config['proprio_max'] = example_batch['proprio_max'].tolist()
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='bc_transformer',

        lr=1e-4,
        weight_decay=0.01,
        batch_size=256,

        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        dropout_rate=0.0,

        # Accepted from command line but unused by BC:
        alpha=1.0,
        attention_entropy_target=((0.0, 0.0), (0.0, 0.0)),
        tokenization_mode='per_modality',
        critic_grad_clip=float('inf'),
        action_chunk_size=1,
        num_action_tokens=1,

        encoder=ml_collections.config_dict.placeholder(str),
        action_dim=ml_collections.config_dict.placeholder(int),
    ))
    return config
