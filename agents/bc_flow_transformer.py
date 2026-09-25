"""Transformer BC agent with flow matching.

Same architecture as bc_transformer.py but trains with a conditional flow
matching objective. The current noisy action x_t and time t are concatenated
to the observation before the transformer, so the velocity field is
time-dependent and multi-step Euler integration is meaningful.

Sampling: start from x0 ~ N(0, 1), integrate for `flow_steps` Euler steps.
"""

from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

import numpy as np

from agents.mtql import TransformerBlock
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

# ---------------------------------------------------------------------------
# 1D UNet with FiLM conditioning (Flax/JAX, channels-last: B, T, C)
# ---------------------------------------------------------------------------

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class Downsample1d(nn.Module):
    """Strided conv to halve temporal resolution."""
    dim: int

    @nn.compact
    def __call__(self, x):
        # x: (B, T, C) → (B, T/2, C)
        return nn.Conv(features=self.dim, kernel_size=(3,), strides=(2,), padding='SAME')(x)


class Upsample1d(nn.Module):
    """Transposed conv to double temporal resolution."""
    dim: int

    @nn.compact
    def __call__(self, x):
        # x: (B, T, C) → (B, 2T, C)
        return nn.ConvTranspose(features=self.dim, kernel_size=(4,), strides=(2,), padding='SAME')(x)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish"""
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.out_channels, kernel_size=(self.kernel_size,), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=self.n_groups)(x)
        return mish(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning and a residual connection.

    x    : (B, T, in_channels)
    cond : (B, cond_dim)
    out  : (B, T, out_channels)
    """
    in_channels: int
    out_channels: int
    cond_dim: int
    kernel_size: int = 3
    n_groups: int = 8

    def setup(self):
        self.block1 = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)
        self.block2 = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)
        self.cond_dense = nn.Dense(self.out_channels * 2)
        self.residual_proj = (
            nn.Dense(self.out_channels)
            if self.in_channels != self.out_channels
            else None
        )

    def __call__(self, x, cond):
        out = self.block1(x)

        # FiLM modulation: predict per-channel scale and bias
        embed = mish(cond)
        embed = self.cond_dense(embed)               # (B, out_channels*2)
        scale = embed[:, None, :self.out_channels]   # (B, 1, out_channels)
        bias  = embed[:, None, self.out_channels:]   # (B, 1, out_channels)
        out = scale * out + bias

        out = self.block2(out)
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        return out + residual


class ConditionalUnet1D(nn.Module):
    """Conditional 1D UNet for diffusion / flow matching.

    All tensors are channels-last: (B, T, C).

    input_dim              : action dimension
    global_cond_dim        : dim of the FiLM conditioning vector (e.g. cls token)
    diffusion_step_embed_dim: sinusoidal embedding size for timestep
    down_dims              : channel widths per UNet level
    kernel_size            : conv kernel size throughout
    n_groups               : GroupNorm groups
    """
    input_dim: int
    global_cond_dim: int
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int, ...] = (256, 256)
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(
        self,
        sample: jnp.ndarray,              # (B, T, input_dim)
        timestep,                          # (B,) or scalar
        global_cond: jnp.ndarray = None,  # (B, global_cond_dim)
    ) -> jnp.ndarray:
        dsed = self.diffusion_step_embed_dim
        all_dims = [self.input_dim] + list(self.down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        cond_dim = dsed + self.global_cond_dim
        mid_dim = all_dims[-1]
        start_dim = self.down_dims[0]
        ks, ng = self.kernel_size, self.n_groups

        # 1. Timestep embedding  →  (B, dsed)
        t = jnp.asarray(timestep).reshape(-1)  # handles scalar, (B,), or (B,1)

        feat = SinusoidalPosEmb(dsed, name='sinusoidal_emb')(t)
        feat = mish(nn.Dense(dsed * 4, name='step_proj1')(feat))
        feat = nn.Dense(dsed, name='step_proj2')(feat)

        if global_cond is not None:
            feat = jnp.concatenate([feat, global_cond], axis=-1)  # (B, cond_dim)

        # 2. Down path
        x = sample
        skips = []
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            x = ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, ks, ng,
                                           name=f'down_r1_{i}')(x, feat)
            x = ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, ks, ng,
                                           name=f'down_r2_{i}')(x, feat)
            skips.append(x)
            if not is_last:
                x = Downsample1d(dim_out, name=f'downsample_{i}')(x)

        # 3. Bottleneck
        x = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, ks, ng,
                                       name='mid_block1')(x, feat)
        x = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, ks, ng,
                                       name='mid_block2')(x, feat)

        # 4. Up path
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= len(in_out) - 1
            x = jnp.concatenate([x, skips.pop()], axis=-1)  # skip on channel dim
            x = ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, ks, ng,
                                           name=f'up_r1_{i}')(x, feat)
            x = ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, ks, ng,
                                           name=f'up_r2_{i}')(x, feat)
            if not is_last:
                x = Upsample1d(dim_in, name=f'upsample_{i}')(x)

        # 5. Final projection
        x = Conv1dBlock(start_dim, ks, ng, name='final_block')(x)
        x = nn.Dense(self.input_dim, name='final_proj')(x)
        # Trim to input length: upsample with stride=2 on odd input (e.g. 25 → 13 → 26)
        # produces one extra timestep; slice back to the original sequence length.
        return x[:, :sample.shape[1], :]

class TransformerActorFlow(nn.Module):
    """Same as TransformerActor but takes (x_t, t) concatenated onto obs.

    Token sequence:
        [CLS, hist_obs_0, ..., hist_obs_{T-1}, cur_obs_with_xt_t]

    Output is a velocity vector (no tanh), initialized near zero.
    """

    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    encoder: Any = None
    action_head: Any = None

    @nn.compact
    def __call__(
        self,
        observations,               # (..., obs_dim) or dict
        x_t,                        # (..., action_chunk, action_dim)
        t,                          # (..., 1)
        history_observations=None,  # (..., T, obs_dim) or dict or None
        training: bool = False,
    ):
        if self.encoder is not None:
            observations, history_observations = self.encoder(observations, history_observations)

        obs_proj = nn.Dense(self.hidden_dim, name='obs_projection')

        def _tok(proj, x):
            return proj(x)[..., None, :]

        if observations.ndim >= 3:
            # Patch tokens (B, P, embed_dim): project each patch → (B, P, hidden_dim).
            # history is always None for token encoders (compressed into current tokens).
            batch_shape = observations.shape[:-2]
            obs_tokens = obs_proj(observations)  # (B, P, hidden_dim)
        else:
            # Flat vector (B, obs_dim): whole obs becomes one token → (B, 1, hidden_dim).
            batch_shape = observations.shape[:-1]
            obs_tokens = _tok(obs_proj, observations)

        token_list = []
        if history_observations is not None:
            T = history_observations.shape[-2]
            for k in range(T):
                token_list.append(_tok(obs_proj, history_observations[..., k, :]))
        token_list.append(obs_tokens)
        # token_list.append(_tok(xt_proj, x_t_flat))
        # token_list.append(_tok(t_proj, t))

        tokens = jnp.concatenate(token_list, axis=-2)
        seq_len = tokens.shape[-2]

        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (seq_len, self.hidden_dim),
        )
        tokens = tokens + jnp.reshape(pos_embedding, (1,) * len(batch_shape) + pos_embedding.shape)

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

        for i in range(self.num_layers):
            tokens = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}',
            )(tokens, training=training)

        tokens = nn.LayerNorm(name='final_ln')(tokens)
        cls_out = tokens[..., 0, :]

        x = self.action_head(sample=x_t, timestep=t, global_cond=cls_out)

        return x

class BCFlowTransformerAgent(flax.struct.PyTreeNode):
    """Behavioral cloning with flow matching and a transformer actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    one_step: bool = nonpytree_field()

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

    def loss_flow_fn(self, batch, params, rng, noise=None):
        noise_rng, time_rng = jax.random.split(rng)
        actions = self._norm_actions(batch['actions'])  # (B, T, action_dim)
        batch_size = actions.shape[0]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        time  = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001  # (B,)
        time_expanded = time[:, None, None]  # (B, 1, 1)

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # target velocity: points from actions → noise

        obs      = self._norm_obs(batch['observations'])
        hist_obs = self._norm_hist_obs(batch.get('history_observations'))
        pred_vel = self.network.select('actor')(
            obs, x_t, time[:, None],
            history_observations=hist_obs,
            training=True,
            params=params,
        )
        loss = jnp.mean((pred_vel - u_t) ** 2)
        return loss, {'bc_flow_loss': loss}

    def loss_one_step_fn(self, batch, params, rng, noise=None):
        actions = self._norm_actions(batch['actions'])  # (B, T, action_dim)
        batch_size = actions.shape[0]
        if noise is None:
            noise = jnp.zeros(actions.shape)
        time  = jnp.zeros((batch_size,))
        time_expanded = time[:, None, None]  # (B, 1, 1)

        x_t = noise

        obs      = self._norm_obs(batch['observations'])
        hist_obs = self._norm_hist_obs(batch.get('history_observations'))
        pred_act = self.network.select('actor')(
            obs, x_t, time[:, None],
            history_observations=hist_obs,
            training=True,
            params=params,
        )
        loss = jnp.mean((pred_act - actions) ** 2)
        return loss, {'bc_one_step_loss': loss}

    @jax.jit
    def update(self, batch, step=None):
        rng, sub_rng = jax.random.split(self.rng)

        if self.one_step:
            loss_fn = lambda params: self.loss_one_step_fn(batch, params, sub_rng)
        else:
            loss_fn = lambda params: self.loss_flow_fn(batch, params, sub_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=rng, network=new_network), info

    def _sample_actions_batch(self, observations, action_seed, history_observations=None, noises=None, temperature=1.0, flow_steps=None):
        if self.one_step:
            return self._sample_actions_one_step_batch(observations, history_observations, action_seed, noises, temperature)
        else:
            return self._sample_actions_flow_batch(observations, history_observations, action_seed, noises, temperature, flow_steps)

    def _sample_actions_flow_batch(self, observations, history_observations, seed, noises=None, temperature=1.0, flow_steps=None):
        if isinstance(observations, dict):
            batch_shape = next(iter(observations.values())).shape[:1]
        else:
            batch_shape = observations.shape[:-1]

        obs      = self._norm_obs(observations)
        hist_obs = self._norm_hist_obs(history_observations)

        if flow_steps is None:
            flow_steps = self.config.get('flow_steps', 10)

        dt = -1.0 / flow_steps
        if noises is None:
            actions = jax.random.normal(seed, (*batch_shape, self.config["action_chunk_size"], self.config['action_dim']))
        else:
            actions = noises

        def step(carry):
            x_t, time = carry
            t = jnp.full((*batch_shape, 1), time)
            vels = self.network.select('actor')(
                obs, x_t, t,
                history_observations=hist_obs,
                training=False,
            )
            return x_t + dt * vels, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2  # robust to floating-point error

        actions, _ = jax.lax.while_loop(cond, step, (actions, 1.0))

        return self._denorm_actions(actions)


    def _sample_actions_one_step_batch(self, observations, history_observations, seed, noises=None, temperature=1.0):
        if isinstance(observations, dict):
            batch_shape = next(iter(observations.values())).shape[:1]
        else:
            batch_shape = observations.shape[:-1]

        obs      = self._norm_obs(observations)
        hist_obs = self._norm_hist_obs(history_observations)

        if noises is None:
            x_t = jnp.zeros((*batch_shape, self.config["action_chunk_size"], self.config['action_dim']))
        else:
            x_t = noises

        t = jnp.zeros((*batch_shape, 1))
        actions = self.network.select('actor')(
            obs, x_t, t,
            history_observations=hist_obs,
            training=False,
        )

        return self._denorm_actions(actions)

    def compute_recon_mse(self, batch, seed=None, temperature=0.0):
        action_seed = seed if seed is not None else jax.random.PRNGKey(0)
        gt = batch['actions']
        result = {}
        for n_steps in [10, 20, 50, 100]:
            pred = self._sample_actions_batch(
                batch['observations'],
                action_seed,
                batch.get('history_observations'),
                temperature=temperature,
                flow_steps=n_steps,
            )
            result[f'recon_mse_{n_steps}steps'] = float(jnp.mean((pred - gt) ** 2))
        return result

    @jax.jit
    def sample_actions(
        self,
        observations,
        history_observations=None,
        seed=None,
        noises=None,
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

        action_seed = seed if seed is not None else jax.random.PRNGKey(0)

        if _is_unbatched(observations):
            observations = _add_batch(observations)
            if history_observations is not None:
                history_observations = _add_batch(history_observations)
            return self._sample_actions_batch(
                observations,
                action_seed,
                history_observations,
                noises=noises,
                temperature=temperature,
            )[0]

        return self._sample_actions_batch(
            observations,
            action_seed,
            history_observations,
            noises=noises,
            temperature=temperature,
        )

    @classmethod
    def create(cls, seed, example_batch, config, optimizer=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_obs     = example_batch['observations']
        ex_actions = example_batch['actions']
        action_dim = ex_actions.shape[-1]
        ex_hist_obs = example_batch.get('history_observations')

        # Example x_t and t for shape tracing
        ex_x_t = ex_actions
        ex_t   = ex_actions[...,0, :1] # TODO: generate a random vector instead

        enc_name = config.get('encoder', None)
        encoder  = encoder_modules[enc_name]() if enc_name else None

        action_head = ConditionalUnet1D(action_dim, config['hidden_dim']) # TODO fix this

        actor_def = TransformerActorFlow(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            encoder=encoder,
            action_head=action_head
        )

        network_info = dict(
            actor=(actor_def, (ex_obs, ex_x_t, ex_t, ex_hist_obs), {}),
        )
        networks     = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def  = ModuleDict(networks)

        network_params = network_def.init(
            {'params': init_rng, 'dropout': jax.random.PRNGKey(0)},
            **network_args,
        )['params']

        if optimizer is None:
            warmup_steps = config.get('warmup_steps', 0)
            train_steps  = config.get('train_steps', 0)
            if warmup_steps > 0 and train_steps > warmup_steps:
                lr_schedule = optax.warmup_cosine_decay_schedule(
                    init_value=config['lr'] * 0.01,
                    peak_value=config['lr'],
                    warmup_steps=warmup_steps,
                    decay_steps=train_steps - warmup_steps,
                    end_value=config['lr'] * 0.1,
                )
            else:
                lr_schedule = config['lr']
            optimizer = optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=config['weight_decay'],
            )
        network = TrainState.create(network_def, network_params, tx=optimizer)

        config['action_dim']   = action_dim
        config['action_min']  = example_batch['action_min'].tolist()
        config['action_max']  = example_batch['action_max'].tolist()
        config['proprio_min'] = example_batch['proprio_min'].tolist()
        config['proprio_max'] = example_batch['proprio_max'].tolist()
        return cls(rng, network=network, config=flax.core.FrozenDict(**config), one_step=config.get('one_step', False))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='bc_flow_transformer',

        # lr=3e-4,
        lr=1e-4,
        weight_decay=0.01,
        batch_size=256,

        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        dropout_rate=0.0,

        flow_steps=10,
        one_step=False,
        warmup_steps=0,

        # Accepted from command line but unused:
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
