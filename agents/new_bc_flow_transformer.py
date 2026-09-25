"""Standalone flow-BC agent matching MTQL's ``actor_bc_flow``.

This agent intentionally contains only MTQL's flow-matching actor.  It uses
the same actor definition, flow-matching objective, and reverse Euler sampler
as ``MTQLTransformerAgent``, without a critic or a distilled one-step actor.
Observations and actions are consumed in their dataset/environment coordinates;
there is no BC-specific min/max normalization.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.bc_flow_transformer import ConditionalUnet1D, TransformerActorFlow
from utils.encoders import _is_dict_obs, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class NewBCFlowTransformerAgent(flax.struct.PyTreeNode):
    """Behavior cloning using MTQL's flow actor and flow-matching loss."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss_flow_fn(self, batch, params, rng, noise=None):
        """MTQL's flow-matching objective, without critic-related losses."""
        noise_rng, time_rng = jax.random.split(rng)
        actions = batch['actions']
        batch_size = actions.shape[0]

        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        time_expanded = time[:, None, None]

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        pred_vel = self.network.select('actor_bc_flow')(
            batch['observations'],
            x_t,
            time[:, None],
            history_observations=batch.get('history_observations'),
            training=True,
            params=params,
        )

        per_sample_loss = jnp.mean((pred_vel - u_t) ** 2, axis=(1, 2))
        actor_mask = batch.get('actor_mask', jnp.ones((batch_size,)))
        actor_mask = actor_mask.astype(per_sample_loss.dtype)
        actor_denom = jnp.maximum(actor_mask.sum(), 1.0)
        loss = jnp.sum(per_sample_loss * actor_mask) / actor_denom
        return loss, {
            'bc_flow_loss': loss,
            'actor_batch_fraction': actor_mask.mean(),
        }

    @jax.jit
    def update(self, batch, step=None):
        """Apply one flow-matching update; ``step`` is API-compatible only."""
        del step
        # Follow MTQL's actor-side RNG splitting so a fixed agent RNG produces
        # the same flow-matching noise and time samples as actor_bc_flow.
        new_rng, update_rng = jax.random.split(self.rng)
        _, actor_rng, _ = jax.random.split(update_rng, 3)
        loss_rng, _ = jax.random.split(actor_rng, 2)

        def loss_fn(grad_params):
            return self.actor_loss_flow_fn(batch, grad_params, loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, network=new_network), info

    @staticmethod
    def _batch_shape(observations):
        if _is_dict_obs(observations):
            return next(iter(observations.values())).shape[:1]
        if observations.ndim >= 3:
            return observations.shape[:1]
        return observations.shape[:-1]

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises=None,
        history_observations=None,
        seed=None,
    ):
        """Integrate MTQL's learned velocity field from noise to actions."""
        batch_shape = self._batch_shape(observations)
        flow_steps = self.config.get('flow_steps', 10)
        dt = -1.0 / flow_steps

        if noises is None:
            actions = jax.random.normal(
                seed,
                (*batch_shape, self.config['action_chunk_size'], self.config['action_dim']),
            )
        else:
            actions = noises

        def step(carry):
            x_t, time = carry
            t = jnp.full((*batch_shape, 1), time)
            vels = self.network.select('actor_bc_flow')(
                observations,
                x_t,
                t,
                history_observations=history_observations,
                training=False,
            )
            return x_t + dt * vels, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (actions, 1.0))
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def _sample_actions_batch(
        self,
        observations,
        action_seed,
        history_observations=None,
        noises=None,
    ):
        return self.compute_flow_actions(
            observations,
            noises=noises,
            history_observations=history_observations,
            seed=action_seed,
        )

    def sample_actions(
        self,
        observations,
        history_observations=None,
        seed=None,
        noises=None,
        temperature=1.0,
    ):
        """Sample an action chunk through the multi-step flow actor.

        ``temperature`` is accepted for evaluator compatibility and intentionally
        ignored, matching MTQL's flow sampler.
        """
        del temperature
        action_seed = seed if seed is not None else jax.random.PRNGKey(0)

        if _is_dict_obs(observations):
            if 'proprio' in observations:
                is_single = observations['proprio'].ndim < 2
            else:
                is_single = next(iter(observations.values())).ndim < 4
        else:
            is_single = observations.ndim == 1

        if is_single:
            observations = jax.tree_util.tree_map(lambda x: x[None], observations)
            if history_observations is not None:
                history_observations = jax.tree_util.tree_map(
                    lambda x: x[None], history_observations
                )
            if noises is not None:
                noises = noises[None]
            return self._sample_actions_batch(
                observations,
                action_seed,
                history_observations,
                noises,
            )[0]

        return self._sample_actions_batch(
            observations,
            action_seed,
            history_observations,
            noises,
        )

    @classmethod
    def create(cls, seed, example_batch, config, optimizer=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_times = ex_actions[..., 0, :1]
        ex_hist_obs = example_batch.get('history_observations')
        action_dim = ex_actions.shape[-1]

        enc_name = config.get('encoder', None)
        encoder = encoder_modules[enc_name]() if enc_name else None
        actor_def = TransformerActorFlow(
            hidden_dim=config['hidden_dim'],
            num_layers=config['actor_num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            encoder=encoder,
            action_head=ConditionalUnet1D(action_dim, config['hidden_dim']),
        )

        network_info = {
            'actor_bc_flow': (
                actor_def,
                (ex_observations, ex_actions, ex_times, ex_hist_obs),
                {},
            ),
        }
        network_def = ModuleDict(
            {name: value[0] for name, value in network_info.items()}
        )
        network_args = {name: value[1] for name, value in network_info.items()}
        network_params = network_def.init(
            {'params': init_rng, 'dropout': jax.random.PRNGKey(0)},
            **network_args,
        )['params']

        if optimizer is None:
            optimizer = cls._make_optimizer(config, network_params)
        network = TrainState.create(network_def, network_params, tx=optimizer)

        config['action_dim'] = action_dim
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )

    @staticmethod
    def _make_optimizer(config, params):
        """Build the optimizer used for MTQL's actor parameters."""
        optimizer_chain = []
        grad_clip = config.get(
            'actor_grad_clip', config.get('critic_grad_clip', 0.0)
        )
        if grad_clip > 0:
            optimizer_chain.append(optax.clip_by_global_norm(grad_clip))

        lr = config['actor_lr']
        warmup_steps = config.get('warmup_steps', 0)
        train_steps = config.get('train_steps', 0)
        if warmup_steps > 0 and train_steps > warmup_steps:
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=lr * 0.01,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=train_steps - warmup_steps,
                end_value=lr * 0.1,
            )
        else:
            learning_rate = lr

        optimizer_name = config.get('optimizer', 'adamw')
        if optimizer_name == 'adamw':
            def should_apply_wd(path, value):
                path_str = '/'.join(
                    str(key.key if hasattr(key, 'key') else key) for key in path
                ).lower()
                if 'projection' in path_str:
                    return False
                if any(
                    name in path_str
                    for name in (
                        'bias',
                        'layernorm',
                        'final_ln',
                        'pos_embedding',
                        'cls_token',
                        'state_type_embed',
                        'action_type_embed',
                        'modules_target_',
                        'modules_attention_entropy_temperature',
                    )
                ):
                    return False
                return value.ndim > 1

            weight_decay_mask = jax.tree_util.tree_map_with_path(
                should_apply_wd, params
            )
            optimizer_chain.append(
                optax.adamw(
                    learning_rate=learning_rate,
                    weight_decay=config['adamw_weight_decay'],
                    mask=weight_decay_mask,
                )
            )
        elif optimizer_name == 'adam':
            optimizer_chain.append(optax.adam(learning_rate=learning_rate))
        else:
            raise ValueError(f'Optimizer {optimizer_name} not supported')

        return optax.chain(*optimizer_chain)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='new_bc_flow_transformer',
            actor_lr=1e-4,
            actor_grad_clip=5.0,
            batch_size=256,
            optimizer='adamw',
            adamw_weight_decay=0.01,
            warmup_steps=10000,
            train_steps=1000000,
            hidden_dim=256,
            actor_num_layers=2,
            num_heads=4,
            mlp_ratio=4,
            dropout_rate=0.0,
            flow_steps=10,
            encoder=ml_collections.config_dict.placeholder(str),
            action_dim=ml_collections.config_dict.placeholder(int),
            action_chunk_size=1,
        )
    )
