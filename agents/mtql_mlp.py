"""MTQL simulator agent with a parameter-matched ExPO-style MLP critic."""

import copy

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.bc_flow_transformer import ConditionalUnet1D, TransformerActorFlow
from agents.mtql_transformer import (
    MTQLTransformerAgent,
    get_config as get_transformer_config,
)
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState


class ExPOQMLP(nn.Module):
    """A scalar Q network using the MLP ordering from the ExPO implementation."""

    hidden_dim: int
    num_layers: int = 4

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                name=f'hidden_{i}',
            )(x)
            x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
            x = nn.relu(x)

        q = nn.Dense(
            1,
            kernel_init=nn.initializers.xavier_uniform(),
            name='q_output',
        )(x)
        return jnp.squeeze(q, axis=-1)


class MLPValue(nn.Module):
    """Q ensemble that replaces transformer blocks while retaining tokenization.

    The encoder, per-modality projections, modality embeddings, positional
    embeddings, and CLS token match ``TransformerValue``. The complete token
    sequence is flattened and passed to independent ExPO-style Q MLPs.
    """

    token_hidden_dim: int = 256
    mlp_hidden_dim: int = 67
    mlp_num_layers: int = 4
    num_ensembles: int = 2
    encoder: object = None
    use_modality_embeddings: bool = True
    tokenization_mode: str = 'per_modality'
    num_action_tokens: int = 1

    @nn.compact
    def __call__(
        self,
        observations,
        actions,
        history_observations=None,
        training: bool = False,
    ):
        del training  # ExPO's critic MLP has no dropout by default.

        if self.tokenization_mode != 'per_modality':
            raise ValueError(
                'MTQLMLPAgent currently requires tokenization_mode="per_modality".'
            )

        if self.encoder is not None:
            observations, history_observations = self.encoder(
                observations, history_observations
            )

        if observations.ndim >= 3:
            batch_shape = observations.shape[:-2]
        else:
            batch_shape = observations.shape[:-1]

        obs_projection = nn.Dense(
            self.token_hidden_dim, name='obs_projection'
        )
        action_projection = nn.Dense(
            self.token_hidden_dim, name='action_projection'
        )

        if self.use_modality_embeddings:
            state_type_embed = self.param(
                'state_type_embed',
                nn.initializers.normal(stddev=0.02),
                (1, self.token_hidden_dim),
            )
            action_type_embed = self.param(
                'action_type_embed',
                nn.initializers.normal(stddev=0.02),
                (1, self.token_hidden_dim),
            )

        def project_observation(observation):
            tokens = obs_projection(observation)[..., None, :]
            if self.use_modality_embeddings:
                embed_shape = (
                    (1,) * len(batch_shape) + (1, self.token_hidden_dim)
                )
                tokens = tokens + jnp.reshape(state_type_embed, embed_shape)
            return tokens

        def project_action(action):
            if self.num_action_tokens > 1:
                action_dim = action.shape[-1]
                if action_dim % self.num_action_tokens != 0:
                    raise ValueError(
                        f'action_dim ({action_dim}) must be divisible by '
                        f'num_action_tokens ({self.num_action_tokens}).'
                    )
                action = action.reshape(
                    *batch_shape,
                    self.num_action_tokens,
                    action_dim // self.num_action_tokens,
                )
                tokens = action_projection(action)
            else:
                tokens = action_projection(action)[..., None, :]

            if self.use_modality_embeddings:
                embed_shape = (
                    (1,) * len(batch_shape) + (1, self.token_hidden_dim)
                )
                tokens = tokens + jnp.reshape(action_type_embed, embed_shape)
            return tokens

        token_list = []
        if history_observations is not None:
            for k in range(history_observations.shape[-2]):
                token_list.append(
                    project_observation(history_observations[..., k, :])
                )

        if observations.ndim == 3:
            for k in range(observations.shape[-2]):
                token_list.append(project_observation(observations[..., k, :]))
        else:
            token_list.append(project_observation(observations))

        for k in range(actions.shape[-2]):
            token_list.append(project_action(actions[..., k, :]))

        tokens = jnp.concatenate(token_list, axis=-2)
        sequence_length = tokens.shape[-2]

        positional_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (sequence_length, self.token_hidden_dim),
        )
        positional_shape = (
            (1,) * len(batch_shape) + positional_embedding.shape
        )
        tokens = tokens + jnp.reshape(positional_embedding, positional_shape)

        cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, self.token_hidden_dim),
        )
        cls_token = jnp.tile(
            jnp.reshape(
                cls_token,
                (1,) * len(batch_shape) + (1, self.token_hidden_dim),
            ),
            batch_shape + (1, 1),
        )
        tokens = jnp.concatenate([cls_token, tokens], axis=-2)
        mlp_input = tokens.reshape(*batch_shape, -1)

        q_values = []
        for i in range(self.num_ensembles):
            q_values.append(
                ExPOQMLP(
                    hidden_dim=self.mlp_hidden_dim,
                    num_layers=self.mlp_num_layers,
                    name=f'q_mlp_{i}',
                )(mlp_input)
            )
        return jnp.stack(q_values, axis=-1)


class MTQLMLPAgent(MTQLTransformerAgent):
    """Simulator MTQL agent with unchanged actors and an MLP Q ensemble."""

    def critic_loss(self, batch, grad_params, rng, step=None):
        del step
        _, sample_rng, dropout_rng = jax.random.split(rng, 3)

        next_actions = self.sample_actions(
            batch['next_observations'],
            history_observations=batch['next_history_observations'],
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

        target_q = (
            batch['rewards']
            + self.config['discount'] ** self.config['action_chunk_size']
            * batch['masks']
            * next_q
        )

        q = self.network.select('critic')(
            batch['observations'],
            actions=batch['actions'],
            history_observations=batch.get('history_observations'),
            params=grad_params,
            training=True,
            rngs={'dropout': dropout_rng},
        )
        q_loss = jnp.square(q - target_q[:, None]).mean()

        return q_loss, {
            'critic_loss': q_loss,
            'q_loss': q_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'rewards_mean': batch['rewards'].mean(),
            'target_q_mean': target_q.mean(),
        }

    @classmethod
    def create(cls, seed, example_batch, config):
        if config['tokenization_mode'] != 'per_modality':
            raise ValueError(
                'MTQLMLPAgent requires --agent.tokenization_mode=per_modality.'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_times = ex_actions[..., 0, :1]
        ex_hist_obs = example_batch.get('history_observations')
        action_dim = ex_actions.shape[-1]

        enc_name = config.get('encoder', None)

        def make_encoder():
            return encoder_modules[enc_name]() if enc_name else None

        critic_def = MLPValue(
            token_hidden_dim=config['hidden_dim'],
            mlp_hidden_dim=config['critic_mlp_hidden_dim'],
            mlp_num_layers=config['critic_mlp_num_layers'],
            num_ensembles=config['num_ensembles'],
            encoder=make_encoder(),
            use_modality_embeddings=config['use_modality_embeddings'],
            tokenization_mode=config['tokenization_mode'],
            num_action_tokens=config['num_action_tokens'],
        )

        network_info = dict(
            critic=(
                critic_def,
                (ex_observations, ex_actions, ex_hist_obs),
                {},
            ),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_actions, ex_hist_obs),
                {},
            ),
            actor_onestep_flow=(
                TransformerActorFlow(
                    hidden_dim=config['hidden_dim'],
                    num_layers=config['actor_num_layers'],
                    num_heads=config['num_heads'],
                    mlp_ratio=config['mlp_ratio'],
                    dropout_rate=config['dropout_rate'],
                    encoder=make_encoder(),
                    action_head=ConditionalUnet1D(
                        action_dim, config['hidden_dim']
                    ),
                ),
                (ex_observations, ex_actions, ex_times, ex_hist_obs),
                {},
            ),
        )

        flow_def = TransformerActorFlow(
            hidden_dim=config['hidden_dim'],
            num_layers=config['actor_num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            encoder=make_encoder(),
            action_head=ConditionalUnet1D(
                action_dim, config['hidden_dim']
            ),
        )
        network_info['actor_bc_flow'] = (
            flow_def,
            (ex_observations, ex_actions, ex_times, ex_hist_obs),
            {},
        )

        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}
        network_def = ModuleDict(networks)

        network_params = network_def.init(
            {'params': init_rng, 'dropout': jax.random.PRNGKey(0)},
            **network_args,
        )['params']

        def make_optimizer(lr, params):
            optimizer_chain = []
            if config['critic_grad_clip'] > 0:
                optimizer_chain.append(
                    optax.clip_by_global_norm(config['critic_grad_clip'])
                )
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
                        decay_steps=max(
                            config['train_steps'], warmup_steps + 1
                        ),
                        end_value=lr * 0.1,
                    )
            else:
                lr_schedule = lr

            if config['optimizer'] == 'adamw':
                def should_apply_wd(path, value):
                    path_str = '/'.join(
                        str(key.key if hasattr(key, 'key') else key)
                        for key in path
                    ).lower()
                    if 'projection' in path_str:
                        return False
                    if any(
                        name in path_str
                        for name in [
                            'bias',
                            'layernorm',
                            'layer_norm',
                            'pos_embedding',
                            'cls_token',
                            'state_type_embed',
                            'action_type_embed',
                            'modules_target_',
                        ]
                    ):
                        return False
                    return value.ndim > 1

                mask = jax.tree_util.tree_map_with_path(
                    should_apply_wd, params
                )
                optimizer_chain.append(
                    optax.adamw(
                        learning_rate=lr_schedule,
                        weight_decay=config['adamw_weight_decay'],
                        mask=mask,
                    )
                )
            elif config['optimizer'] == 'adam':
                optimizer_chain.append(
                    optax.adam(learning_rate=lr_schedule)
                )
            else:
                raise ValueError(
                    f"Optimizer {config['optimizer']} not supported"
                )
            return optax.chain(*optimizer_chain)

        critic_optimizer = make_optimizer(
            config['critic_lr'], network_params
        )
        actor_optimizer = make_optimizer(
            config['actor_lr'], network_params
        )

        def partition_fn(path, value):
            del value
            path_str = '/'.join(
                str(key.key if hasattr(key, 'key') else key)
                for key in path
            ).lower()
            if 'modules_critic' in path_str:
                return 'critic'
            return 'actor'

        param_labels = jax.tree_util.tree_map_with_path(
            partition_fn, network_params
        )
        network_tx = optax.multi_transform(
            {'critic': critic_optimizer, 'actor': actor_optimizer},
            param_labels,
        )
        network = TrainState.create(
            network_def, network_params, tx=network_tx
        )
        network.params['modules_target_critic'] = copy.deepcopy(
            network.params['modules_critic']
        )

        config['action_dim'] = action_dim
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = get_transformer_config()
    config.agent_name = 'mtql_mlp'
    # Matches the H20/C25, 256-d token transformer critic within 0.31%.
    config.critic_mlp_hidden_dim = 67
    config.critic_mlp_num_layers = 4
    return config
