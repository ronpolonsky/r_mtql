"""MTQL with transformer actor."""

import copy
from typing import Any

import flax
import flax.linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.mtql import TransformerBlock, TransformerValue
from agents.bc_flow_transformer import BCFlowTransformerAgent, TransformerActorFlow, ConditionalUnet1D
from utils.encoders import encoder_modules, _is_dict_obs
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import LogParam, SignedExpParam


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MTQLTransformerAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng, step=None):
        rng, sample_rng, dropout_rng = jax.random.split(rng, 3)

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

        target_q = batch['rewards'] + self.config['discount'] ** self.config["action_chunk_size"] * batch['masks'] * next_q

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
            attn_weights * jnp.log(jnp.clip(attn_weights, a_min=1e-9)), axis=-1
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
            temperature_loss       = jnp.asarray(0.0, dtype=q.dtype)

        critic_loss = q_loss + attention_entropy_loss + temperature_loss

        info = {
            'critic_loss': critic_loss,
            'q_loss': q_loss,
            'attention_entropy_loss': attention_entropy_loss,
            'temperature_loss': temperature_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'rewards_mean': batch["rewards"].mean(),
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


    def _sample_actions_one_step_batch(self, observations, history_observations, training=False, params=None, seed=None, noises=None, temperature=1.0):
        if _is_dict_obs(observations):
            batch_shape = next(iter(observations.values())).shape[:1]
        else:
            batch_shape = observations.shape[:-1]

        if noises is None:
            x_t  = jax.random.normal(seed, (*batch_shape, self.config["action_chunk_size"], self.config['action_dim']))
        else:
            x_t = noises

        t = jnp.zeros((*batch_shape, 1))
        actions = self.network.select('actor_onestep_flow')(
            observations, x_t, t,
            history_observations=history_observations,
            training=training,
            params=params
        )

        return actions
        
    def actor_loss_flow_fn(self, batch, params, rng, noise=None):
        noise_rng, time_rng = jax.random.split(rng)
        actions = batch['actions']  # (B, T, action_dim)
        batch_size = actions.shape[0]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        time  = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001  # (B,)
        time_expanded = time[:, None, None]  # (B, 1, 1)

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # target velocity: points from actions → noise

        obs      = batch['observations']
        hist_obs = batch.get('history_observations')
        pred_vel = self.network.select('actor_bc_flow')(
            obs, x_t, time[:, None],
            history_observations=hist_obs,
            training=True,
            params=params,
        )
        loss = jnp.mean((pred_vel - u_t) ** 2)
        return loss, {'bc_flow_loss': loss}

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_chunk, action_dim = batch['actions'].shape
        rng, x_rng = jax.random.split(rng, 2)

        hist_obs = batch.get('history_observations')

        bc_flow_loss, _ = self.actor_loss_flow_fn(batch, grad_params, rng)
        if self.config['extraction'] == 'implicit':
            assert False # TODO: not implemented
            return bc_flow_loss, {'bc_flow_loss': bc_flow_loss}

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_chunk, action_dim))

        target_flow_actions = self.compute_flow_actions(
            batch['observations'], noises, hist_obs, seed=x_rng
        )

        actor_actions = self._sample_actions_one_step_batch(batch["observations"], hist_obs, training=True, params=grad_params, noises=noises)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(
            batch['observations'],
            actions=actor_actions_clipped,
            history_observations=hist_obs,
            training=False,
        )
        q     = jnp.mean(qs, axis=-1)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam    = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        actions = self.sample_actions(
            batch['observations'],
            history_observations=hist_obs,
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

    @staticmethod
    def _batch_shape(observations):
        """Return leading batch dims of observations (dict or array)."""
        if _is_dict_obs(observations):
            # Use any leaf; drop the per-modality trailing dims
            leaf = next(iter(observations.values()))
            # image leaf: (B, H, W, C) → batch = (B,); proprio leaf: (B, D) → batch = (B,)
            return leaf.shape[:1]
        if observations.ndim >= 3:
            # Plain image array (B, H, W, C) — only the first dim is batch.
            return observations.shape[:1]
        return observations.shape[:-1]

    @jax.jit
    def compute_flow_actions(self, observations, noises, history_observations=None, seed=None):
        """Euler integration of bc flow."""
        if _is_dict_obs(observations):
            batch_shape = next(iter(observations.values())).shape[:1]
        else:
            batch_shape = observations.shape[:-1]

        obs      = observations
        hist_obs = history_observations

        flow_steps = self.config.get('flow_steps', 10)

        dt = -1.0 / flow_steps
        if noises is None:
            actions = jax.random.normal(seed, (*batch_shape, self.config["action_chunk_size"], self.config['action_dim']))
        else:
            actions = noises

        def step(carry):
            x_t, time = carry
            t = jnp.full((*batch_shape, 1), time)
            vels = self.network.select('actor_bc_flow')(
                obs, x_t, t,
                history_observations=hist_obs,
                training=False,
            )
            return x_t + dt * vels, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2  # robust to floating-point error

        actions, _ = jax.lax.while_loop(cond, step, (actions, 1.0))

        return jnp.clip(actions, -1, 1)

    @jax.jit
    def _sample_actions_batch(self, observations, action_seed, history_observations=None):
        batch_shape = self._batch_shape(observations)
        if 'implicit' in self.config['extraction']:
            assert False # "TODO: not implemented"
            action_seed, noise_seed = jax.random.split(action_seed)
            actions = jax.random.normal(
                action_seed,
                (*batch_shape, self.config['num_samples'], self.config['action_dim']),
            )
            n_obs = jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, -len(x.shape)), self.config['num_samples'], axis=-len(x.shape)),
                observations,
            )
            n_hist_obs = (
                jax.tree_util.tree_map(
                    lambda x: jnp.repeat(jnp.expand_dims(x, -len(x.shape)), self.config['num_samples'], axis=-len(x.shape)),
                    history_observations,
                ) if history_observations is not None else None
            )

            for i in range(self.config['flow_steps']):
                t    = jnp.full((*batch_shape, self.config['num_samples'], 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(
                    n_obs, actions, t,
                    history_observations=n_hist_obs,
                )
                actions = actions + vels / self.config['flow_steps']
            actions = jnp.clip(actions, -1, 1)

            q = self.network.select('critic')(
                n_obs,
                actions=actions,
                history_observations=n_hist_obs,
            ).min(axis=-1)
            actions = actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = self._sample_actions_one_step_batch(observations, history_observations, seed=action_seed, training=False)
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
        # Detect unbatched single obs: array with ndim==1 or dict whose leaves have ndim==1/3
        # (e.g. proprio (D,) or image (H,W,C) — no leading batch dim).
        if _is_dict_obs(observations):
            # Use proprio (1-D when unbatched) to detect single obs; image is always >= 3-D.
            if 'proprio' in observations:
                is_single = observations['proprio'].ndim < 2
            else:
                # Fallback: image is (H,W,C) unbatched vs (B,H,W,C) batched.
                is_single = next(iter(observations.values())).ndim < 4
        else:
            is_single = observations.ndim == 1
        if is_single:
            observations = jax.tree_util.tree_map(lambda x: x[None], observations)
            if history_observations is not None:
                history_observations = jax.tree_util.tree_map(lambda x: x[None], history_observations)
            return self._sample_actions_batch(observations, action_seed, history_observations)[0]
        return self._sample_actions_batch(observations, action_seed, history_observations)

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_observations = example_batch['observations']
        ex_actions      = example_batch['actions']
        ex_times        = ex_actions[...,0, :1]
        action_dim      = ex_actions.shape[-1]

        ex_hist_obs = example_batch.get('history_observations')

        if config['use_attention_entropy_loss']:
            target_entropy = config['attention_entropy_target']
            ok = (
                isinstance(target_entropy, (list, tuple))
                and len(target_entropy) == config['num_layers']
                and all(isinstance(te, (list, tuple)) and len(te) == 2 for te in target_entropy)
            )
            if not ok:
                raise ValueError(
                    'attention_entropy_target must be a (num_layers, 2) nested list/tuple.'
                )

        # Build encoders (one instance per network; parameters are separate).
        enc_name = config.get('encoder', None)
        def _make_encoder():
            return encoder_modules[enc_name]() if enc_name else None

        critic_def = TransformerValue(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            num_ensembles=config['num_ensembles'],
            encoder=_make_encoder(),
            use_modality_embeddings=config['use_modality_embeddings'],
            tokenization_mode=config['tokenization_mode'],
            obs_token_dim=config['obs_token_dim'],
            action_token_dim=config['action_token_dim'],
            num_action_tokens=config['num_action_tokens'],
        )

        # actor_kwargs = dict(
        #     hidden_dim=config['hidden_dim'],
        #     num_layers=config['actor_num_layers'],
        #     num_heads=config['num_heads'],
        #     mlp_ratio=config['mlp_ratio'],
        #     dropout_rate=config['dropout_rate'],
        #     num_action_tokens=config['num_action_tokens'],
        # )

        # actor_onestep_flow_def = TransformerActorOneStep(
        #     **actor_kwargs, action_dim=action_dim, encoder=_make_encoder()
        # )

        temp_shape = (config['num_layers'], 2)
        attention_temp_init = config['attention_entropy_temperature_init']
        if config['temp_type'] == 'exp':
            attention_temp_def = LogParam(init_value=attention_temp_init, shape=temp_shape)
        elif config['temp_type'] == 'sign':
            attention_temp_def = SignedExpParam(init_value=attention_temp_init, shape=temp_shape)
        else:
            raise ValueError("temp_type must be one of: 'exp', 'sign'")

        ex_noise = ex_actions

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
                    encoder=_make_encoder(),
                    action_head=ConditionalUnet1D(action_dim, config['hidden_dim']),
                ),
                (ex_observations, ex_actions, ex_times, ex_hist_obs),
                {},
            ),
            attention_entropy_temperature=(attention_temp_def, (), {}),
        )

                    
        flow_def = TransformerActorFlow(
            hidden_dim=config['hidden_dim'],
            num_layers=config['actor_num_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout_rate=config['dropout_rate'],
            encoder=_make_encoder(),
            action_head=ConditionalUnet1D(action_dim, config['hidden_dim']),
        )
        network_info['actor_bc_flow'] = (
            flow_def,
            (ex_observations, ex_actions, ex_times, ex_hist_obs),
            {},
        )

        networks     = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def  = ModuleDict(networks)

        network_params = network_def.init(
            {'params': init_rng, 'dropout': jax.random.PRNGKey(0)},
            **network_args,
        )['params']

        def make_optimizer(lr, params):
            optimizer_chain = []
            if config['critic_grad_clip'] > 0:
                optimizer_chain.append(optax.clip_by_global_norm(config['critic_grad_clip']))
            if config['warmup_steps'] > 0:
                # Keep tiny smoke tests valid.  Full runs retain the original
                # 20k-step warmup; short runs cannot use a warmup longer than
                # the entire schedule because Optax rejects non-positive
                # decay_steps.
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
                        # Optax interprets decay_steps as the total schedule
                        # length, including warmup_steps.
                        decay_steps=max(config['train_steps'], warmup_steps + 1),
                        end_value=lr * 0.1,
                    )
            else:
                lr_schedule = lr

            if config['optimizer'] == 'adamw':
                def should_apply_wd(path, value):
                    path_str = '/'.join(str(k.key if hasattr(k, 'key') else k) for k in path).lower()
                    if 'projection' in path_str:
                        return False
                    if any(x in path_str for x in [
                        'bias', 'layernorm', 'final_ln', 'pos_embedding',
                        'cls_token', 'state_type_embed', 'action_type_embed',
                        'modules_target_', 'modules_attention_entropy_temperature',
                    ]):
                        return False
                    return value.ndim > 1
                mask = jax.tree_util.tree_map_with_path(should_apply_wd, params)
                optimizer_chain.append(optax.adamw(
                    learning_rate=lr_schedule,
                    weight_decay=config['adamw_weight_decay'],
                    mask=mask,
                ))
            elif config['optimizer'] == 'adam':
                optimizer_chain.append(optax.adam(learning_rate=lr_schedule))
            else:
                raise ValueError(f"Optimizer {config['optimizer']} not supported")
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

        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='mtql_transformer',
        action_dim=ml_collections.config_dict.placeholder(int),

        critic_lr=1e-4,
        actor_lr=1e-4,
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
        train_steps=1000000,

        use_attention_entropy_loss=True,
        temp_type='exp',
        attention_entropy_temperature_init=1.0,
        attention_entropy_temperature_min=-100.0,
        attention_entropy_temperature_max=100.0,
        attention_entropy_target=((0.0, 0.0), (0.0, 0.0)),

        hidden_dim=256,
        num_layers=2,
        actor_num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        dropout_rate=0.0,
        num_ensembles=2,
        use_modality_embeddings=True,
        tokenization_mode='per_modality',
        obs_token_dim=32,
        action_token_dim=8,

        # Set to 'proprioencoder' (or any key in encoder_modules) to encode image+proprio dict observations.
        encoder=ml_collections.config_dict.placeholder(str),

        action_chunk_size=1,
        num_action_tokens=1,  # tokens per action in the transformer (split action_dim evenly); set to action_chunk_size for 1 token per timestep
    ))
    return config
