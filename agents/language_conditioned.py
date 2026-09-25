"""Language-conditioned real-DROID agents.

This module is deliberately separate from the existing pixel-cue agents.  It
implements the same MTQL transformer, MTQL MLP, and flow-BC algorithms, but
adds one learned prompt token to each network.  The training batch is expected
to contain ``language_prompt_ids`` and, for bootstrap targets,
``next_language_prompt_ids``.  Prompt ID 0 is reserved; IDs 1--3 correspond
to the one-, two-, and three-candy instructions.

The data/entrypoint wiring is intentionally left outside this module so the
current pixel-cue training path remains unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.bc_flow_transformer import ConditionalUnet1D
from agents.mtql import TransformerBlock
from agents.mtql_mlp import ExPOQMLP
from utils.encoders import _is_dict_obs, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import LogParam, SignedExpParam


LANGUAGE_PROMPTS = {
    1: "Scoop exactly one candy.",
    2: "Scoop exactly two candies.",
    3: "Scoop exactly three candies.",
}
LANGUAGE_PROMPT_NUM_EMBEDDINGS = 4  # 0 is reserved for invalid/unset.


class LanguagePromptToken(nn.Module):
    """Map a discrete prompt ID to one transformer token."""

    hidden_dim: int
    num_embeddings: int = LANGUAGE_PROMPT_NUM_EMBEDDINGS

    @nn.compact
    def __call__(self, prompt_ids):
        prompt_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
        prompt_ids = jnp.clip(prompt_ids, 0, self.num_embeddings - 1)
        return nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.hidden_dim,
            name="language_prompt_embedding",
        )(prompt_ids)


class LanguageTransformerValue(nn.Module):
    """Transformer Q network with one learned language token."""

    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    num_ensembles: int = 2
    encoder: Any = None
    use_modality_embeddings: bool = True
    tokenization_mode: str = "per_modality"
    obs_token_dim: int = 32
    action_token_dim: int = 8
    num_action_tokens: int = 1
    language_prompt_num_embeddings: int = LANGUAGE_PROMPT_NUM_EMBEDDINGS

    @nn.compact
    def __call__(
        self,
        observations,
        actions,
        history_observations=None,
        language_prompt_ids=None,
        training: bool = False,
        return_attention_weights: bool = False,
    ):
        if language_prompt_ids is None:
            raise ValueError("Language-conditioned critic requires language_prompt_ids.")

        if self.encoder is not None:
            observations, history_observations = self.encoder(
                observations, history_observations
            )

        if observations.ndim >= 3:
            batch_shape = observations.shape[:-2]
        else:
            batch_shape = observations.shape[:-1]
        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        obs_proj = nn.Dense(self.hidden_dim, name="obs_projection")
        act_proj = nn.Dense(self.hidden_dim, name="action_projection")
        obs_down_proj = nn.Dense(self.obs_token_dim, name="obs_down_projection")
        act_down_proj = nn.Dense(self.action_token_dim, name="action_down_projection")

        if self.use_modality_embeddings:
            state_type_embed = self.param(
                "state_type_embed", nn.initializers.normal(stddev=0.02),
                (1, self.hidden_dim)
            )
            action_type_embed = self.param(
                "action_type_embed", nn.initializers.normal(stddev=0.02),
                (1, self.hidden_dim)
            )

        def project_obs(obs_slice):
            if self.tokenization_mode == "per_modality":
                tokens = obs_proj(obs_slice)[..., None, :]
                token_count = 1
            elif self.tokenization_mode == "per_dim":
                tokens = obs_proj(obs_slice[..., None])
                token_count = obs_dim
            elif self.tokenization_mode == "linear_projected_per_dim":
                latent = obs_down_proj(obs_slice)
                tokens = obs_proj(latent[..., None])
                token_count = self.obs_token_dim
            else:
                raise ValueError(f"Unsupported tokenization_mode: {self.tokenization_mode}")
            if self.use_modality_embeddings:
                shape = (1,) * len(batch_shape) + (1, self.hidden_dim)
                embed = jnp.broadcast_to(
                    jnp.reshape(state_type_embed, shape),
                    batch_shape + (token_count, self.hidden_dim),
                )
                tokens = tokens + embed
            return tokens

        def project_action(action_slice):
            nat = self.num_action_tokens
            if nat > 1:
                if action_slice.shape[-1] % nat:
                    raise ValueError("action_dim must be divisible by num_action_tokens")
                sub_dim = action_slice.shape[-1] // nat
                tokens = act_proj(
                    action_slice.reshape(*batch_shape, nat, sub_dim)
                )
                token_count = nat
            elif self.tokenization_mode == "per_modality":
                tokens = act_proj(action_slice)[..., None, :]
                token_count = 1
            elif self.tokenization_mode == "per_dim":
                tokens = act_proj(action_slice[..., None])
                token_count = action_dim
            elif self.tokenization_mode == "linear_projected_per_dim":
                latent = act_down_proj(action_slice)
                tokens = act_proj(latent[..., None])
                token_count = self.action_token_dim
            else:
                raise ValueError(f"Unsupported tokenization_mode: {self.tokenization_mode}")
            if self.use_modality_embeddings:
                shape = (1,) * len(batch_shape) + (1, self.hidden_dim)
                embed = jnp.broadcast_to(
                    jnp.reshape(action_type_embed, shape),
                    batch_shape + (token_count, self.hidden_dim),
                )
                tokens = tokens + embed
            return tokens

        prompt = LanguagePromptToken(
            self.hidden_dim, self.language_prompt_num_embeddings
        )(language_prompt_ids)[..., None, :]
        token_list = [prompt]

        if history_observations is not None:
            for k in range(history_observations.shape[-2]):
                token_list.append(project_obs(history_observations[..., k, :]))

        if observations.ndim == 3:
            for k in range(observations.shape[-2]):
                token_list.append(project_obs(observations[..., k, :]))
        else:
            token_list.append(project_obs(observations))

        for k in range(actions.shape[-2]):
            token_list.append(project_action(actions[..., k, :]))

        tokens = jnp.concatenate(token_list, axis=-2)
        seq_len = tokens.shape[-2]
        pos_embedding = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02),
            (seq_len, self.hidden_dim)
        )
        tokens = tokens + jnp.reshape(
            pos_embedding, (1,) * len(batch_shape) + pos_embedding.shape
        )

        cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=0.02),
            (1, self.hidden_dim)
        )
        cls = jnp.tile(
            jnp.reshape(cls_token, (1,) * len(batch_shape) + (1, self.hidden_dim)),
            batch_shape + (1, 1),
        )
        tokens = jnp.concatenate([cls, tokens], axis=-2)

        attention_weights = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i}",
            )
            if return_attention_weights:
                tokens, weights = block(
                    tokens, training=training, return_attention_weights=True
                )
                attention_weights.append(weights)
            else:
                tokens = block(tokens, training=training)

        tokens = nn.LayerNorm(name="final_ln")(tokens)
        cls_output = tokens[..., 0, :]
        qs = []
        for i in range(self.num_ensembles):
            x = nn.Dense(self.hidden_dim // 2, name=f"q_head_{i}_hidden")(cls_output)
            x = nn.LayerNorm(name=f"q_head_{i}_ln")(x)
            x = nn.relu(x)
            qs.append(nn.Dense(1, name=f"q_head_{i}_out")(x)[..., 0])
        q_values = jnp.stack(qs, axis=-1)
        if return_attention_weights:
            return q_values, jnp.stack(attention_weights, axis=0)
        return q_values


class LanguageTransformerActorFlow(nn.Module):
    """Flow actor with one learned language token in its context sequence."""

    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    encoder: Any = None
    action_head: Any = None
    language_prompt_num_embeddings: int = LANGUAGE_PROMPT_NUM_EMBEDDINGS

    @nn.compact
    def __call__(
        self,
        observations,
        x_t,
        t,
        history_observations=None,
        language_prompt_ids=None,
        training: bool = False,
    ):
        if language_prompt_ids is None:
            raise ValueError("Language-conditioned actor requires language_prompt_ids.")
        if self.encoder is not None:
            observations, history_observations = self.encoder(
                observations, history_observations
            )

        obs_proj = nn.Dense(self.hidden_dim, name="obs_projection")

        def one_token(value):
            return obs_proj(value)[..., None, :]

        if observations.ndim >= 3:
            batch_shape = observations.shape[:-2]
            obs_tokens = obs_proj(observations)
        else:
            batch_shape = observations.shape[:-1]
            obs_tokens = one_token(observations)

        prompt = LanguagePromptToken(
            self.hidden_dim, self.language_prompt_num_embeddings
        )(language_prompt_ids)[..., None, :]
        token_list = [prompt]
        if history_observations is not None:
            for k in range(history_observations.shape[-2]):
                token_list.append(one_token(history_observations[..., k, :]))
        token_list.append(obs_tokens)
        tokens = jnp.concatenate(token_list, axis=-2)

        pos_embedding = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02),
            (tokens.shape[-2], self.hidden_dim)
        )
        tokens = tokens + jnp.reshape(
            pos_embedding, (1,) * len(batch_shape) + pos_embedding.shape
        )
        cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=0.02),
            (1, self.hidden_dim)
        )
        cls = jnp.tile(
            jnp.reshape(cls_token, (1,) * len(batch_shape) + (1, self.hidden_dim)),
            batch_shape + (1, 1),
        )
        tokens = jnp.concatenate([cls, tokens], axis=-2)
        for i in range(self.num_layers):
            tokens = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i}",
            )(tokens, training=training)
        cls_out = nn.LayerNorm(name="final_ln")(tokens)[..., 0, :]
        return self.action_head(sample=x_t, timestep=t, global_cond=cls_out)


class LanguageMLPValue(nn.Module):
    """MLP critic using the same tokens as the language transformer critic."""

    token_hidden_dim: int = 256
    mlp_hidden_dim: int = 67
    mlp_num_layers: int = 4
    num_ensembles: int = 2
    encoder: Any = None
    use_modality_embeddings: bool = True
    tokenization_mode: str = "per_modality"
    num_action_tokens: int = 1
    language_prompt_num_embeddings: int = LANGUAGE_PROMPT_NUM_EMBEDDINGS

    @nn.compact
    def __call__(
        self,
        observations,
        actions,
        history_observations=None,
        language_prompt_ids=None,
        training: bool = False,
    ):
        del training
        if language_prompt_ids is None:
            raise ValueError("Language-conditioned MLP critic requires language_prompt_ids.")
        if self.tokenization_mode != "per_modality":
            raise ValueError("LanguageMLPValue requires tokenization_mode='per_modality'.")
        if self.encoder is not None:
            observations, history_observations = self.encoder(
                observations, history_observations
            )
        batch_shape = observations.shape[:-2] if observations.ndim >= 3 else observations.shape[:-1]

        obs_projection = nn.Dense(self.token_hidden_dim, name="obs_projection")
        action_projection = nn.Dense(self.token_hidden_dim, name="action_projection")
        if self.use_modality_embeddings:
            state_type_embed = self.param(
                "state_type_embed", nn.initializers.normal(stddev=0.02),
                (1, self.token_hidden_dim)
            )
            action_type_embed = self.param(
                "action_type_embed", nn.initializers.normal(stddev=0.02),
                (1, self.token_hidden_dim)
            )

        def project_obs(value):
            tokens = obs_projection(value)[..., None, :]
            if self.use_modality_embeddings:
                shape = (1,) * len(batch_shape) + (1, self.token_hidden_dim)
                tokens = tokens + jnp.reshape(state_type_embed, shape)
            return tokens

        def project_action(value):
            if self.num_action_tokens > 1:
                if value.shape[-1] % self.num_action_tokens:
                    raise ValueError("action_dim must be divisible by num_action_tokens")
                value = value.reshape(
                    *batch_shape, self.num_action_tokens,
                    value.shape[-1] // self.num_action_tokens,
                )
                tokens = action_projection(value)
            else:
                tokens = action_projection(value)[..., None, :]
            if self.use_modality_embeddings:
                shape = (1,) * len(batch_shape) + (1, self.token_hidden_dim)
                tokens = tokens + jnp.reshape(action_type_embed, shape)
            return tokens

        tokens = [
            LanguagePromptToken(
                self.token_hidden_dim, self.language_prompt_num_embeddings
            )(language_prompt_ids)[..., None, :]
        ]
        if history_observations is not None:
            for k in range(history_observations.shape[-2]):
                tokens.append(project_obs(history_observations[..., k, :]))
        if observations.ndim == 3:
            for k in range(observations.shape[-2]):
                tokens.append(project_obs(observations[..., k, :]))
        else:
            tokens.append(project_obs(observations))
        for k in range(actions.shape[-2]):
            tokens.append(project_action(actions[..., k, :]))

        tokens = jnp.concatenate(tokens, axis=-2)
        pos_embedding = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02),
            (tokens.shape[-2], self.token_hidden_dim)
        )
        tokens = tokens + jnp.reshape(
            pos_embedding, (1,) * len(batch_shape) + pos_embedding.shape
        )
        cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=0.02),
            (1, self.token_hidden_dim)
        )
        cls = jnp.tile(
            jnp.reshape(cls_token, (1,) * len(batch_shape) + (1, self.token_hidden_dim)),
            batch_shape + (1, 1),
        )
        mlp_input = jnp.concatenate([cls, tokens], axis=-2).reshape(*batch_shape, -1)
        values = [
            ExPOQMLP(
                hidden_dim=self.mlp_hidden_dim,
                num_layers=self.mlp_num_layers,
                name=f"q_mlp_{i}",
            )(mlp_input)
            for i in range(self.num_ensembles)
        ]
        return jnp.stack(values, axis=-1)


def _prompt(batch, key="language_prompt_ids"):
    value = batch.get(key)
    if value is None:
        raise KeyError(
            f"Language-conditioned agents require batch['{key}']; "
            "the language-only data entrypoint must add it."
        )
    return jnp.asarray(value, dtype=jnp.int32)


def _make_encoder(config):
    name = config.get("encoder", None)
    return encoder_modules[name]() if name else None


def _make_optimizer(config, lr, params, grad_clip_key="critic_grad_clip"):
    chain = []
    grad_clip = config.get(grad_clip_key, 0.0)
    if grad_clip > 0:
        chain.append(optax.clip_by_global_norm(grad_clip))
    warmup_steps = min(config.get("warmup_steps", 0), max(config.get("train_steps", 0) - 1, 0))
    if warmup_steps == 0:
        schedule = lr
    else:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.01,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=max(config.get("train_steps", 0), warmup_steps + 1),
            end_value=lr * 0.1,
        )
    if config.get("optimizer", "adamw") == "adamw":
        def apply_wd(path, value):
            path_str = "/".join(str(k.key if hasattr(k, "key") else k) for k in path).lower()
            if "projection" in path_str:
                return False
            if any(name in path_str for name in (
                "bias", "layernorm", "final_ln", "pos_embedding", "cls_token",
                "state_type_embed", "action_type_embed", "language_prompt_embedding",
                "modules_target_", "modules_attention_entropy_temperature",
            )):
                return False
            return value.ndim > 1
        mask = jax.tree_util.tree_map_with_path(apply_wd, params)
        chain.append(optax.adamw(
            learning_rate=schedule,
            weight_decay=config.get("adamw_weight_decay", 0.01),
            mask=mask,
        ))
    else:
        chain.append(optax.adam(learning_rate=schedule))
    return optax.chain(*chain)


class LanguageMTQLTransformerAgent(flax.struct.PyTreeNode):
    """MTQL transformer agent conditioned on a discrete language prompt."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng, step=None):
        del step
        rng, sample_rng, dropout_rng = jax.random.split(rng, 3)
        prompt = _prompt(batch)
        next_prompt = _prompt(batch, "next_language_prompt_ids")
        next_actions = self.sample_actions(
            batch["next_observations"],
            history_observations=batch.get("next_history_observations"),
            language_prompt_ids=next_prompt,
            seed=sample_rng,
        )
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=jnp.clip(next_actions, -1, 1),
            history_observations=batch.get("next_history_observations"),
            language_prompt_ids=next_prompt,
            training=False,
        )
        next_q = next_qs.min(axis=-1) if self.config["q_agg"] == "min" else next_qs.mean(axis=-1)
        target_q = batch["rewards"] + self.config["discount"] ** self.config["action_chunk_size"] * batch["masks"] * next_q
        q, attn_weights = self.network.select("critic")(
            batch["observations"], actions=batch["actions"],
            history_observations=batch.get("history_observations"),
            language_prompt_ids=prompt, params=grad_params, training=True,
            rngs={"dropout": dropout_rng}, return_attention_weights=True,
        )
        q_loss = jnp.square(q - target_q[:, None]).mean()
        entropy = -jnp.sum(attn_weights * jnp.log(jnp.clip(attn_weights, a_min=1e-9)), axis=-1)
        cls_mean = entropy[..., 0].mean(axis=(1, 2))
        other_mean = entropy[..., 1:].mean(axis=(1, 2, 3))
        entropy_mean = jnp.stack([cls_mean, other_mean], axis=-1)
        if bool(self.config["use_attention_entropy_loss"]):
            temperature = self.network.select("attention_entropy_temperature")(params=grad_params)
            clipped = jax.lax.stop_gradient(jnp.clip(
                temperature,
                self.config["attention_entropy_temperature_min"],
                self.config["attention_entropy_temperature_max"],
            ))
            entropy_loss = (-clipped * entropy_mean).mean()
            target = jnp.asarray(self.config["attention_entropy_target"], dtype=entropy_mean.dtype)
            temp_loss = (temperature * (jax.lax.stop_gradient(entropy_mean) - target)).mean()
        else:
            entropy_loss = jnp.asarray(0.0, dtype=q.dtype)
            temp_loss = jnp.asarray(0.0, dtype=q.dtype)
        loss = q_loss + entropy_loss + temp_loss
        info = {
            "critic_loss": loss, "q_loss": q_loss,
            "attention_entropy_loss": entropy_loss, "temperature_loss": temp_loss,
            "q_mean": q.mean(), "q_max": q.max(), "q_min": q.min(),
            "rewards_mean": batch["rewards"].mean(), "target_q_mean": target_q.mean(),
        }
        return loss, info

    def _sample_actions_one_step_batch(
        self, observations, history_observations, language_prompt_ids,
        training=False, params=None, seed=None, noises=None, temperature=1.0,
    ):
        del temperature
        batch_shape = (
            next(iter(observations.values())).shape[:1]
            if _is_dict_obs(observations) else observations.shape[:-1]
        )
        if noises is None:
            noises = jax.random.normal(seed, (*batch_shape, self.config["action_chunk_size"], self.config["action_dim"]))
        t = jnp.zeros((*batch_shape, 1))
        return self.network.select("actor_onestep_flow")(
            observations, noises, t, history_observations=history_observations,
            language_prompt_ids=language_prompt_ids, training=training, params=params,
        )

    def actor_loss_flow_fn(self, batch, params, rng, noise=None):
        noise_rng, time_rng = jax.random.split(rng)
        actions = batch["actions"]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (actions.shape[0],)) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, time[:, None],
            history_observations=batch.get("history_observations"),
            language_prompt_ids=_prompt(batch), training=True, params=params,
        )
        loss = jnp.mean((pred - (noise - actions)) ** 2)
        return loss, {"bc_flow_loss": loss}

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_chunk, action_dim = batch["actions"].shape
        rng, x_rng = jax.random.split(rng, 2)
        hist = batch.get("history_observations")
        prompt = _prompt(batch)
        bc_loss, _ = self.actor_loss_flow_fn(batch, grad_params, rng)
        noises = jax.random.normal(rng, (batch_size, action_chunk, action_dim))
        target = self.compute_flow_actions(
            batch["observations"], noises, hist, language_prompt_ids=prompt, seed=x_rng
        )
        one_step = self._sample_actions_one_step_batch(
            batch["observations"], hist, prompt, training=True,
            params=grad_params, noises=noises,
        )
        distill_loss = jnp.mean((one_step - target) ** 2)
        qs = self.network.select("critic")(
            batch["observations"], actions=jnp.clip(one_step, -1, 1),
            history_observations=hist, language_prompt_ids=prompt, training=False,
        )
        q = qs.mean(axis=-1)
        q_loss = -q.mean()
        if self.config["normalize_q_loss"]:
            q_loss = jax.lax.stop_gradient(1 / jnp.abs(q).mean()) * q_loss
        total = bc_loss + self.config["alpha"] * distill_loss + q_loss
        sampled = self.sample_actions(
            batch["observations"], history_observations=hist,
            language_prompt_ids=prompt, seed=rng,
        )
        return total, {
            "actor_loss": total, "bc_flow_loss": bc_loss,
            "distill_loss": distill_loss, "q_loss": q_loss,
            "q": q.mean(), "mse": jnp.mean((sampled - batch["actions"]) ** 2),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=None):
        rng = self.rng if rng is None else rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, step)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {f"critic/{k}": v for k, v in critic_info.items()}
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        network.params[f"modules_target_{module_name}"] = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )

    @jax.jit
    def update(self, batch, step=None):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda params: self.total_loss(batch, params, rng=rng, step=step)
        )
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_flow_actions(self, observations, noises, history_observations=None, language_prompt_ids=None, seed=None):
        del seed
        batch_shape = (
            next(iter(observations.values())).shape[:1]
            if _is_dict_obs(observations) else observations.shape[:-1]
        )
        flow_steps = self.config.get("flow_steps", 10)
        dt = -1.0 / flow_steps
        actions = noises

        def step(carry):
            x_t, time = carry
            t = jnp.full((*batch_shape, 1), time)
            vels = self.network.select("actor_bc_flow")(
                observations, x_t, t, history_observations=history_observations,
                language_prompt_ids=language_prompt_ids, training=False,
            )
            return x_t + dt * vels, time + dt

        actions, _ = jax.lax.while_loop(
            lambda carry: carry[1] >= -dt / 2, step, (actions, 1.0)
        )
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def _sample_actions_batch(self, observations, action_seed, history_observations=None, language_prompt_ids=None):
        actions = self._sample_actions_one_step_batch(
            observations, history_observations, language_prompt_ids, seed=action_seed,
            training=False,
        )
        return jnp.clip(actions, -1, 1)

    def sample_actions(self, observations, history_observations=None, language_prompt_ids=None, seed=None, temperature=1.0):
        del temperature
        if language_prompt_ids is None:
            raise ValueError("sample_actions requires language_prompt_ids")
        seed = seed if seed is not None else jax.random.PRNGKey(0)
        if _is_dict_obs(observations):
            is_single = observations.get("proprio", next(iter(observations.values()))).ndim < 2
        else:
            is_single = observations.ndim == 1
        if is_single:
            observations = jax.tree_util.tree_map(lambda x: x[None], observations)
            if history_observations is not None:
                history_observations = jax.tree_util.tree_map(lambda x: x[None], history_observations)
            language_prompt_ids = jnp.asarray(language_prompt_ids)[None]
            return self._sample_actions_batch(observations, seed, history_observations, language_prompt_ids)[0]
        return self._sample_actions_batch(observations, seed, history_observations, language_prompt_ids)

    @classmethod
    def create(cls, seed, example_batch, config):
        return _create_mtql_agent(cls, seed, example_batch, config, critic_cls=LanguageTransformerValue)


class LanguageMTQLMLPAgent(LanguageMTQLTransformerAgent):
    """MTQL actor with a language-conditioned MLP Q critic."""

    def critic_loss(self, batch, grad_params, rng, step=None):
        del step
        _, sample_rng, dropout_rng = jax.random.split(rng, 3)
        prompt = _prompt(batch)
        next_prompt = _prompt(batch, "next_language_prompt_ids")
        next_actions = self.sample_actions(
            batch["next_observations"], batch.get("next_history_observations"), next_prompt, sample_rng
        )
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=jnp.clip(next_actions, -1, 1),
            history_observations=batch.get("next_history_observations"),
            language_prompt_ids=next_prompt,
        )
        next_q = next_qs.min(axis=-1) if self.config["q_agg"] == "min" else next_qs.mean(axis=-1)
        target_q = batch["rewards"] + self.config["discount"] ** self.config["action_chunk_size"] * batch["masks"] * next_q
        q = self.network.select("critic")(
            batch["observations"], actions=batch["actions"],
            history_observations=batch.get("history_observations"),
            language_prompt_ids=prompt, params=grad_params, training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = jnp.square(q - target_q[:, None]).mean()
        return loss, {
            "critic_loss": loss, "q_loss": loss, "q_mean": q.mean(),
            "q_max": q.max(), "q_min": q.min(),
            "rewards_mean": batch["rewards"].mean(), "target_q_mean": target_q.mean(),
        }

    @classmethod
    def create(cls, seed, example_batch, config):
        return _create_mtql_agent(cls, seed, example_batch, config, critic_cls=LanguageMLPValue)


def _create_mtql_agent(cls, seed, example_batch, config, critic_cls):
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    ex_obs = example_batch["observations"]
    ex_actions = example_batch["actions"]
    ex_times = ex_actions[..., 0, :1]
    ex_hist = example_batch.get("history_observations")
    ex_prompt = _prompt(example_batch)
    action_dim = ex_actions.shape[-1]
    enc_name = config.get("encoder", None)
    make_encoder = lambda: encoder_modules[enc_name]() if enc_name else None

    if critic_cls is LanguageTransformerValue:
        critic = critic_cls(
            hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
            num_heads=config["num_heads"], mlp_ratio=config["mlp_ratio"],
            dropout_rate=config["dropout_rate"], num_ensembles=config["num_ensembles"],
            encoder=make_encoder(), use_modality_embeddings=config["use_modality_embeddings"],
            tokenization_mode=config["tokenization_mode"], obs_token_dim=config["obs_token_dim"],
            action_token_dim=config["action_token_dim"], num_action_tokens=config["num_action_tokens"],
        )
    else:
        critic = critic_cls(
            token_hidden_dim=config["hidden_dim"],
            mlp_hidden_dim=config["critic_mlp_hidden_dim"],
            mlp_num_layers=config["critic_mlp_num_layers"],
            num_ensembles=config["num_ensembles"], encoder=make_encoder(),
            use_modality_embeddings=config["use_modality_embeddings"],
            tokenization_mode=config["tokenization_mode"],
            num_action_tokens=config["num_action_tokens"],
        )

    def actor():
        return LanguageTransformerActorFlow(
            hidden_dim=config["hidden_dim"], num_layers=config["actor_num_layers"],
            num_heads=config["num_heads"], mlp_ratio=config["mlp_ratio"],
            dropout_rate=config["dropout_rate"], encoder=make_encoder(),
            action_head=ConditionalUnet1D(action_dim, config["hidden_dim"]),
        )

    network_info = {
        "critic": (critic, (ex_obs, ex_actions, ex_hist, ex_prompt), {}),
        "target_critic": (copy.deepcopy(critic), (ex_obs, ex_actions, ex_hist, ex_prompt), {}),
        "actor_onestep_flow": (actor(), (ex_obs, ex_actions, ex_times, ex_hist, ex_prompt), {}),
        "actor_bc_flow": (actor(), (ex_obs, ex_actions, ex_times, ex_hist, ex_prompt), {}),
    }
    if critic_cls is LanguageTransformerValue:
        temp_shape = (config["num_layers"], 2)
        temp_init = config["attention_entropy_temperature_init"]
        temp = (
            LogParam(init_value=temp_init, shape=temp_shape)
            if config["temp_type"] == "exp"
            else SignedExpParam(init_value=temp_init, shape=temp_shape)
        )
        network_info["attention_entropy_temperature"] = (temp, (), {})
    network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
    network_args = {k: v[1] for k, v in network_info.items()}
    params = network_def.init(
        {"params": init_rng, "dropout": jax.random.PRNGKey(0)}, **network_args
    )["params"]
    critic_opt = _make_optimizer(config, config["critic_lr"], params)
    actor_opt = _make_optimizer(config, config["actor_lr"], params, "critic_grad_clip")

    def label(path, value):
        del value
        path_str = "/".join(str(k.key if hasattr(k, "key") else k) for k in path).lower()
        return "critic" if "modules_critic" in path_str or "modules_attention_entropy_temperature" in path_str else "actor"

    labels = jax.tree_util.tree_map_with_path(label, params)
    tx = optax.multi_transform({"critic": critic_opt, "actor": actor_opt}, labels)
    network = TrainState.create(network_def, params, tx=tx)
    network.params["modules_target_critic"] = copy.deepcopy(network.params["modules_critic"])
    config["action_dim"] = action_dim
    return cls(rng, network=network, config=flax.core.FrozenDict(**config))


class LanguageNewBCFlowTransformerAgent(flax.struct.PyTreeNode):
    """Pure flow-matching BC agent conditioned on the language prompt."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss_flow_fn(self, batch, params, rng, noise=None):
        noise_rng, time_rng = jax.random.split(rng)
        actions = batch["actions"]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (actions.shape[0],)) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, time[:, None],
            history_observations=batch.get("history_observations"),
            language_prompt_ids=_prompt(batch), training=True, params=params,
        )
        per_sample = jnp.mean((pred - (noise - actions)) ** 2, axis=(1, 2))
        mask = batch.get("actor_mask", jnp.ones((actions.shape[0],))).astype(per_sample.dtype)
        loss = jnp.sum(per_sample * mask) / jnp.maximum(mask.sum(), 1.0)
        return loss, {"bc_flow_loss": loss, "actor_batch_fraction": mask.mean()}

    @jax.jit
    def update(self, batch, step=None):
        del step
        new_rng, update_rng = jax.random.split(self.rng)
        loss_rng, _ = jax.random.split(update_rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda params: self.actor_loss_flow_fn(batch, params, loss_rng)
        )
        return self.replace(rng=new_rng, network=new_network), info

    @staticmethod
    def _batch_shape(observations):
        if _is_dict_obs(observations):
            return next(iter(observations.values())).shape[:1]
        return observations.shape[:1] if observations.ndim >= 3 else observations.shape[:-1]

    @jax.jit
    def compute_flow_actions(self, observations, noises=None, history_observations=None, language_prompt_ids=None, seed=None):
        batch_shape = self._batch_shape(observations)
        flow_steps = self.config.get("flow_steps", 10)
        dt = -1.0 / flow_steps
        actions = noises if noises is not None else jax.random.normal(seed, (*batch_shape, self.config["action_chunk_size"], self.config["action_dim"]))
        def step(carry):
            x_t, time = carry
            t = jnp.full((*batch_shape, 1), time)
            vels = self.network.select("actor_bc_flow")(
                observations, x_t, t, history_observations=history_observations,
                language_prompt_ids=language_prompt_ids, training=False,
            )
            return x_t + dt * vels, time + dt
        actions, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (actions, 1.0))
        return jnp.clip(actions, -1, 1)

    def sample_actions(self, observations, history_observations=None, language_prompt_ids=None, seed=None, noises=None, temperature=1.0):
        del temperature
        if language_prompt_ids is None:
            raise ValueError("sample_actions requires language_prompt_ids")
        seed = seed if seed is not None else jax.random.PRNGKey(0)
        is_single = (
            observations.get("proprio", next(iter(observations.values()))).ndim < 2
            if _is_dict_obs(observations) else observations.ndim == 1
        )
        if is_single:
            observations = jax.tree_util.tree_map(lambda x: x[None], observations)
            if history_observations is not None:
                history_observations = jax.tree_util.tree_map(lambda x: x[None], history_observations)
            language_prompt_ids = jnp.asarray(language_prompt_ids)[None]
            if noises is not None:
                noises = noises[None]
            return self.compute_flow_actions(observations, noises, history_observations, language_prompt_ids, seed)[0]
        return self.compute_flow_actions(observations, noises, history_observations, language_prompt_ids, seed)

    @classmethod
    def create(cls, seed, example_batch, config, optimizer=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_obs = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_times = ex_actions[..., 0, :1]
        ex_hist = example_batch.get("history_observations")
        ex_prompt = _prompt(example_batch)
        action_dim = ex_actions.shape[-1]
        enc_name = config.get("encoder", None)
        encoder = encoder_modules[enc_name]() if enc_name else None
        actor = LanguageTransformerActorFlow(
            hidden_dim=config["hidden_dim"], num_layers=config["actor_num_layers"],
            num_heads=config["num_heads"], mlp_ratio=config["mlp_ratio"],
            dropout_rate=config["dropout_rate"], encoder=encoder,
            action_head=ConditionalUnet1D(action_dim, config["hidden_dim"]),
        )
        network_def = ModuleDict({"actor_bc_flow": actor})
        params = network_def.init(
            {"params": init_rng, "dropout": jax.random.PRNGKey(0)},
            actor_bc_flow=(ex_obs, ex_actions, ex_times, ex_hist, ex_prompt),
        )["params"]
        if optimizer is None:
            optimizer = _make_optimizer(config, config["actor_lr"], params, "actor_grad_clip")
        network = TrainState.create(network_def, params, tx=optimizer)
        config["action_dim"] = action_dim
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def _language_config(base_config, agent_name):
    config = base_config
    config.agent_name = agent_name
    config.language_prompt_num_embeddings = LANGUAGE_PROMPT_NUM_EMBEDDINGS
    return config


def transformer_language_config():
    from agents.mtql_transformer_real import get_config
    return _language_config(get_config(), "mtql_transformer_language_real")


def mlp_language_config():
    from agents.mtql_mlp_real import get_config
    return _language_config(get_config(), "mtql_mlp_language_real")


def bc_language_config():
    from agents.new_bc_flow_transformer_real import get_config
    return _language_config(get_config(), "new_bc_flow_transformer_language_real")
