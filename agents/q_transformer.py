import copy
import math

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

def _set_top_level(params, key, value):
    if isinstance(params, flax.core.frozen_dict.FrozenDict):
        d = flax.core.unfreeze(params)
        d[key] = value
        return flax.core.freeze(d)
    d = dict(params)
    d[key] = value
    return d

def _normalize_actions(actions, action_low, action_high):
    action_low = jnp.asarray(action_low, dtype=jnp.float32)
    action_high = jnp.asarray(action_high, dtype=jnp.float32)
    actions = actions.astype(jnp.float32)
    u = (actions - action_low) / (action_high - action_low)
    u = jnp.clip(u, 0.0, 1.0)
    return u * 2.0 - 1.0


def _unnormalize_actions(actions, action_low, action_high):
    action_low = jnp.asarray(action_low, dtype=jnp.float32)
    action_high = jnp.asarray(action_high, dtype=jnp.float32)
    a = jnp.clip(actions.astype(jnp.float32), -1.0, 1.0)
    u = (a + 1.0) * 0.5
    return action_low + u * (action_high - action_low)


def _actions_to_bins(actions, num_bins, action_low, action_high):
    a = _normalize_actions(actions, action_low, action_high)
    u = (jnp.clip(a, -1.0, 1.0) + 1.0) * 0.5
    idx = jnp.floor(u * num_bins).astype(jnp.int32)
    return jnp.clip(idx, 0, num_bins - 1)


def _bins_to_actions(bins, num_bins, action_low, action_high):
    x = (bins.astype(jnp.float32) + 0.5) / float(num_bins)
    a = jnp.clip(x * 2.0 - 1.0, -1.0, 1.0)
    return _unnormalize_actions(a, action_low, action_high)


class _TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_ratio: int
    layer_norm: bool
    use_qk_norm: bool = False
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask, training: bool = False):
        x = x.astype(self.dtype)
        h = (
            nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
            if self.layer_norm
            else x
        )

        attn_kwargs = dict(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype,
        )

        if self.dtype == jnp.bfloat16 or self.use_qk_norm:
            def attention_fn(
                query,
                key,
                value,
                bias=None,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=False,
                dtype=None,
                precision=None,
            ):
                if self.use_qk_norm:
                    query = nn.RMSNorm(
                        name="q_norm",
                        epsilon=1e-6,
                        dtype=self.dtype,
                        param_dtype=jnp.float32,
                        use_scale=True,
                        reduction_axes=-1,
                        feature_axes=-1,
                    )(query)
                    key = nn.RMSNorm(
                        name="k_norm",
                        epsilon=1e-6,
                        dtype=self.dtype,
                        param_dtype=jnp.float32,
                        use_scale=True,
                        reduction_axes=-1,
                        feature_axes=-1,
                    )(key)
                if self.dtype == jnp.bfloat16:
                    q = query.astype(jnp.float32)
                    k = key.astype(jnp.float32)
                    attn_w = nn.attention.dot_product_attention_weights(
                        q,
                        k,
                        bias=bias,
                        mask=mask,
                        broadcast_dropout=broadcast_dropout,
                        dropout_rng=dropout_rng,
                        dropout_rate=dropout_rate,
                        deterministic=deterministic,
                        dtype=jnp.float32,
                        precision=precision,
                    )
                else:
                    query, key, value = nn.attention.promote_dtype(query, key, value, dtype=dtype)
                    attn_w = nn.attention.dot_product_attention_weights(
                        query,
                        key,
                        bias=bias,
                        mask=mask,
                        broadcast_dropout=broadcast_dropout,
                        dropout_rng=dropout_rng,
                        dropout_rate=dropout_rate,
                        deterministic=deterministic,
                        dtype=dtype,
                        precision=precision,
                    )
                return jnp.einsum(
                    "...hqk,...khd->...qhd",
                    attn_w.astype(value.dtype),
                    value,
                    precision=precision,
                )

            attn_kwargs["attention_fn"] = attention_fn

        h = nn.SelfAttention(**attn_kwargs)(h, mask=mask)
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not training)
        x = x + h
        h = (
            nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
            if self.layer_norm
            else x
        )
        h = nn.Dense(self.hidden_dim * self.mlp_ratio, dtype=self.dtype)(h)
        h = nn.gelu(h)
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not training)
        h = nn.Dense(self.hidden_dim, dtype=self.dtype)(h)
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not training)
        x = x + h
        return x


class _TransformerBlockWithAttentionWeights(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_ratio: int
    layer_norm: bool
    use_qk_norm: bool = False
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask, training: bool = False):
        x = x.astype(self.dtype)
        h = (
            nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
            if self.layer_norm
            else x
        )

        def attention_fn(
            query,
            key,
            value,
            bias=None,
            mask=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.0,
            deterministic=False,
            dtype=None,
            precision=None,
        ):
            if self.use_qk_norm:
                query = nn.RMSNorm(
                    name="q_norm",
                    epsilon=1e-6,
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    use_scale=True,
                    reduction_axes=-1,
                    feature_axes=-1,
                )(query)
                key = nn.RMSNorm(
                    name="k_norm",
                    epsilon=1e-6,
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    use_scale=True,
                    reduction_axes=-1,
                    feature_axes=-1,
                )(key)
            if self.dtype == jnp.bfloat16:
                q = query.astype(jnp.float32)
                k = key.astype(jnp.float32)
                attn_weights = nn.attention.dot_product_attention_weights(
                    q,
                    k,
                    bias=bias,
                    mask=mask,
                    broadcast_dropout=broadcast_dropout,
                    dropout_rng=dropout_rng,
                    dropout_rate=dropout_rate,
                    deterministic=deterministic,
                    dtype=jnp.float32,
                    precision=precision,
                )
            else:
                query, key, value = nn.attention.promote_dtype(query, key, value, dtype=dtype)
                attn_weights = nn.attention.dot_product_attention_weights(
                    query,
                    key,
                    bias=bias,
                    mask=mask,
                    broadcast_dropout=broadcast_dropout,
                    dropout_rng=dropout_rng,
                    dropout_rate=dropout_rate,
                    deterministic=deterministic,
                    dtype=dtype,
                    precision=precision,
                )
            self.sow("intermediates", "attention_weights", attn_weights.astype(jnp.float32))
            return jnp.einsum(
                "...hqk,...khd->...qhd",
                attn_weights.astype(value.dtype),
                value,
                precision=precision,
            )

        h = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=0.0,
            deterministic=True,
            attention_fn=attention_fn,
            dtype=self.dtype,
        )(h, mask=mask)
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not training)
        x = x + h
        h = (
            nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
            if self.layer_norm
            else x
        )
        h = nn.Dense(self.hidden_dim * self.mlp_ratio, dtype=self.dtype)(h)
        h = nn.gelu(h)
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not training)
        h = nn.Dense(self.hidden_dim, dtype=self.dtype)(h)
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not training)
        x = x + h
        return x


class _QTransformerHead(nn.Module):
    action_dim: int
    num_bins: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm: bool
    init_logit: float
    use_qk_norm: bool = False
    dropout_rate: float = 0.0
    final_layer_norm: bool = False
    use_state_tokens_per_dim: bool = False
    use_modality_embeddings: bool = False
    dtype: jnp.dtype = jnp.float32
    block_cls = _TransformerBlock

    def _embed_prev_actions(self, action_in_embed, action_bins_prev):
        tables = action_in_embed
        idxs = jnp.swapaxes(action_bins_prev, 0, 1)

        def take_one(table, idx):
            return table[idx]

        out = jax.vmap(take_one, in_axes=(0, 0), out_axes=0)(tables, idxs)
        return jnp.swapaxes(out, 0, 1)

    @nn.compact
    def __call__(self, state_embed, action_bins, training: bool = False):
        b = state_embed.shape[0]
        if self.use_state_tokens_per_dim:
            assert state_embed.ndim == 3
            state_tokens = state_embed.astype(self.dtype)
        else:
            assert state_embed.ndim == 2
            state_tokens = state_embed.astype(self.dtype)[:, None, :]
        state_len = state_tokens.shape[1]
        seq_len = self.action_dim + state_len

        start_token = self.param(
            "start_token",
            nn.initializers.normal(stddev=0.02),
            (self.hidden_dim,),
        )
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (seq_len, self.hidden_dim),
        )
        action_out_embed = self.param(
            "action_out_embed",
            nn.initializers.normal(stddev=0.02),
            (self.action_dim, self.num_bins, self.hidden_dim),
        )
        action_out_bias = self.param(
            "action_out_bias",
            nn.initializers.constant(self.init_logit),
            (self.action_dim, self.num_bins),
        )

        pos = pos_embed.astype(self.dtype)
        start = start_token.astype(self.dtype)
        state_tok = state_tokens + pos[:state_len]
        start_tok = jnp.broadcast_to(start, (b, self.hidden_dim)) + pos[state_len]

        if self.action_dim > 1:
            action_in_embed = self.param(
                "action_in_embed",
                nn.initializers.normal(stddev=0.02),
                (self.action_dim - 1, self.num_bins, self.hidden_dim),
            )
            prev_bins = action_bins[:, :-1]
            prev_tok = self._embed_prev_actions(action_in_embed, prev_bins).astype(self.dtype)
            prev_tok = prev_tok + pos[state_len + 1 :]
        else:
            prev_tok = jnp.zeros((b, 0, self.hidden_dim), dtype=self.dtype)

        if self.use_modality_embeddings:
            state_type_embed = self.param(
                "state_type_embed",
                nn.initializers.normal(stddev=0.02),
                (self.hidden_dim,),
            )
            action_type_embed = self.param(
                "action_type_embed",
                nn.initializers.normal(stddev=0.02),
                (self.hidden_dim,),
            )
            state_tok = state_tok + state_type_embed.astype(self.dtype)
            start_tok = start_tok + action_type_embed.astype(self.dtype)
            prev_tok = prev_tok + action_type_embed.astype(self.dtype)

        tokens = jnp.concatenate([state_tok, start_tok[:, None, :], prev_tok], axis=1)

        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn_mask = causal[None, None, :, :]

        x = tokens
        for i in range(self.depth):
            x = self.block_cls(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm=self.layer_norm,
                use_qk_norm=self.use_qk_norm,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name=f"block_{i}",
            )(x, attn_mask, training=training)

        if self.final_layer_norm:
            x = nn.LayerNorm(dtype=jnp.float32, name="final_ln")(x.astype(jnp.float32)).astype(self.dtype)

        h = x[:, state_len:, :]
        q_logits = jnp.einsum("bde,dne->bdn", h, action_out_embed) + action_out_bias[None, :, :]
        return q_logits


class _QTransformerHeadWithAttentionWeights(_QTransformerHead):
    block_cls = _TransformerBlockWithAttentionWeights


class _AttentionEntropyTemperature(nn.Module):
    num_layers: int
    init_value: float

    @nn.compact
    def __call__(self):
        assert self.init_value > 0.0
        raw = self.param(
            "raw",
            nn.initializers.constant(math.log(self.init_value)),
            (self.num_layers,),
        )
        return jnp.exp(raw)


class QTransformerCritic(nn.Module):
    action_dim: int
    num_bins: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm: bool
    encoder: nn.Module = None
    init_logit: float = 0.0
    use_qk_norm: bool = False
    dropout_rate: float = 0.0
    final_layer_norm: bool = False
    use_state_tokens_per_dim: bool = False
    use_modality_embeddings: bool = False
    dtype: jnp.dtype = jnp.float32
    head_cls = _QTransformerHead

    def setup(self):
        self.obs_proj = nn.Dense(self.hidden_dim, dtype=self.dtype)
        self.head = self.head_cls(
            action_dim=self.action_dim,
            num_bins=self.num_bins,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            layer_norm=self.layer_norm,
            init_logit=self.init_logit,
            use_qk_norm=self.use_qk_norm,
            dropout_rate=self.dropout_rate,
            final_layer_norm=self.final_layer_norm,
            use_state_tokens_per_dim=self.use_state_tokens_per_dim,
            use_modality_embeddings=self.use_modality_embeddings,
            dtype=self.dtype,
        )

    def __call__(self, observations, action_bins=None, training: bool = False):
        if self.encoder is not None:
            obs = self.encoder(observations, train=training)
        else:
            obs = observations

        obs = obs.astype(self.dtype)
        if self.use_state_tokens_per_dim:
            obs_tokens = self.obs_proj(obs[..., None])
            batch_axes = obs_tokens.shape[:-2]
            obs_embed_flat = obs_tokens.reshape((-1, obs_tokens.shape[-2], self.hidden_dim))
        else:
            obs_embed = self.obs_proj(obs)
            batch_axes = obs_embed.shape[:-1]
            obs_embed_flat = obs_embed.reshape((-1, self.hidden_dim))

        if action_bins is None:
            action_bins = jnp.zeros((*batch_axes, self.action_dim), dtype=jnp.int32)
        action_bins = action_bins.astype(jnp.int32)
        action_bins_flat = action_bins.reshape((-1, self.action_dim))

        q_logits = self.head(obs_embed_flat, action_bins_flat, training=training).astype(jnp.float32)
        q_logits = q_logits.reshape((*batch_axes, self.action_dim, self.num_bins))

        return q_logits


class QTransformerCriticWithAttentionWeights(QTransformerCritic):
    head_cls = _QTransformerHeadWithAttentionWeights


class QTransformerAgent(flax.struct.PyTreeNode):
    rng: any
    network: any
    config: any = nonpytree_field()

    def _reward_scaling_and_q_bounds(self):
        discount = float(self.config["discount"])
        reward_min = float(self.config["reward_min"])
        reward_max = float(self.config["reward_max"])
        sparse = bool(self.config["sparse"])

        if sparse:
            reward_scale = 1.0
            reward_shift = 0.0
            q_min = 0.0
            q_max = 1.0
            return (
                jnp.asarray(reward_scale, dtype=jnp.float32),
                jnp.asarray(reward_shift, dtype=jnp.float32),
                jnp.asarray(q_min, dtype=jnp.float32),
                jnp.asarray(q_max, dtype=jnp.float32),
            )

        reward_range = reward_max - reward_min
        assert reward_range > 0.0

        reward_scale = (1.0 - discount) / reward_range
        reward_shift = reward_min
        q_min = 0.0
        q_max = 1.0

        return (
            jnp.asarray(reward_scale, dtype=jnp.float32),
            jnp.asarray(reward_shift, dtype=jnp.float32),
            jnp.asarray(q_min, dtype=jnp.float32),
            jnp.asarray(q_max, dtype=jnp.float32),
        )

    def critic_loss(self, batch, grad_params, rng):
        num_bins = int(self.config["action_bins"])
        action_dim = int(self.config["action_dim"])
        ob_dims = tuple(self.config["ob_dims"])
        discount = float(self.config["discount"])
        conservative_weight = float(self.config["conservative_weight"])
        clip_q_target = bool(self.config["clip_q_target"])
        use_mc = bool(self.config["use_mc_return"])
        log_attention_entropy = self.config["log_attention_entropy"]
        dropout_rate = float(self.config["dropout_rate"])
        use_attention_entropy_loss = False
        if "use_attention_entropy_loss" in self.config:
            use_attention_entropy_loss = bool(self.config["use_attention_entropy_loss"])

        target_bootstrap = self.config["target_bootstrap"]
        allowed_bootstraps = (
            "one_step",
            "action_dim_n_step_argmax_rollout",
            "action_dim_n_step_greedy",
            "action_dim_n_step_sarsa",
        )
        if target_bootstrap not in allowed_bootstraps:
            raise ValueError(
                f"Unknown target_bootstrap={target_bootstrap}. Expected one of {allowed_bootstraps}."
            )

        reward_scale, reward_shift, q_min, q_max = self._reward_scaling_and_q_bounds()
        rewards = (batch["rewards"].astype(jnp.float32) - reward_shift) * reward_scale

        target_params = self.network.params if grad_params is None else jax.lax.stop_gradient(grad_params)

        action_low = self.config["action_low"]
        action_high = self.config["action_high"]
        action_bins = _actions_to_bins(batch["actions"], num_bins, action_low, action_high)

        training = grad_params is not None
        need_attention_weights = bool(log_attention_entropy) or use_attention_entropy_loss

        dropout_rng = None
        if training:
            rng, dropout_rng = jax.random.split(rng)

        critic_kwargs = dict(params=grad_params, training=training)
        if training and dropout_rate > 0.0:
            critic_kwargs["rngs"] = {"dropout": dropout_rng}

        if need_attention_weights:
            q_logits, intermediates = self.network.select("critic")(
                batch["observations"],
                action_bins=action_bins,
                mutable=["intermediates"],
                **critic_kwargs,
            )
            head_intermediates = intermediates["intermediates"]["modules_critic"]["head"]
            layer_weights = []
            for i in range(self.config["depth"]):
                layer_weights.append(head_intermediates[f"block_{i}"]["attention_weights"][0])
            attn_weights = jnp.stack(layer_weights, axis=0)
            attn_entropy = -jnp.sum(
                attn_weights * jnp.log(jnp.clip(attn_weights, a_min=1e-9)),
                axis=-1,
            )
            if self.config["attention_entropy_normalize"]:
                seq_len = attn_entropy.shape[-1]
                denom = jnp.log(jnp.arange(seq_len, dtype=attn_entropy.dtype) + 1.0)
                denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
                attn_entropy = attn_entropy / denom[None, None, None, :]
            attn_entropy_layer_mean = attn_entropy.mean(axis=tuple(range(1, attn_entropy.ndim)))
        else:
            q_logits = self.network.select("critic")(
                batch["observations"],
                action_bins=action_bins,
                **critic_kwargs,
            )
            attn_entropy_layer_mean = None
        q_logits_targ_cur = self.network.select("target_critic")(
            batch["observations"],
            action_bins=action_bins,
            params=target_params,
        )
        next_obs = batch["next_observations"]
        bootstrap_action_bins = None
        if target_bootstrap == "one_step":
            q_logits_targ_next = self.network.select("target_critic")(
                next_obs,
                action_bins=None,
                params=target_params,
            )
        elif target_bootstrap in ("action_dim_n_step_argmax_rollout", "action_dim_n_step_sarsa"):
            assert "next_actions" in batch
            bootstrap_action_bins = _actions_to_bins(
                batch["next_actions"],
                num_bins,
                action_low,
                action_high,
            )
            q_logits_targ_next = self.network.select("target_critic")(
                next_obs,
                action_bins=bootstrap_action_bins,
                params=target_params,
            )
        elif target_bootstrap == "action_dim_n_step_greedy":
            lead_shape = next_obs.shape[: -len(ob_dims)] if len(ob_dims) > 0 else next_obs.shape[:-1]
            greedy_bins = jnp.zeros((*lead_shape, action_dim), dtype=jnp.int32)

            def greedy_step(i, bins):
                q_logits_tmp = self.network.select("target_critic")(
                    next_obs,
                    action_bins=bins,
                    params=target_params,
                )
                scores = q_logits_tmp[..., i, :]
                sel = jnp.argmax(scores, axis=-1).astype(jnp.int32)
                return bins.at[..., i].set(sel)

            greedy_bins = jax.lax.fori_loop(0, action_dim - 1, greedy_step, greedy_bins)
            q_logits_targ_next = self.network.select("target_critic")(
                next_obs,
                action_bins=greedy_bins,
                params=target_params,
            )

        q_targ_cur = q_min + jax.nn.sigmoid(q_logits_targ_cur)
        q_targ_cur_max = q_targ_cur.max(axis=-1)

        q_next = q_min + jax.nn.sigmoid(q_logits_targ_next)
        if target_bootstrap == "one_step":
            q_next_bootstrap = q_next[..., 0, :].max(axis=-1)
        elif target_bootstrap == "action_dim_n_step_sarsa":
            assert bootstrap_action_bins is not None
            last_dim_q = q_next[..., -1, :]
            last_dim_bin = bootstrap_action_bins[..., -1]
            q_next_bootstrap = jnp.take_along_axis(
                last_dim_q,
                last_dim_bin[..., None],
                axis=-1,
            ).squeeze(-1)
        else:
            q_next_bootstrap = q_next[..., -1, :].max(axis=-1)

        q_target_prefix = q_targ_cur_max[..., 1:]
        assert "terminals" in batch
        bootstrap_mask = batch["masks"].astype(jnp.float32) * (1.0 - batch["terminals"].astype(jnp.float32))
        q_target_last = rewards + discount * bootstrap_mask * q_next_bootstrap
        q_target = jnp.concatenate([q_target_prefix, q_target_last[..., None]], axis=-1)

        if use_mc and ("mc_returns" in batch):
            mc = batch["mc_returns"]
            if mc.ndim == 2 and mc.shape[-1] == 1:
                mc = mc[:, 0]
            mc = mc.astype(jnp.float32)
            mc = (mc - reward_shift / (1.0 - discount)) * reward_scale
            q_target = jnp.maximum(q_target, mc[..., None])

        if clip_q_target:
            q_target = jnp.clip(q_target, q_min, q_max)

        q_target = jax.lax.stop_gradient(q_target)

        idx = jnp.broadcast_to(action_bins, q_logits.shape[:-1])[..., None]
        q_pred_logits = jnp.take_along_axis(q_logits, idx, axis=-1).squeeze(-1)
        q_pred_sigmoid = jax.nn.sigmoid(q_pred_logits)
        q_pred = q_min + q_pred_sigmoid

        td = 0.5 * jnp.mean(jnp.square(q_pred - q_target))

        q_all_sigmoid = jax.nn.sigmoid(q_logits)
        q_sq_sum = jnp.sum(jnp.square(q_all_sigmoid), axis=-1)
        q_data_sq = jnp.square(q_pred_sigmoid)
        reg = (q_sq_sum - q_data_sq) / float(num_bins - 1)
        reg = 0.5 * jnp.mean(reg)

        loss = td + conservative_weight * reg

        if use_attention_entropy_loss:
            temperature = self.network.select("attention_entropy_temperature")(params=grad_params)
            temp_min = float(self.config["attention_entropy_temperature_min"])
            temp_max = float(self.config["attention_entropy_temperature_max"])
            temperature_clipped = jnp.clip(temperature, temp_min, temp_max)
            temperature_no_grad = jax.lax.stop_gradient(temperature_clipped)

            attention_entropy_loss = (-temperature_no_grad * attn_entropy_layer_mean).mean()

            attn_entropy_layer_mean_no_grad = jax.lax.stop_gradient(attn_entropy_layer_mean)
            target_entropy = jnp.asarray(
                self.config["attention_entropy_target"],
                dtype=attn_entropy_layer_mean_no_grad.dtype,
            )
            temperature_loss = (temperature * (attn_entropy_layer_mean_no_grad - target_entropy)).mean()
            loss = loss + attention_entropy_loss + temperature_loss

        info = {
            "loss": loss,
            "td_loss": td,
            "conservative_loss": reg,
            "q_pred_mean": q_pred.mean(),
            "q_targ_mean": q_target.mean(),
            "q_targ_max": q_target.max(),
            "q_targ_min": q_target.min(),
            "reward_scale": reward_scale,
            "reward_shift": reward_shift,
            "q_min": q_min,
            "q_max": q_max,
            "raw_reward_min": float(self.config["reward_min"]),
            "raw_reward_max": float(self.config["reward_max"]),
        }
        if need_attention_weights:
            for i in range(attn_entropy_layer_mean.shape[0]):
                info[f"attention_entropy_mean_layer_{i}"] = attn_entropy_layer_mean[i]
        if use_attention_entropy_loss:
            info["attention_entropy_loss"] = attention_entropy_loss
            info["attention_entropy_temperature_loss"] = temperature_loss
            for i in range(int(temperature.shape[0])):
                info[f"attention_entropy_temperature_layer_{i}"] = temperature[i]
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        loss, info = self.critic_loss(batch, grad_params, rng)
        info = {f"critic/{k}": v for k, v in info.items()}
        return loss, info

    def target_update(self, network, module_name):
        tau = float(self.config["tau"])
        online = network.params[f"modules_{module_name}"]
        target = network.params[f"modules_target_{module_name}"]
        new_target = jax.tree_util.tree_map(lambda p, tp: tp + tau * (p - tp), online, target)
        new_params = _set_top_level(network.params, f"modules_target_{module_name}", new_target)
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        lr_step = jnp.maximum(jnp.asarray(self.network.step, dtype=jnp.int32) - 1, 0)
        base_lr = self.config["lr"]
        warmup_steps = self.config["warmup_steps"] if "warmup_steps" in self.config else 0
        if warmup_steps is None:
            warmup_steps = 0
        total_steps = self.config["train_steps"] if "train_steps" in self.config else 1000000
        if total_steps is None:
            total_steps = 1000000
        decay_steps = total_steps - warmup_steps
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=base_lr * 0.01,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=base_lr * 0.1,
        )
        info["lr"] = lr_schedule(lr_step)
        new_network = self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=0.0):
        num_bins = int(self.config["action_bins"])
        temperature = jnp.asarray(temperature, dtype=jnp.float32)
        action_dim = int(self.config["action_dim"])
        ob_dims = tuple(self.config["ob_dims"])
        action_low = self.config["action_low"]
        action_high = self.config["action_high"]

        if seed is None:
            seed = self.rng

        lead_shape = observations.shape[: -len(ob_dims)] if len(ob_dims) > 0 else observations.shape[:-1]
        bins = jnp.zeros((*lead_shape, action_dim), dtype=jnp.int32)

        def body(i, carry):
            key, cur_bins = carry
            key, subkey = jax.random.split(key)
            q_logits = self.network.select("critic")(observations, action_bins=cur_bins)
            q_logits_i = q_logits[..., i, :]

            sel = jax.lax.cond(
                temperature <= 0.0,
                lambda _: jnp.argmax(q_logits_i, axis=-1).astype(jnp.int32),
                lambda _: jax.random.categorical(subkey, q_logits_i / temperature, axis=-1).astype(jnp.int32),
                operand=None,
            )

            cur_bins = cur_bins.at[..., i].set(sel)
            return key, cur_bins

        _, bins = jax.lax.fori_loop(0, action_dim, body, (seed, bins))
        actions = _bins_to_actions(bins, num_bins, action_low, action_high)
        return actions

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        action_low = config["action_low"]
        action_high = config["action_high"]
        if action_low is None or action_high is None:
            # Prefer dataset-wide bounds if available (computed in main.py)
            if "dataset_action_min" in example_batch and "dataset_action_max" in example_batch:
                action_low = jnp.asarray(example_batch["dataset_action_min"], dtype=jnp.float32)
                action_high = jnp.asarray(example_batch["dataset_action_max"], dtype=jnp.float32)
            else:
                # Fallback: infer from example_batch only
                ex_actions_f32 = ex_actions.astype(jnp.float32).reshape((-1, action_dim))
                action_low = jnp.min(ex_actions_f32, axis=0)
                action_high = jnp.max(ex_actions_f32, axis=0)
                if "next_actions" in example_batch:
                    na = example_batch["next_actions"].astype(jnp.float32).reshape((-1, action_dim))
                    action_low = jnp.minimum(action_low, jnp.min(na, axis=0))
                    action_high = jnp.maximum(action_high, jnp.max(na, axis=0))
        else:
            action_low = jnp.asarray(action_low, dtype=jnp.float32)
            action_high = jnp.asarray(action_high, dtype=jnp.float32)

        assert action_low.shape == (action_dim,)
        assert action_high.shape == (action_dim,)

        eps = jnp.asarray(1e-6, dtype=jnp.float32)
        rng_ = action_high - action_low
        expand = (rng_ < eps).astype(jnp.float32)
        action_low = action_low - expand
        action_high = action_high + expand

        action_low_h = jax.device_get(action_low)
        action_high_h = jax.device_get(action_high)
        assert action_low_h.shape == (action_dim,)
        assert action_high_h.shape == (action_dim,)
        assert (action_high_h > action_low_h).all(), (action_low_h, action_high_h)
        config["action_low"] = tuple(float(x) for x in action_low_h.tolist())
        config["action_high"] = tuple(float(x) for x in action_high_h.tolist())

        assert "min_reward" in example_batch
        assert "max_reward" in example_batch
        if config["reward_min"] is None:
            config["reward_min"] = float(example_batch["min_reward"])
        if config["reward_max"] is None:
            config["reward_max"] = float(example_batch["max_reward"])
        assert config["reward_min"] <= config["reward_max"]

        enc = None
        use_bf16 = bool(config["use_bf16"])
        dtype = jnp.bfloat16 if use_bf16 else jnp.float32
        if config["encoder"] is not None:
            enc = encoder_modules[config["encoder"]](dtype=dtype)

        use_attention_entropy_loss = False
        if "use_attention_entropy_loss" in config:
            use_attention_entropy_loss = bool(config["use_attention_entropy_loss"])

        if use_attention_entropy_loss:
            depth = int(config["depth"])

            if "attention_entropy_temperature_init" not in config:
                config["attention_entropy_temperature_init"] = 1.0
            if "attention_entropy_temperature_min" not in config:
                config["attention_entropy_temperature_min"] = -100.0
            if "attention_entropy_temperature_max" not in config:
                config["attention_entropy_temperature_max"] = 100.0
            if "attention_entropy_target" not in config:
                config["attention_entropy_target"] = 0.0

            target_entropy = config["attention_entropy_target"]
            if isinstance(target_entropy, (int, float)):
                config["attention_entropy_target"] = tuple(float(target_entropy) for _ in range(depth))
            else:
                ok = isinstance(target_entropy, (list, tuple)) and len(target_entropy) == depth
                if not ok:
                    raise ValueError(
                        "attention_entropy_target must be a scalar or a sequence of length depth "
                        f"(depth={depth})."
                    )
                config["attention_entropy_target"] = tuple(float(x) for x in target_entropy)

        critic_cls = (
            QTransformerCriticWithAttentionWeights
            if bool(config["log_attention_entropy"]) or use_attention_entropy_loss
            else QTransformerCritic
        )
        critic_def = critic_cls(
            action_dim=action_dim,
            num_bins=int(config["action_bins"]),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            num_heads=int(config["num_heads"]),
            mlp_ratio=int(config["mlp_ratio"]),
            layer_norm=bool(config["layer_norm"]),
            encoder=enc,
            init_logit=float(config["init_logit"]),
            use_qk_norm=bool(config["use_qk_norm"]),
            dropout_rate=float(config["dropout_rate"]),
            final_layer_norm=bool(config["final_layer_norm"]),
            use_state_tokens_per_dim=config["use_state_tokens_per_dim"],
            use_modality_embeddings=config["use_modality_embeddings"],
            dtype=dtype,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, jnp.zeros((ex_observations.shape[0], action_dim), dtype=jnp.int32))),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, jnp.zeros((ex_observations.shape[0], action_dim), dtype=jnp.int32)),
            ),
        )
        if use_attention_entropy_loss:
            network_info["attention_entropy_temperature"] = (
                _AttentionEntropyTemperature(
                    num_layers=int(config["depth"]),
                    init_value=float(config["attention_entropy_temperature_init"]),
                ),
                (),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        network_params = network_def.init(init_rng, **network_args)["params"]
        network_params = _set_top_level(
            network_params, "modules_target_critic", network_params["modules_critic"]
        )

        def should_apply_wd(path, value):
            keys = []
            for key in path:
                if hasattr(key, "key"):
                    keys.append(key.key)
                else:
                    keys.append(key)
            if "modules_target_critic" in keys:
                return False
            if value.ndim <= 1:
                return False
            if "bias" in keys:
                return False
            if (
                "pos_embed" in keys
                or "start_token" in keys
                or "action_in_embed" in keys
                or "action_out_embed" in keys
                or "action_out_bias" in keys
            ):
                return False
            return True

        optimizer_type = config["optimizer"]
        base_lr = float(config["lr"])
        warmup_steps = config["warmup_steps"] if "warmup_steps" in config else 0
        if warmup_steps is None:
            warmup_steps = 0
        total_steps = config["train_steps"] if "train_steps" in config else 1000000
        if total_steps is None:
            total_steps = 1000000
        assert total_steps > warmup_steps, (
            f"total_steps {total_steps} must be > warmup_steps {warmup_steps}"
        )
        decay_steps = total_steps - warmup_steps
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=base_lr * 0.01,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=base_lr * 0.1,
        )
        if optimizer_type == "adam":
            base_tx = optax.adam(learning_rate=lr_schedule)
        elif optimizer_type == "adamw":
            mask = jax.tree_util.tree_map_with_path(should_apply_wd, network_params)
            base_tx = optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=config["adamw_weight_decay"],
                mask=mask,
            )
        else:
            raise ValueError(optimizer_type)

        tx = optax.chain(
            optax.clip_by_global_norm(float(config["max_grad_norm"])),
            base_tx,
        )
        network = TrainState.create(network_def, network_params, tx=tx)

        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="q_transformer",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=2e-5,
            optimizer="adamw",
            adamw_weight_decay=0.02,
            warmup_steps=10000,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            action_bins=64,
            hidden_dim=128,
            depth=2,
            num_heads=8,
            mlp_ratio=4,
            layer_norm=True,
            final_layer_norm=True,
            use_qk_norm=True,
            dropout_rate=0.0,
            use_bf16=False,
            conservative_weight=0.5,
            max_grad_norm=1.0,
            use_mc_return=True,
            clip_q_target=True,
            target_bootstrap="action_dim_n_step_greedy",
            init_logit=-2.5,
            log_attention_entropy=False,
            attention_entropy_normalize=True,
            use_attention_entropy_loss=False,
            attention_entropy_temperature_init=1.0,
            attention_entropy_temperature_min=-100.0,
            attention_entropy_temperature_max=100.0,
            attention_entropy_target=(0.75, 0.75),
            use_state_tokens_per_dim=False,
            use_modality_embeddings=False,
            encoder=ml_collections.config_dict.placeholder(str),
            reward_min=ml_collections.config_dict.placeholder(float),
            reward_max=ml_collections.config_dict.placeholder(float),
            sparse=False,
            action_low=None,
            action_high=None,
        )
    )
