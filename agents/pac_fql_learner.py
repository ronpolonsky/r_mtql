import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField

# ============================================================
# Attention building blocks
# ============================================================
class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention + MLP (pre-LN)."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, context, training: bool = False):
        q_input = nn.LayerNorm(name="q_ln")(x)
        kv_input = nn.LayerNorm(name="kv_ln")(context)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        head_dim = self.hidden_dim // self.num_heads

        query = nn.DenseGeneral(features=(self.num_heads, head_dim), name="query")(q_input)
        key = nn.DenseGeneral(features=(self.num_heads, head_dim), name="key")(kv_input)
        value = nn.DenseGeneral(features=(self.num_heads, head_dim), name="value")(kv_input)

        scale = jnp.sqrt(head_dim).astype(q_input.dtype)
        logits = jnp.einsum("...qhd,...khd->...hqk", query, key) / scale
        attn = nn.softmax(logits, axis=-1)

        out = jnp.einsum("...hqk,...khd->...qhd", attn, value)
        out = out.reshape(*q_input.shape[:-1], self.hidden_dim)
        out = nn.Dense(self.hidden_dim, name="out")(out)
        out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)

        x = x + out

        mlp_in = nn.LayerNorm(name="mlp_ln")(x)
        mlp_dim = self.hidden_dim * self.mlp_ratio
        mlp = nn.Dense(mlp_dim, name="mlp_fc1")(mlp_in)
        mlp = nn.gelu(mlp)
        mlp = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp)
        mlp = nn.Dense(self.hidden_dim, name="mlp_fc2")(mlp)
        mlp = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp)

        return x + mlp


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention + MLP (pre-LN)."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        attn_in = nn.LayerNorm(name="attn_ln")(x)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        head_dim = self.hidden_dim // self.num_heads

        query = nn.DenseGeneral(features=(self.num_heads, head_dim), name="query")(attn_in)
        key = nn.DenseGeneral(features=(self.num_heads, head_dim), name="key")(attn_in)
        value = nn.DenseGeneral(features=(self.num_heads, head_dim), name="value")(attn_in)

        scale = jnp.sqrt(head_dim).astype(attn_in.dtype)
        logits = jnp.einsum("...qhd,...khd->...hqk", query, key) / scale
        attn = nn.softmax(logits, axis=-1)

        out = jnp.einsum("...hqk,...khd->...qhd", attn, value)
        out = out.reshape(*attn_in.shape[:-1], self.hidden_dim)
        out = nn.Dense(self.hidden_dim, name="out")(out)
        out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)

        x = x + out

        mlp_in = nn.LayerNorm(name="mlp_ln")(x)
        mlp_dim = self.hidden_dim * self.mlp_ratio
        mlp = nn.Dense(mlp_dim, name="mlp_fc1")(mlp_in)
        mlp = nn.gelu(mlp)
        mlp = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp)
        mlp = nn.Dense(self.hidden_dim, name="mlp_fc2")(mlp)
        mlp = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp)

        return x + mlp


# ============================================================
# Perceiver backbone
# ============================================================

class PerceiverBackbone(nn.Module):
    """Latents cross-attend to inputs, then latent self-attention blocks."""
    hidden_dim: int
    num_latents: int = 32
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs, training: bool = False):
        batch_shape = inputs.shape[:-2]

        latents = self.param(
            "latents",
            nn.initializers.normal(stddev=0.02),
            (self.num_latents, self.hidden_dim),
        )
        latents = latents.reshape((1,) * len(batch_shape) + latents.shape)
        latents = jnp.tile(latents, batch_shape + (1, 1))

        latents = CrossAttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="enc_xattn",
        )(latents, inputs, training=training)

        for i in range(self.num_layers):
            latents = SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f"latent_block_{i}",
            )(latents, training=training)

        latents = nn.LayerNorm(name="latent_final_ln")(latents)
        return latents


# ============================================================
# Tokenizers
# ============================================================

class PACContinuousObsTokenizer(nn.Module):
    """obs (..., D) -> tokens (..., D, H) by treating each dimension as a token."""
    hidden_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        batch_shape = obs.shape[:-1]
        obs_dim = obs.shape[-1]

        obs_tokens = obs[..., None]  # (..., D, 1)
        obs_tokens = nn.Dense(self.hidden_dim, name="obs_token_proj")(obs_tokens)  # (..., D, H)

        pos = self.param("obs_pos", nn.initializers.normal(0.02), (obs_dim, self.hidden_dim))
        pos = jnp.reshape(pos, (1,) * len(batch_shape) + pos.shape)
        return obs_tokens + pos


class PACActionQueryEncoder(nn.Module):
    """action -> query (..., 1, H) via a single linear embedding + normalization.

    Per spec: directly embed the full action vector into one token using a linear
    layer followed by normalization (no positional embeddings, no mean pooling).
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        # 1) Tokenize each action dimension independently: (..., A) -> (..., A, H)
        # No positional embeddings (action dims are fixed and their identities are encoded via weights).
        action_tokens = nn.Dense(self.hidden_dim, name="action_token_proj")(actions[..., None])  # (..., A, H)

        # 2) Collapse tokens into a single query token with a learned linear projection (no mean pooling).
        flat = action_tokens.reshape(*action_tokens.shape[:-2], -1)  # (..., A*H)
        q = nn.Dense(self.hidden_dim, name="action_query_proj")(flat)  # (..., H)
        q = nn.LayerNorm(name="action_query_ln")(q)
        return q[..., None, :]  # (..., 1, H)


# ============================================================
# Shared backbone + two heads (policy + distributional Q)
# ============================================================

class PACBackbone(nn.Module):
    """Shared obs tokenizer + perceiver latents."""
    hidden_dim: int
    num_latents: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    dropout_rate: float
    encoder: Any = None

    @nn.compact
    def __call__(self, observations, training: bool = False) -> jnp.ndarray:
        if self.encoder is not None:
            observations = self.encoder(observations)

        obs_tokens = PACContinuousObsTokenizer(hidden_dim=self.hidden_dim, name="obs_tok")(observations)

        latents = PerceiverBackbone(
            hidden_dim=self.hidden_dim,
            num_latents=self.num_latents,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="perceiver",
        )(obs_tokens, training=training)

        return latents  # (..., L, H)


class PACActorCritic(nn.Module):
    """
    Single network:
      shared backbone (obs -> latents)
      policy head: latents -> actions (A,)
      critic head: (latents, action query) -> logits (num_atoms)
    """
    hidden_dim: int
    action_dim: int
    num_atoms: int

    num_latents: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    dropout_rate: float

    encoder: Any = None

    @nn.compact
    def __call__(self, observations, actions=None, training: bool = False, return_latents: bool = False):
        latents = PACBackbone(
            hidden_dim=self.hidden_dim,
            num_latents=self.num_latents,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            encoder=self.encoder,
            name="backbone",
        )(observations, training=training)

        

        # ---------------- policy head ----------------
        pi_q = self.param(
            "pi_queries",
            nn.initializers.normal(stddev=0.02),
            (self.action_dim, self.hidden_dim),
        )
        batch_shape = latents.shape[:-2]
        pi_q = jnp.reshape(pi_q, (1,) * len(batch_shape) + pi_q.shape)
        pi_q = jnp.tile(pi_q, batch_shape + (1, 1))

        pi_rep = CrossAttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="pi_decode_xattn",
        )(pi_q, latents, training=training)

        pi_rep = nn.LayerNorm(name="pi_rep_ln")(pi_rep)
        policy_actions = nn.Dense(1, name="pi_out")(pi_rep)[..., 0]  # (...,A)
        policy_actions = jnp.tanh(policy_actions)

        if actions is None:
            if return_latents:
                return policy_actions, latents
            return policy_actions

        # ---------------- critic head ----------------
        q_query = PACActionQueryEncoder(hidden_dim=self.hidden_dim, name="action_query")(actions)  # (...,1,H)

        q_rep = CrossAttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="q_decode_xattn",
        )(q_query, latents, training=training)
        q_rep = nn.LayerNorm(name="q_rep_ln")(q_rep)[..., 0, :]  # (...,H)

        h = nn.Dense(self.hidden_dim // 2, name="q_head_h")(q_rep)
        h = nn.gelu(h)
        q_logits = nn.Dense(self.num_atoms, name="q_head_out")(h)  # (...,num_atoms)

        if return_latents:
            return policy_actions, q_logits, latents
        return policy_actions, q_logits


class PACFQLActorAgent(flax.struct.PyTreeNode):
    """PAC critic + FQL-style flow actor (BC flow + distill + Q-loss).

    - Critic: distributional Q with categorical projection (same as PACAgent).
    - Actor: continuous flow-matching actor (same style as FQL/TQL-MC actors).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ---------------------------
    # Distributional critic utilities (same as PACAgent)
    # ---------------------------
    def support(self) -> jnp.ndarray:
        n = int(self.config["num_atoms"])
        v_min = float(self.config["v_min"])
        v_max = float(self.config["v_max"])
        return jnp.linspace(v_min, v_max, n, dtype=jnp.float32)

    def expected_q(self, logits: jnp.ndarray) -> jnp.ndarray:
        p = jax.nn.softmax(logits, axis=-1)
        z = self.support()
        return jnp.sum(p * z, axis=-1)

    def project_categorical(self, next_probs: jnp.ndarray, r: jnp.ndarray, disc: jnp.ndarray) -> jnp.ndarray:
        z = self.support()  # (N,)
        v_min = float(self.config["v_min"])
        v_max = float(self.config["v_max"])
        N = z.shape[0]
        dz = (v_max - v_min) / (N - 1)

        Tz = r[:, None] + disc[:, None] * z[None, :]  # (B,N)
        Tz = jnp.clip(Tz, v_min, v_max)
        b = (Tz - v_min) / dz
        l = jnp.floor(b).astype(jnp.int32)
        l = jnp.clip(l, 0, N - 1)
        u = jnp.clip(l + 1, 0, N - 1)

        w_u = b - l.astype(b.dtype)
        w_l = 1.0 - w_u

        def project_one(p, l_idx, u_idx, wl, wu):
            m = jnp.zeros((N,), dtype=p.dtype)
            m = m.at[l_idx].add(p * wl)
            m = m.at[u_idx].add(p * wu)
            return m

        m = jax.vmap(project_one)(next_probs, l, u, w_l, w_u)  # (B,N)
        m = jnp.clip(m, a_min=0.0)
        m = m / (jnp.sum(m, axis=-1, keepdims=True) + 1e-8)
        return m

    # ---------------------------
    # Critic loss (same as PACAgent)
    # ---------------------------
    def critic_loss(self, batch, grad_params, rng, step=None):
        rng, sample_rng, dropout_rng = jax.random.split(rng, 3)

        next_actions = self.sample_actions(batch["next_observations"], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        # target critic distribution at (s', a')
        _, next_q_logits = self.network.select("target_ac")(
            batch["next_observations"],
            actions=next_actions,
            training=False,
        )  # (B,N)

        next_probs = jax.nn.softmax(next_q_logits, axis=-1)  # (B,N)
        disc = jnp.asarray(self.config["discount"], jnp.float32) * batch["masks"]  # (B,)

        target_probs = self.project_categorical(
            next_probs=next_probs,
            r=batch["rewards"],
            disc=disc,
        )  # (B,N)

        # current logits at (s, a)
        _, q_logits = self.network.select("ac")(
            batch["observations"],
            actions=batch["actions"],
            params=grad_params,
            training=True,
            rngs={"dropout": dropout_rng},
        )  # (B,N)

        logp = jax.nn.log_softmax(q_logits, axis=-1)
        q_loss = -jnp.sum(target_probs * logp, axis=-1).mean()

        q = self.expected_q(q_logits)
        target_q = batch["rewards"] + disc * self.expected_q(next_q_logits)

        info = {
            "critic_loss": q_loss,
            "q_loss": q_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "target_q_mean": target_q.mean(),
        }
        return q_loss, info

    # ---------------------------
    # FQL-style flow actor
    # ---------------------------
    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config.get("encoder", None) is not None:
            # Optional encoder for flow BC model (separate module if created).
            if "actor_bc_flow_encoder" in self.network.params:
                observations = self.network.select("actor_bc_flow_encoder")(observations)
        actions = noises
        for i in range(int(self.config["flow_steps"])):
            t = jnp.full((*observations.shape[:-1], 1), i / float(self.config["flow_steps"]))
            vels = self.network.select("actor_bc_flow")(observations, actions, t, is_encoded=True)
            actions = actions + vels / float(self.config["flow_steps"])
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature: float = 1.0):
        rng = seed if seed is not None else jax.random.PRNGKey(0)
        if observations.ndim == 1:
            observations = observations[None, :]
        noises = jax.random.normal(rng, (*observations.shape[:-1], int(self.config["action_dim"])))
        actions = self.network.select("actor_onestep_flow")(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions[0] if actions.shape[0] == 1 else actions

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(batch["observations"], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss (match one-step flow to multi-step flow)
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch["observations"], noises=noises)
        actor_actions = self.network.select("actor_onestep_flow")(batch["observations"], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss (maximize expected Q under the current critic distribution)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        _, q_logits = self.network.select("ac")(
            batch["observations"],
            actions=actor_actions,
            training=False,
        )  # (B,N)
        q = self.expected_q(q_logits)  # (B,)
        q_loss = -q.mean()

        actor_loss = bc_flow_loss + jnp.asarray(self.config["alpha"], dtype=jnp.float32) * distill_loss + q_loss
        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q": q.mean(),
        }

    # ---------------------------
    # Train step
    # ---------------------------
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=None):
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        c_loss, c_info = self.critic_loss(batch, grad_params, critic_rng, step=step)
        a_loss, a_info = self.actor_loss(batch, grad_params, actor_rng)

        total = c_loss + a_loss

        info = {}
        for k, v in c_info.items():
            info[f"critic/{k}"] = v
        for k, v in a_info.items():
            info[f"actor/{k}"] = v
        info["loss/critic_loss"] = c_loss
        info["loss/actor_loss"] = a_loss
        info["loss/total"] = total
        return total, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch, step=None):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "ac")
        return self.replace(network=new_network, rng=new_rng), info

    # ---------------------------
    # Create
    # ---------------------------
    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_obs = example_batch["observations"]
        ex_act = example_batch["actions"]
        ob_dims = ex_obs.shape[1:]
        action_dim = ex_act.shape[-1]

        # Optional shared encoder
        encoder = None
        if config.get("encoder", None) is not None:
            encoder = encoder_modules[config["encoder"]]()

        # PAC critic module (we will ignore policy_logits during training, but it's fine).
        ac_def = PACActorCritic(
            hidden_dim=config["hidden_dim"],
            action_dim=action_dim,
            num_atoms=config["num_atoms"],
            num_latents=config["num_latents"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            dropout_rate=config["dropout_rate"],
            encoder=encoder,
        )

        # Flow actor modules (FQL style).
        ex_times = ex_act[..., :1]
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoder,
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoder,
        )

        network_info = dict(
            ac=(ac_def, (ex_obs, ex_act)),
            target_ac=(copy.deepcopy(ac_def), (ex_obs, ex_act)),
            actor_bc_flow=(actor_bc_flow_def, (ex_obs, ex_act, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_obs, ex_act)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        init_rng_dict = {"params": init_rng, "dropout": jax.random.PRNGKey(0)}
        params = network_def.init(init_rng_dict, **network_args)["params"]

        def make_optimizer(lr, warmup_steps=0, total_steps=1_000_000):
            if warmup_steps > 0:
                decay_steps = total_steps - warmup_steps
                lr_schedule = optax.warmup_cosine_decay_schedule(
                    init_value=lr * 0.01,
                    peak_value=lr,
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps,
                    end_value=lr * 0.1,
                )
            else:
                lr_schedule = lr

            # Weight decay mask (follow `agents/tql_temp_multi_l2.py` style).
            def should_apply_wd(path, value):
                p = "/".join(str(k.key if hasattr(k, "key") else k) for k in path).lower()
                if "modules_target_" in p:
                    return False
                if any(
                    x in p
                    for x in (
                        "bias",
                        "layernorm",
                        "final_ln",
                        "pos_embedding",
                        "cls_token",
                        "state_type_embed",
                        "action_type_embed",
                        "latents",
                        "pi_queries",
                    )
                ):
                    return False
                return value.ndim > 1

            mask = jax.tree_util.tree_map_with_path(should_apply_wd, params)
            return optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=config.get("adamw_weight_decay", 0.0),
                mask=mask,
            )

        critic_lr = float(config["critic_lr"])
        actor_lr = float(config["actor_lr"])

        critic_optimizer = make_optimizer(
            lr=critic_lr,
            warmup_steps=int(config.get("warmup_steps", 0)),
            total_steps=int(config.get("train_steps", 1_000_000)),
        )
        actor_optimizer = make_optimizer(
            lr=actor_lr,
            warmup_steps=int(config.get("warmup_steps", 0)),
            total_steps=int(config.get("train_steps", 1_000_000)),
        )

        def partition_fn(path, _value):
            path_str = "/".join(str(k.key if hasattr(k, "key") else k) for k in path).lower()
            if "modules_ac" in path_str or "modules_target_ac" in path_str:
                return "critic"
            return "actor"

        param_labels = jax.tree_util.tree_map_with_path(partition_fn, params)
        network_tx = optax.multi_transform(
            {"critic": critic_optimizer, "actor": actor_optimizer},
            param_labels,
        )
        network = TrainState.create(network_def, params, tx=network_tx)
        network.params["modules_target_ac"] = copy.deepcopy(network.params["modules_ac"])

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="pac_fql_actor",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),

            # Optim / training
            critic_lr=1e-4,
            actor_lr=1e-4,
            adamw_weight_decay=0.001,
            warmup_steps=10_000,
            train_steps=1_000_000,
            batch_size=256,
            discount=0.99,
            tau=0.005,

            # Distributional critic (same as PACAgent)
            num_atoms=201,
            v_min=-200.0,
            v_max=0.0,

            # PAC perceiver sizes
            hidden_dim=64,
            num_latents=16,
            num_layers=2,
            num_heads=4,
            mlp_ratio=4,
            dropout_rate=0.0,

            # FQL-style flow actor
            alpha=10.0,  # distillation coefficient
            flow_steps=10,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,

            # Optional shared encoder name
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )