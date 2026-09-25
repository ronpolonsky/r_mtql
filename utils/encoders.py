import functools
from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.core import FrozenDict

from utils.networks import MLP


def _is_dict_obs(obs):
    """Return True for both plain dict and Flax FrozenDict observations."""
    return isinstance(obs, (dict, FrozenDict))


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class ObservationEncoder(nn.Module):
    """Unified encoder for flat-vector or image+proprio dict observations.

    Signature: ``(observations, history_observations=None) -> (obs_enc, hist_enc)``

    Two input forms are supported:
      - **dict** ``{'image': (..., H, W, C), 'proprio': (..., D)}``:
        image is passed through ``image_encoder``; result is concatenated with proprio.
      - **plain array** ``(..., obs_dim)``:
        returned unchanged (no CNN needed; ``image_encoder`` is ignored).

    History observations follow the same logic but operate over the T dimension:
      - dict history: ``{'image': (..., T, H, W, C), 'proprio': (..., T, D)}``
      - plain history: ``(..., T, obs_dim)``
    """
    image_encoder: Any = None  # ImpalaEncoder instance; None for flat-only obs

    def _encode_single(self, obs):
        if _is_dict_obs(obs):
            image_embs = []
            image_views_keys = sorted(
                k for k in obs.keys() if "image" in k or "view" in k
            )

            for img_key in image_views_keys:

                img_enc = self.image_encoder(obs[img_key])
                image_embs.append(img_enc)

            if 'proprio' in obs:
                image_embs.append(obs['proprio'].astype(jnp.float32))
            return jnp.concatenate(image_embs, axis=-1)
        if self.image_encoder is not None and obs.ndim >= 3:
            # Plain image array: (..., H, W, C) — encode directly.
            return self.image_encoder(obs.astype(jnp.float32))
        return obs  # flat vector: identity

    def _encode_history(self, hist_obs):
        if _is_dict_obs(hist_obs):
            # Mirror _encode_single: sort image/view keys, encode each view,
            # concatenate with proprio. Each image value: (..., T, H, W, C).
            image_views_keys = sorted(
                k for k in hist_obs.keys() if "image" in k or "view" in k
            )
            first_img = hist_obs[image_views_keys[0]]
            prefix = first_img.shape[:-4]
            T = first_img.shape[-4]
            img_shape = first_img.shape[-3:]

            image_embs = []
            for img_key in image_views_keys:
                flat_img = hist_obs[img_key].reshape(-1, *img_shape)
                enc = self.image_encoder(flat_img)  # (B*T, enc_dim)
                image_embs.append(enc.reshape(*prefix, T, enc.shape[-1]))

            if 'proprio' in hist_obs:
                image_embs.append(hist_obs['proprio'].astype(jnp.float32))
            return jnp.concatenate(image_embs, axis=-1)
        if self.image_encoder is not None and hist_obs.ndim >= 4:
            # Plain image history: (..., T, H, W, C) — flatten T into batch, encode, restore.
            prefix = hist_obs.shape[:-4]
            T = hist_obs.shape[-4]
            img_shape = hist_obs.shape[-3:]
            flat_img = hist_obs.reshape(-1, *img_shape)
            enc_img = self.image_encoder(flat_img.astype(jnp.float32))  # (B*T, enc_dim)
            return enc_img.reshape(*prefix, T, enc_img.shape[-1])
        return hist_obs  # flat array: identity

    def __call__(self, observations, history_observations=None):
        return (
            self._encode_single(observations),
            self._encode_history(history_observations) if history_observations is not None else None,
        )


def _wrap(impala_factory):
    """Wrap an ImpalaEncoder factory in ObservationEncoder for a uniform interface."""
    def _make():
        return ObservationEncoder(image_encoder=impala_factory())
    return _make


# ---------------------------------------------------------------------------
# mem-ViT: ViT with interleaved causal temporal attention
# Paper: https://arxiv.org/abs/2603.03596 (video encoder section)
# ---------------------------------------------------------------------------

class _MemViTSpatialBlock(nn.Module):
    """Standard ViT block: pre-norm spatial MHA + FFN.

    Operates on (..., P, D) — treats all leading dimensions as batch.
    """
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        head_dim = self.embed_dim // self.num_heads
        # Spatial self-attention
        residual = x
        x = nn.LayerNorm()(x)
        q = nn.DenseGeneral((self.num_heads, head_dim), name='q')(x)
        k = nn.DenseGeneral((self.num_heads, head_dim), name='k')(x)
        v = nn.DenseGeneral((self.num_heads, head_dim), name='v')(x)
        scale = jnp.sqrt(jnp.array(head_dim, dtype=x.dtype))
        logits = jnp.einsum('...qhd,...khd->...hqk', q, k) / scale
        attn = nn.softmax(logits, axis=-1)
        out = jnp.einsum('...hqk,...khd->...qhd', attn, v)
        out = out.reshape(*x.shape[:-1], self.embed_dim)
        out = nn.Dense(self.embed_dim, name='proj')(out)
        x = residual + out
        # FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim * self.mlp_ratio)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        return residual + x


class _MemViTTemporalBlock(nn.Module):
    """Causal temporal MHA: each patch attends across frames.

    Input/output: (B, T, P, D)
    Internally reshapes to (B*P, T, D) for attention, then back.
    Uses a lower-triangular causal mask so frame t can only attend to 0..t.
    """
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, T, P, D = x.shape
        head_dim = self.embed_dim // self.num_heads
        # (B, T, P, D) → (B, P, T, D) → (B*P, T, D)
        xp = jnp.transpose(x, (0, 2, 1, 3)).reshape(B * P, T, D)
        residual = xp
        xp = nn.LayerNorm()(xp)
        q = nn.DenseGeneral((self.num_heads, head_dim), name='q')(xp)
        k = nn.DenseGeneral((self.num_heads, head_dim), name='k')(xp)
        v = nn.DenseGeneral((self.num_heads, head_dim), name='v')(xp)
        scale = jnp.sqrt(jnp.array(head_dim, dtype=xp.dtype))
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / scale
        # Lower-triangular causal mask: query at t can attend to keys 0..t
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        logits = jnp.where(causal[None, None], logits, jnp.finfo(logits.dtype).min)
        attn = nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(B * P, T, D)
        out = nn.Dense(self.embed_dim, name='proj')(out)
        xp = residual + out
        # (B*P, T, D) → (B, P, T, D) → (B, T, P, D)
        return jnp.transpose(xp.reshape(B, P, T, D), (0, 2, 1, 3))


class MemViTEncoder(nn.Module):
    """mem-ViT image encoder: ViT with interleaved causal temporal attention.

    Processes (B, T, H, W, C) uint8 frames (last frame = current) and returns
    (B, embed_dim) — the current frame's embedding with temporal context from
    all prior frames compressed via causal temporal attention layers.

    Architecture:
      1. Strided-conv patch embedding applied to all T frames: (B*T, P, embed_dim)
      2. Add learnable spatial positional embeddings
      3. Add sinusoidal temporal positional embeddings (position 0 = oldest frame)
      4. num_layers transformer blocks; every temporal_attention_every-th block
         (1-indexed) additionally applies causal temporal attention across frames
      5. Mean-pool patches of the last (current) frame → (B, embed_dim)
    """
    patch_size: int = 16
    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    temporal_attention_every: int = 4
    mlp_ratio: int = 4
    pool: bool = True  # if False, return (B, P, embed_dim) patch tokens instead of (B, embed_dim)

    @nn.compact
    def __call__(self, x, train: bool = True):
        del train
        B, T, H, W, C = x.shape
        x = x.astype(jnp.float32) / 255.0

        # 1. Patch embedding (applied to all frames jointly via batch reshape)
        x_flat = x.reshape(B * T, H, W, C)
        x_flat = nn.Conv(
            self.embed_dim,
            (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            kernel_init=nn.initializers.xavier_uniform(),
            name='patch_embed',
        )(x_flat)  # (B*T, H//ps, W//ps, embed_dim)
        P = (H // self.patch_size) * (W // self.patch_size)
        x_flat = x_flat.reshape(B * T, P, self.embed_dim)

        # 2. Learnable spatial positional embeddings
        spatial_pe = self.param(
            'spatial_pos_emb',
            nn.initializers.normal(stddev=0.02),
            (P, self.embed_dim),
        )
        x_flat = x_flat + spatial_pe[None]

        # 3. Sinusoidal temporal positional embeddings (position 0 = oldest, T-1 = current)
        half = self.embed_dim // 2
        t_pos = jnp.arange(T, dtype=jnp.float32)
        freq = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -(jnp.log(10000.0) / half))
        args = t_pos[:, None] * freq[None]  # (T, half)
        temporal_pe = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (T, embed_dim)

        x = x_flat.reshape(B, T, P, self.embed_dim)
        x = x + temporal_pe[None, :, None, :]  # broadcast over B and P dims

        # 4. Transformer blocks with interleaved temporal attention
        t_ptr = 0
        for i in range(self.num_layers):
            # Spatial block: flatten T into batch dimension
            x_2d = x.reshape(B * T, P, self.embed_dim)
            x_2d = _MemViTSpatialBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f'spatial_{i}',
            )(x_2d)
            x = x_2d.reshape(B, T, P, self.embed_dim)
            # Temporal block every temporal_attention_every layers
            if (i + 1) % self.temporal_attention_every == 0:
                x = _MemViTTemporalBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    name=f'temporal_{t_ptr}',
                )(x)
                t_ptr += 1

        x = nn.LayerNorm(name='final_ln')(x)

        # 5. Extract current frame (last time step); optionally mean-pool patches
        current = x[:, -1, :, :]  # (B, P, embed_dim)
        if self.pool:
            return current.mean(axis=1)  # (B, embed_dim)

        return current  # (B, P, embed_dim)


class MemViTObservationEncoder(nn.Module):
    """ObservationEncoder wrapping MemViTEncoder for temporal history compression.

    Takes (current_obs, history_obs) together, stacks frames in time, passes
    them through mem-ViT, and returns (compressed_enc, None).  The history
    context is fully absorbed into the current-frame representation; downstream
    modules (e.g. TransformerValue) should not use separate history tokens.

    Supports dict observations {'image'/'view*': ..., 'proprio': ...} and plain
    image arrays.  For dict obs, one MemViTEncoder call is made per camera view
    and the results are concatenated with proprio.

    When history_observations is None (e.g. actor-only calls), a single frame
    (T=1) is encoded — equivalent to a plain ViT with no temporal context.
    """
    mem_vit: Any = None  # MemViTEncoder instance

    def __call__(self, observations, history_observations=None):
        if _is_dict_obs(observations):
            return self._encode_dict(observations, history_observations)
        return self._encode_plain(observations, history_observations)

    def _encode_dict(self, obs, hist_obs):
        image_keys = sorted(k for k in obs.keys() if 'image' in k or 'view' in k)
        embs = []
        for img_key in image_keys:
            cur_img = obs[img_key]  # (B, H, W, C)
            if hist_obs is not None and _is_dict_obs(hist_obs):
                # hist_obs[img_key]: (B, T, H, W, C) — stack history before current
                frames = jnp.concatenate([hist_obs[img_key], cur_img[:, None]], axis=1)
            else:
                frames = cur_img[:, None]  # (B, 1, H, W, C)

            embs.append(self.mem_vit(frames))  # (B, embed_dim)
        proprio = obs['proprio'].astype(jnp.float32)
        return jnp.concatenate([*embs, proprio], axis=-1), None

    def _encode_plain(self, obs, hist_obs):
        if obs.ndim >= 3:
            # Plain image array (B, H, W, C)
            if hist_obs is not None and hist_obs.ndim >= 4:
                frames = jnp.concatenate([hist_obs, obs[:, None]], axis=1)
            else:
                frames = obs[:, None]  # (B, 1, H, W, C)
            return self.mem_vit(frames), None
        return obs, None  # flat vector: identity


def _wrap_mem_vit(vit_factory):
    """Wrap a MemViTEncoder factory in MemViTObservationEncoder."""
    def _make():
        return MemViTObservationEncoder(mem_vit=vit_factory())
    return _make


class MemViTTokenObservationEncoder(nn.Module):
    """mem-ViT encoder that returns patch tokens instead of a pooled vector.

    Returns (B, n_cams*P + 1, embed_dim), None for dict observations:
      - One (B, P, embed_dim) block per camera view (from MemViTEncoder with pool=False)
      - One proprio token (B, 1, embed_dim) projected from the proprio vector
    The downstream transformer (e.g. TransformerActorFlow) receives these tokens
    directly and attends over them.

    For plain image arrays: returns (B, P, embed_dim), None.
    History is always None — compressed into current-frame tokens via temporal attention.
    """
    mem_vit: Any = None  # MemViTEncoder with pool=False

    def setup(self):
        self.proprio_proj = nn.Dense(self.mem_vit.embed_dim)

    def __call__(self, observations, history_observations=None):
        if _is_dict_obs(observations):
            return self._encode_dict(observations, history_observations)
        return self._encode_plain(observations, history_observations)

    def _encode_dict(self, obs, hist_obs):
        image_keys = sorted(k for k in obs.keys() if 'image' in k or 'view' in k)
        patch_list = []
        for img_key in image_keys:
            cur_img = obs[img_key]  # (B, H, W, C)
            if hist_obs is not None and _is_dict_obs(hist_obs):
                frames = jnp.concatenate([hist_obs[img_key], cur_img[:, None]], axis=1)
            else:
                frames = cur_img[:, None]  # (B, 1, H, W, C)
            patch_list.append(self.mem_vit(frames))  # (B, P, embed_dim)
        # Project proprio to embed_dim and treat as an extra token (if present)
        if 'proprio' in obs:
            proprio = obs['proprio'].astype(jnp.float32)  # (B, D_p)
            proprio_tok = self.proprio_proj(proprio)[:, None, :]  # (B, 1, embed_dim)
            patch_list.append(proprio_tok)
        return jnp.concatenate(patch_list, axis=1), None

    def _encode_plain(self, obs, hist_obs):
        if obs.ndim >= 3:
            if hist_obs is not None and hist_obs.ndim >= 4:
                frames = jnp.concatenate([hist_obs, obs[:, None]], axis=1)
            else:
                frames = obs[:, None]
            return self.mem_vit(frames), None  # (B, P, embed_dim)
        return obs, None  # flat vector: identity


def _wrap_mem_vit_tokens(vit_factory):
    """Wrap a pool=False MemViTEncoder in MemViTTokenObservationEncoder."""
    def _make():
        return MemViTTokenObservationEncoder(mem_vit=vit_factory())
    return _make


class ProprioOnlyEncoder(nn.Module):
    """Encoder that uses only the proprio from dict observations, ignoring the image.

    For dict obs {'image': ..., 'proprio': (D,)}: returns proprio as-is.
    For plain arrays: identity (passed through unchanged).
    """
    mlp_hidden_dims: Sequence[int] = (256,)

    def setup(self):
        self.mlp = MLP(self.mlp_hidden_dims, activate_final=True)

    def _encode(self, obs):
        if _is_dict_obs(obs):
            return self.mlp(obs['proprio'].astype(jnp.float32))
        return obs  # flat vector: identity

    def __call__(self, observations, history_observations=None):
        obs_enc = self._encode(observations)
        if history_observations is not None:
            if _is_dict_obs(history_observations):
                # history_observations['proprio']: (..., T, D) — encode each step
                p = history_observations['proprio'].astype(jnp.float32)
                prefix, T, D = p.shape[:-2], p.shape[-2], p.shape[-1]
                flat = p.reshape(-1, D)
                enc = self.mlp(flat)
                hist_enc = enc.reshape(*prefix, T, enc.shape[-1])
            else:
                hist_enc = history_observations  # flat: identity
        else:
            hist_enc = None
        return obs_enc, hist_enc


encoder_modules = {
    'impala': _wrap(ImpalaEncoder),
    'impala_debug': _wrap(functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4))),
    'impala_small': _wrap(functools.partial(ImpalaEncoder, num_blocks=1)),
    'impala_large': _wrap(functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,))),
    # Lightweight encoder for 80×80 house-env images (2 stacks, 1 block, 256-dim output).
    'resnet': _wrap(functools.partial(ImpalaEncoder, stack_sizes=(16, 32), num_blocks=1, mlp_hidden_dims=(256,))),
    # Image+proprio encoder: ResNet on image, concatenated with proprio flat vector.
    'proprioencoder': _wrap(functools.partial(ImpalaEncoder, stack_sizes=(16, 32), num_blocks=1, mlp_hidden_dims=(256,))),
    # Proprio-only: ignores image entirely, runs proprio through a small MLP.
    'proprio_only': ProprioOnlyEncoder,
    # mem-ViT: ViT with interleaved causal temporal attention (https://arxiv.org/abs/2603.03596).
    # Processes history+current frames jointly; compresses temporal context into current-frame enc.
    # Output: (B, embed_dim) — downstream receives None for history_observations.
    # Default: 8 layers, embed_dim=256, patch_size=16, temporal block every 4 layers.
    'mem_vit': _wrap_mem_vit(functools.partial(MemViTEncoder)),
    # Smaller variant: 4 layers, embed_dim=128 — faster, lower memory.
    'mem_vit_small': _wrap_mem_vit(functools.partial(
        MemViTEncoder, num_layers=4, embed_dim=128, num_heads=4,
    )),
    # Larger variant: 12 layers, embed_dim=512, 8 heads — higher capacity.
    'mem_vit_large': _wrap_mem_vit(functools.partial(
        MemViTEncoder, num_layers=12, embed_dim=512, num_heads=8,
    )),
    # mem-ViT token variants: return (B, n_cams*P + 1, embed_dim) patch tokens (no pooling).
    # Use with TransformerActorFlow which attends over patch tokens directly.
    'mem_vit_tokens': _wrap_mem_vit_tokens(functools.partial(MemViTEncoder, pool=False)),
    'mem_vit_tokens_small': _wrap_mem_vit_tokens(functools.partial(
        MemViTEncoder, num_layers=4, embed_dim=128, num_heads=4, pool=False,
    )),
    'mem_vit_tokens_large': _wrap_mem_vit_tokens(functools.partial(
        MemViTEncoder, num_layers=12, embed_dim=512, num_heads=8, pool=False,
    )),
}
