from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Any
import math

import jax
import flax.linen as nn
from jax.nn import silu, sigmoid
import flax.linen.initializers as init
import jax.numpy as jnp
from chex import Array
from einops.layers.flax import Rearrange

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))

dtype = jnp.float32
conv_init = nn.initializers.variance_scaling(
    2.0, mode="fan_out", distribution="truncated_normal", dtype=dtype
)
dense_init = nn.initializers.variance_scaling(
    1.0 / 3, mode="fan_out", distribution="truncated_normal", dtype=dtype
)


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 384
    heads: int = 6
    num_registers: int = 4

    labels: int = 1_000

    patch_size: int = 14
    image_size: int = 518

    args: Any = None

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )

        self.wpe = self.param(
            "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
        )

    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        return x


def do_aggregate(attn_map, attn_aggregate):
    aggs = []

    if "cls_token" in attn_aggregate:
        attn_agg = attn_map[:, 0:1, :].mean(
            axis=1
        )  # CLS token is index 0, mean is not necessary
        aggs.append(attn_agg)

    if "reg_token" in attn_aggregate:
        attn_agg = attn_map[:, 1:5, :].mean(axis=1)  # Register tokens are indices 1-4
        aggs.append(attn_agg)

    if "patch_token" in attn_aggregate:
        attn_agg = attn_map[:, 5:1374, :].mean(
            axis=1
        )  # Patch tokens are the remaining 1369 indices
        aggs.append(attn_agg)

    aggs = jnp.stack(aggs, axis=1)  # (bs, num_aggregated_tokens, num_keys)
    return aggs.mean(axis=1)  # (bs, num_keys)


class Attention(ViTBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim))
        self.wk = DenseGeneral((self.heads, self.head_dim))
        self.wv = DenseGeneral((self.heads, self.head_dim))
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.num_prefix = self.num_registers + 1

    def __call__(
        self,
        x: Array,
        attn_aggregate=[],
        k_patches=None,
        keep_patch_indices=None,
        zoom_map_criterion="mse",
    ) -> Array:
        z = jnp.einsum("bqhd,bkhd->bhqk", self.wq(x) / self.head_dim**0.5, self.wk(x))
        attn_logits = z
        # Mask out the attention map for the patches that are not selected (the ones after the first `k_patches`)
        if k_patches is not None:
            # ex. [0...14] k=5, [15, 15, 15, 15, 15, 5, 6, 7, ..., 14]
            num_selected_patches = keep_patch_indices.shape[1]
            patch_indices = jnp.arange(num_selected_patches)
            indices_to_mask = (
                jnp.where(
                    patch_indices >= k_patches,
                    patch_indices,
                    num_selected_patches,  # Not masked
                )
                + self.num_prefix  # Don't mask prefix tokens
            )
            z = z.at[:, :, :, indices_to_mask].set(-jnp.inf)
        attn_map = nn.softmax(z)  # (1, num_heads, num_queries, num_keys)

        z = jnp.einsum("bhqk,bkhd->bqhd", attn_map, self.wv(x))
        output = self.wo(z)

        if len(attn_aggregate) == 0:
            return output

        # Grab the logits pre-masking, just incase
        attn_map = attn_map if zoom_map_criterion not in ["kl", "mse"] else attn_logits

        attn_map = jnp.mean(attn_map, axis=1)  # (bs, num_queries, num_keys)
        attn_map = do_aggregate(attn_map, attn_aggregate)  # (bs, num_keys)
        return output, attn_map


class FeedForward(ViTBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.dim)

    def __call__(self, x: Array) -> Array:
        return self.w2(nn.gelu(self.w1(x)))


class ViTLayer(ViTBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm(epsilon=1e-5)
        self.norm2 = nn.LayerNorm(epsilon=1e-5)

        self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
        self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))

    def __call__(
        self,
        x: Array,
        attn_aggregate=[],
        k_patches=None,
        keep_patch_indices=None,
        zoom_map_criterion="mse",
    ) -> Array:
        if len(attn_aggregate) == 0:
            # normal layer
            x = x + self.scale1 * self.attn(
                self.norm1(x),
                k_patches=k_patches,
                keep_patch_indices=keep_patch_indices,
                zoom_map_criterion=zoom_map_criterion,
            )
            x = x + self.scale2 * self.ff(self.norm2(x))
            return x
        else:
            # grab and return attn
            attn_out, attn_map = self.attn(
                self.norm1(x),
                attn_aggregate=attn_aggregate,
                k_patches=k_patches,
                keep_patch_indices=keep_patch_indices,
                zoom_map_criterion=zoom_map_criterion,
            )
            x = x + self.scale1 * attn_out
            x = x + self.scale2 * self.ff(self.norm2(x))
            return x, attn_map


def index_sequence(x, ids):
    return x[:, ids, ...]


class ViT(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)

        layer_fn = ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        self.cls_token = self.param(
            "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
        )
        self.reg_token = self.param(
            "reg_token", init.truncated_normal(0.02), (1, self.num_registers, self.dim)
        )
        self.num_prefix = self.num_registers + 1

        self.norm = nn.LayerNorm(epsilon=1e-5)
        self.head = Dense(self.labels)

    def __call__(
        self,
        x: Array,
        k_patches=None,
        keep_patch_indices=None,  # (bs, k)
        conditioning_tokens={},
        attn_aggregate=[],
        zoom_map_criterion="mse",
    ) -> Array:
        x = self.embed(x)  # (bs, num_patches, dim)
        bs = x.shape[0]

        if keep_patch_indices is not None:
            # keep_patch_indices: (bs, k)
            x = jnp.take_along_axis(
                x, keep_patch_indices[:, :, None], axis=1
            )  # (bs, k, dim)

        # Optionally initialize cls and reg tokens externally
        cls_token = conditioning_tokens.get(
            "cls_token", jnp.tile(self.cls_token, (bs, 1, 1))
        )  # (bs, 1, dim)
        reg_token = conditioning_tokens.get(
            "reg_token", jnp.tile(self.reg_token, (bs, 1, 1))
        )  # (bs, num_registers, dim)
        x = jnp.concatenate(
            (cls_token, reg_token, x), axis=1
        )  # (bs, num_prefix + num_patches, dim)

        attn_maps = []
        for layer in self.layer:
            if len(attn_aggregate) == 0:
                x = layer(
                    x,
                    k_patches=k_patches,
                    keep_patch_indices=keep_patch_indices,
                    zoom_map_criterion=zoom_map_criterion,
                )  # (bs, num_prefix + num_patches, dim)
            else:
                x, attn_map = layer(
                    x,
                    attn_aggregate=attn_aggregate,
                    k_patches=k_patches,
                    keep_patch_indices=keep_patch_indices,
                    zoom_map_criterion=zoom_map_criterion,
                )  # (bs, num_prefix + num_patches, dim), (bs, num_prefix + num_patches)
                attn_maps.append(attn_map)

        x = self.norm(x)  # (bs, num_prefix + num_patches, dim)
        cls_token = x[:, 0, :]  # (bs, dim)
        reg_token = x[:, 1 : self.num_prefix, :]  # (bs, num_registers, dim)
        patch_token = x[:, self.num_prefix :, :]  # (bs, num_patches, dim)
        logits = self.head(cls_token)  # (bs, num_classes)

        if len(attn_aggregate) == 0:
            return logits, cls_token, reg_token, patch_token
        else:
            attn_maps = jnp.stack(
                attn_maps, axis=1
            )  # (batch, num_layers, num_prefix + num_patches)
            match self.args.aggregate_layers:
                case "first6":
                    attn_maps = attn_maps[:, :6, :]
                case "all":
                    pass
                case "last":
                    attn_maps = attn_maps[:, -1:, :]
                case "last6":
                    attn_maps = attn_maps[:, -6:, :]
                case "last3":
                    attn_maps = attn_maps[:, -3:, :]
                case _:
                    raise ValueError(
                        f"Unknown aggregation type: {self.args.aggregate_layers}"
                    )

            attn_maps = jnp.mean(attn_maps, axis=1)  # (bs, num_prefix + num_patches)
            attn_maps = attn_maps[:, self.num_prefix :]  # (bs, num_patches)

            # normalize
            if zoom_map_criterion not in ["kl", "mse"]:
                min_vals = jnp.min(
                    attn_maps, axis=1, keepdims=True
                )  # (bs, num_patches)
                max_vals = jnp.max(
                    attn_maps, axis=1, keepdims=True
                )  # (bs, num_patches)
                attn_maps = (attn_maps - min_vals) / (
                    max_vals - min_vals + 1e-8
                )  # (bs, num_patches)

            return logits, cls_token, reg_token, patch_token, attn_maps


class SimpleZoomHead(nn.Module):
    hidden_dim: int
    num_output: int

    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.num_output)

    def __call__(self, x: Array) -> Array:
        return self.w2(nn.gelu(self.w1(x)))


class Zoomer(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)

        layer_fn = ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        self.cls_token = self.param(
            "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
        )
        self.reg_token = self.param(
            "reg_token", init.truncated_normal(0.02), (1, self.num_registers, self.dim)
        )
        self.num_prefix = self.num_registers + 1
        self.norm = nn.LayerNorm(epsilon=1e-5)

        # setup head
        input_grid_size = int(self.kwargs["image_size"] / self.kwargs["patch_size"])
        self.target_grid_size = self.args.target_grid_size
        resolution_multiplier = math.ceil(self.target_grid_size / input_grid_size)
        num_output = int(resolution_multiplier * resolution_multiplier)
        self.head = SimpleZoomHead(hidden_dim=int(4 * self.dim), num_output=num_output)

        self.unflatten_map = Rearrange(
            "b (h w) (i j) -> b (h i) (w j)",
            sizes={
                "h": input_grid_size,
                "w": input_grid_size,
                "i": resolution_multiplier,
                "j": resolution_multiplier,
            },
        )
        self.flatten_map = Rearrange("b h w -> b (h w)")

    def __call__(self, x: Array) -> Array:
        x = jax.image.resize(
            x,
            shape=(
                x.shape[0],
                self.kwargs["image_size"],
                self.kwargs["image_size"],
                x.shape[-1],
            ),
            method="bilinear",
        )
        x = self.embed(x)  # (bs, num_patches, dim)
        bs = x.shape[0]

        cls_token = jnp.tile(self.cls_token, (bs, 1, 1))  # (bs, 1, dim)
        reg_token = jnp.tile(self.reg_token, (bs, 1, 1))  # (bs, num_registers, dim)
        x = jnp.concatenate(
            (cls_token, reg_token, x), axis=1
        )  # (bs, num_prefix + num_patches, dim)

        for layer in self.layer:
            x = layer(x)  # (bs, num_prefix + num_patches, dim)

        x = self.norm(x)  # (bs, num_prefix + num_patches, dim)

        cls_token = x[:, 0, :]  # (bs, dim)
        reg_token = x[:, 1 : self.num_prefix, :]  # (bs, num_registers, dim)
        patch_token = x[:, self.num_prefix :, :]  # (bs, num_patches, dim)

        zoom_map = self.head(
            x[:, self.num_prefix :, :]
        )  # (bsz, num_patches, num_output)
        zoom_map = self.unflatten_map(
            zoom_map
        )  # (bsz, ~self.target_grid_size, ~self.target_grid_size)
        zoom_map = jax.image.resize(
            zoom_map,
            shape=(zoom_map.shape[0], self.target_grid_size, self.target_grid_size),
            method="bilinear",
        )
        zoom_map = self.flatten_map(zoom_map)  # (bsz, num_high-res_patches)

        return zoom_map, cls_token, reg_token, patch_token
