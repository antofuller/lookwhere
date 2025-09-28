from __future__ import annotations

import argparse
import json
import os
import re
import threading
from collections import defaultdict
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import webdataset as wds
from chex import Array, ArrayTree
from jax.tree_util import DictKey
import timm
import torch
from flax.traverse_util import unflatten_dict, flatten_dict


def timm_to_flax(timm_name):
    model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    state_dict = model.state_dict()
    num_heads = model.blocks[0].attn.num_heads

    cls_token = state_dict["cls_token"]
    reg_token = state_dict["reg_token"]

    pos_embed = state_dict["pos_embed"].squeeze(0)
    pos_embed = pos_embed.unflatten(0, (int(pos_embed.size(0) ** 0.5), -1))
    wte = state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0)

    params = {
        "cls_token": cls_token.numpy(),
        "reg_token": reg_token.numpy(),
        "embed": {
            "wpe": pos_embed.numpy(),
            "wte": {
                "kernel": wte.numpy(),
                "bias": state_dict["patch_embed.proj.bias"].numpy(),
            },
        },
        "norm": {
            "scale": state_dict["norm.weight"].numpy(),
            "bias": state_dict["norm.bias"].numpy(),
        },
    }

    # Create layer params
    layer_params = {}
    layer_idx = 0
    while f"blocks.{layer_idx}.norm1.weight" in state_dict:
        wqkv = state_dict[f"blocks.{layer_idx}.attn.qkv.weight"].transpose(1, 0)
        wq, wk, wv = wqkv.unflatten(1, (3, num_heads, -1)).permute(1, 0, 2, 3)
        wo = state_dict[f"blocks.{layer_idx}.attn.proj.weight"].transpose(1, 0)
        wo = wo.unflatten(0, (num_heads, -1))

        bqkv = state_dict[f"blocks.{layer_idx}.attn.qkv.bias"]
        bq, bk, bv = bqkv.view(3, num_heads, -1)
        bo = state_dict[f"blocks.{layer_idx}.attn.proj.bias"]

        wfc1 = state_dict[f"blocks.{layer_idx}.mlp.fc1.weight"].transpose(1, 0)
        wfc2 = state_dict[f"blocks.{layer_idx}.mlp.fc2.weight"].transpose(1, 0)
        bfc1 = state_dict[f"blocks.{layer_idx}.mlp.fc1.bias"]
        bfc2 = state_dict[f"blocks.{layer_idx}.mlp.fc2.bias"]

        snorm1 = state_dict[f"blocks.{layer_idx}.norm1.weight"]
        snorm2 = state_dict[f"blocks.{layer_idx}.norm2.weight"]
        bnorm1 = state_dict[f"blocks.{layer_idx}.norm1.bias"]
        bnorm2 = state_dict[f"blocks.{layer_idx}.norm2.bias"]

        layer_param = {
            "attn": {
                "wq": {"kernel": wq.numpy(), "bias": bq.numpy()},
                "wk": {"kernel": wk.numpy(), "bias": bk.numpy()},
                "wv": {"kernel": wv.numpy(), "bias": bv.numpy()},
                "wo": {"kernel": wo.numpy(), "bias": bo.numpy()},
            },
            "ff": {
                "w1": {"kernel": wfc1.numpy(), "bias": bfc1.numpy()},
                "w2": {"kernel": wfc2.numpy(), "bias": bfc2.numpy()},
            },
            "norm1": {"scale": snorm1.numpy(), "bias": bnorm1.numpy()},
            "norm2": {"scale": snorm2.numpy(), "bias": bnorm2.numpy()},
        }

        layer_param["scale1"] = state_dict[f"blocks.{layer_idx}.ls1.gamma"].numpy()
        layer_param["scale2"] = state_dict[f"blocks.{layer_idx}.ls2.gamma"].numpy()

        layer_params[f"layer_{layer_idx}"] = layer_param
        layer_idx += 1

    # Merge all params
    params.update(layer_params)
    return params


def get_franca_flax():
    model = torch.hub.load('valeoai/Franca', 'franca_vitb14')
    state_dict = model.state_dict()
    num_heads = 12

    cls_token = state_dict["cls_token"] # (1, 1, 768)
    cls_token = cls_token + state_dict["pos_embed"][:, 0:1, :] # (1, 1, 768)

    pos_embed = state_dict["pos_embed"].squeeze(0)[1:, :]
    pos_embed = pos_embed.unflatten(0, (int(pos_embed.size(0) ** 0.5), -1))
    wte = state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0)

    params = {
        "cls_token": cls_token.numpy(),
        "embed": {
            "wpe": pos_embed.numpy(),
            "wte": {
                "kernel": wte.numpy(),
                "bias": state_dict["patch_embed.proj.bias"].numpy(),
            },
        },
        "norm": {
            "scale": state_dict["norm.weight"].numpy(),
            "bias": state_dict["norm.bias"].numpy(),
        },
    }

    # Create layer params
    layer_params = {}
    layer_idx = 0
    while f"blocks.{layer_idx}.norm1.weight" in state_dict:
        wqkv = state_dict[f"blocks.{layer_idx}.attn.qkv.weight"].transpose(1, 0)
        wq, wk, wv = wqkv.unflatten(1, (3, num_heads, -1)).permute(1, 0, 2, 3)
        wo = state_dict[f"blocks.{layer_idx}.attn.proj.weight"].transpose(1, 0)
        wo = wo.unflatten(0, (num_heads, -1))

        bqkv = state_dict[f"blocks.{layer_idx}.attn.qkv.bias"]
        bq, bk, bv = bqkv.view(3, num_heads, -1)
        bo = state_dict[f"blocks.{layer_idx}.attn.proj.bias"]

        wfc1 = state_dict[f"blocks.{layer_idx}.mlp.w12.weight"].transpose(1, 0)
        wfc2 = state_dict[f"blocks.{layer_idx}.mlp.w3.weight"].transpose(1, 0)
        bfc1 = state_dict[f"blocks.{layer_idx}.mlp.w12.bias"]
        bfc2 = state_dict[f"blocks.{layer_idx}.mlp.w3.bias"]

        snorm1 = state_dict[f"blocks.{layer_idx}.norm1.weight"]
        snorm2 = state_dict[f"blocks.{layer_idx}.norm2.weight"]
        bnorm1 = state_dict[f"blocks.{layer_idx}.norm1.bias"]
        bnorm2 = state_dict[f"blocks.{layer_idx}.norm2.bias"]

        layer_param = {
            "attn": {
                "wq": {"kernel": wq.numpy(), "bias": bq.numpy()},
                "wk": {"kernel": wk.numpy(), "bias": bk.numpy()},
                "wv": {"kernel": wv.numpy(), "bias": bv.numpy()},
                "wo": {"kernel": wo.numpy(), "bias": bo.numpy()},
            },
            "ff": {
                "w12": {"kernel": wfc1.numpy(), "bias": bfc1.numpy()},
                "w3": {"kernel": wfc2.numpy(), "bias": bfc2.numpy()},
            },
            "norm1": {"scale": snorm1.numpy(), "bias": bnorm1.numpy()},
            "norm2": {"scale": snorm2.numpy(), "bias": bnorm2.numpy()},
        }

        layer_param["scale1"] = state_dict[f"blocks.{layer_idx}.ls1.gamma"].numpy()
        layer_param["scale2"] = state_dict[f"blocks.{layer_idx}.ls2.gamma"].numpy()

        layer_params[f"layer_{layer_idx}"] = layer_param
        layer_idx += 1

    # Merge all params
    params.update(layer_params)
    return params


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def summary(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {
            f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v)
            for k, v in buffer.items()
        }


def save_checkpoint_in_background(
    args: argparse.Namespace,
    params_bytes: bytes,
    postfix: str = "last",
    safe_run_name: str = None,
) -> threading.Thread:

    run_name_to_use = safe_run_name if safe_run_name is not None else args.name
    if safe_run_name is None:
        print(
            f"[Thread Saving Checkpoint] Warning: safe_run_name not provided, using potentially unsafe args.name: {args.name}"
        )

    filename = os.path.join(args.output_dir, f"{run_name_to_use}-{postfix}.msgpack")

    def thread_fn():
        print(
            f"[Thread Saving Checkpoint] Attempting to save parameters to: {filename}"
        )
        try:
            with wds.gopen(filename, "wb") as fp:
                fp.write(params_bytes)
            print(
                f"[Thread Saving Checkpoint] Successfully saved parameters to: {filename}"
            )
        except Exception as e:
            print(
                f"[Thread Saving Checkpoint] ERROR saving parameters to {filename}: {e}"
            )

    thread = threading.Thread(target=thread_fn)
    thread.start()
    return thread


class Mixup(nn.Module):
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0

    def apply_mixup(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.mixup_alpha,) * 2)
        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = ratio * images + (1 - ratio) * images[randperm]
        labels = ratio * labels + (1 - ratio) * labels[randperm]
        return images, labels

    def apply_cutmix(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        image_mask = self.random_bounding_box(ratio, images.shape[2], images.shape[1])
        label_mask = image_mask.mean((1, 2))

        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = image_mask * images + (1 - image_mask) * images[randperm]
        labels = label_mask * labels + (1 - label_mask) * labels[randperm]
        return images, labels

    def random_bounding_box(self, ratio: Array, width: int, height: int) -> Array:
        size = (1 - ratio) ** 0.5
        xstart, ystart = jax.random.uniform(self.make_rng("mixup"), (2,))
        xrange, yrange = jnp.linspace(0, 1, width), jnp.linspace(0, 1, height)

        xmask = (xstart - 0.5 * size <= xrange) & (xrange < xstart + 0.5 * size)
        ymask = (ystart - 0.5 * size <= yrange) & (yrange < ystart + 0.5 * size)
        return ~(xmask[None, None, :, None] & ymask[None, :, None, None])

    def __call__(self, images: Array, labels: Array) -> tuple[Array, Array]:
        if self.mixup_alpha == 0 and self.cutmix_alpha == 0:
            return images, labels
        if self.mixup_alpha > 0 and self.cutmix_alpha == 0:
            return self.apply_mixup(images, labels)
        if self.mixup_alpha == 0 and self.cutmix_alpha > 0:
            return self.apply_cutmix(images, labels)

        # If both mixup and cutmix are enabled, only one operation will be selected and
        # applied. Since jax does not support conditional branching on JIT, mixup and
        # cutmix are performed first and only one output will be selected.
        images1, labels1 = self.apply_mixup(images, labels)
        images2, labels2 = self.apply_cutmix(images, labels)

        cond = jax.random.uniform(self.make_rng("mixup")) > 0.5
        return jnp.where(cond, images1, images2), jnp.where(cond, labels1, labels2)


def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> Array:
    freqs = 1 / (10000 ** jnp.linspace(0, 1, dim // 4))
    x = jnp.outer(jnp.arange(0, nrows, dtype=jnp.float32), freqs)
    y = jnp.outer(jnp.arange(0, ncols, dtype=jnp.float32), freqs)

    x = jnp.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = jnp.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=2)


def modified_lamb(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mask: optax.MaskOrFn = None,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay=weight_decay, mask=mask),
        optax.masked(optax.scale_by_trust_ratio(), mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def get_layer_index_fn(path: tuple[DictKey, ...], _: Any, num_layers: int = 12) -> int: # type: ignore
    if path[0].key == "model" and path[1].key.startswith("layer_"):
        return int(re.match(r"layer_(\d+)", path[1].key).group(1)) + 1
    if path[0].key == "model" and path[1].key == "embed":
        return 0
    return num_layers


def load_pretrained_params(args: argparse.Namespace, params: ArrayTree) -> ArrayTree:
    with wds.gopen(args.pretrained_ckpt) as fp:
        new_params = flax.serialization.msgpack_restore(fp.read())

    if (
        args.posemb == "learnable"
        and new_params["model"]["embed"]["wpe"].shape
        != params["model"]["embed"]["wpe"].shape
    ):
        new_params["model"]["embed"]["wpe"] = jax.image.resize(
            new_params["model"]["embed"]["wpe"],
            params["model"]["embed"]["wpe"].shape,
            method="bicubic",
        )

    if (
        "head" not in new_params["model"]
        or args.label_mapping is None
        and new_params["model"]["head"]["kernel"].shape
        != params["model"]["head"]["kernel"].shape
    ):
        new_params["model"]["head"] = params["model"]["head"]

    if args.label_mapping:
        with wds.gopen(args.label_mapping) as fp:
            label_mapping = json.load(fp)
            src, dst = label_mapping["src"], label_mapping["dst"]

        kernel = np.zeros_like(params["model"]["head"]["kernel"])
        kernel[:, dst] = new_params["model"]["head"]["kernel"][:, src]

        bias = np.full_like(params["model"]["head"]["bias"], fill_value=-10.0)
        bias[dst] = new_params["model"]["head"]["bias"][src]

        new_params["model"]["head"] = {"kernel": kernel, "bias": bias}
    return new_params


def upsample_grid_nn(
    all_keep_ids, all_keep_values, grid_size, K=1, distance_power=1, masked_ids=None
):
    batch_size, n_keep = all_keep_ids.shape
    channels = all_keep_values.shape[-1]

    rows = jnp.arange(grid_size)
    cols = jnp.arange(grid_size)
    full_coords = jnp.stack(jnp.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(
        -1, 2
    )

    def get_known_coords(ids):
        r = ids // grid_size
        c = ids % grid_size
        return jnp.stack([r, c], axis=-1)

    known_coords = jax.vmap(get_known_coords)(all_keep_ids)

    def fill_sample(known_coords_sample, keep_values_sample, masked_ids):
        dists = jnp.sqrt(
            jnp.sum(
                (full_coords[:, None, :] - known_coords_sample[None, :, :]) ** 2,
                axis=-1,
            )
        )

        if distance_power != 1:
            dists = jnp.power(dists, distance_power)

        if masked_ids is not None:
            dists = dists.at[:, masked_ids].set(jnp.inf)

        sorted_indices = jnp.argsort(dists, axis=-1)[:, :K]
        nearest_dists = jnp.take_along_axis(
            dists, sorted_indices, axis=-1
        )

        epsilon = 1e-8
        weights = 1.0 / (nearest_dists + epsilon)
        weights = weights / jnp.sum(
            weights, axis=-1, keepdims=True
        )

        neighbor_values = keep_values_sample[sorted_indices]
        filled = jnp.sum(
            neighbor_values * weights[..., None], axis=-2
        )
        return filled

    filled_full = jax.vmap(fill_sample)(
        known_coords, all_keep_values, masked_ids
    )
    filled_full = filled_full.reshape(batch_size, grid_size, grid_size, channels)
    return filled_full
