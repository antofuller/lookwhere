from __future__ import annotations

import argparse
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from chex import Array, ArrayTree

from dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from modeling import ViT, Zoomer
from training_util import CRITERION_COLLECTION, patch_selection, upsample_patch_embeds


class TrainModule(nn.Module):
    args: argparse.Namespace
    student: ViT
    zoomer: Zoomer
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]
    zoom_map_criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["bce"]

    def compute_distillation_loss(
        self,
        cls_embeds,
        reg_embeds,
        patch_embeds,
        teacher_cls_embeds,
        teacher_reg_embeds,
        teacher_patch_embeds,
        zoom_map,
        attn_maps,
        keep_patch_indices,
    ):
        """Computes the distillation loss."""
        loss = 0
        metrics = {}
        for key, weight in self.args.distillation_losses.items():
            if key == "cls":
                loss_item = self.criterion(cls_embeds, teacher_cls_embeds)
            elif key == "reg":
                loss_item = self.criterion(reg_embeds, teacher_reg_embeds).mean(-1)
            elif key == "patch":
                if self.args.patch_distill == "one-to-one":
                    # Extract the kept patches from the teacher.
                    # patch_embeds are (bs, num_keep_patches, dim)
                    kept_teacher_patch_embeds = jnp.take_along_axis(
                        teacher_patch_embeds,
                        keep_patch_indices[:, :, None],
                        axis=1,
                    )  # (bs, num_keep_patches, dim)
                    loss_item = self.criterion(
                        patch_embeds, kept_teacher_patch_embeds
                    ).mean(-1)
                else:
                    loss_item = self.criterion(patch_embeds, teacher_patch_embeds).mean(
                        -1
                    )
            elif key == "map":
                loss_item = self.zoom_map_criterion(zoom_map, attn_maps)
            else:
                raise ValueError(f"Unknown distillation loss type: {key}")
            loss += weight * loss_item
            metrics[f"loss_{key}_distill"] = loss_item
        return loss, metrics

    def __call__(
        self,
        images: Array,
        labels: Array,
        k_patches: Array,
        teacher_params: ArrayTree,
        return_embeds=False,
    ) -> ArrayTree:
        # Preprocess images.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        images = (images - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

        # Forward pass for teacher using its parameters.
        _, teacher_cls_embeds, teacher_reg_embeds, teacher_patch_embeds, attn_maps = (
            self.student.apply(
                {"params": teacher_params},
                images,
                attn_aggregate=self.args.attn_aggregate,
                zoom_map_criterion=self.args.zoom_map_criterion,
            )
        )  # (bs, dim), (bs, num_registers, dim), (bs, num_patches, dim)

        # Forward pass for zoomer.
        zoom_map, zoom_cls_token, zoom_reg_token, _patch_token = self.zoomer(
            images
        )  # (bs, num_patches * num_patches), (bs, dim), (bs, num_registers, dim), (bs, num_low_res_patches, dim)

        if self.args.get("reg_pool", None) is not None:
            # Reshape patch tokens to 11x11
            selector_patches = _patch_token.reshape(
                (images.shape[0], 11, 11, -1)
            )
            # Reshape teacher tokens to 37x37
            teacher_patches = teacher_patch_embeds.reshape(
                (images.shape[0], 37, 37, -1)
            )

            selector_patches_pool = nn.avg_pool(
                selector_patches,
                window_shape=(5, 5),
                strides=(5, 5),
                padding="VALID"
            ).reshape(
                (images.shape[0], -1, selector_patches.shape[-1]))
            teacher_patches_pool = nn.avg_pool(
                teacher_patches,
                window_shape=(18, 18),
                strides=(18, 18),
                padding="VALID"
            ).reshape(
                (images.shape[0], -1, teacher_patches.shape[-1]))

        if self.args.get("reg_pool", None) is not None:
            assert self.args.conditioning_tokens == ["reg_token"]
            if self.args.reg_pool == "selector":
                zoom_reg_token = selector_patches_pool
            elif self.args.reg_pool == "teacher":
                zoom_reg_token = teacher_patches_pool
            else:
                raise ValueError(f"Unknown reg_pool type: {self.args.reg_pool}")

        # Optionally extract a subset of the zoomer's outputs to condition the student.
        if len(self.args.conditioning_tokens):
            zoom_cls_token = jnp.expand_dims(zoom_cls_token, axis=1)  # (bs, 1, dim)
            conditioning_tokens = {
                "cls_token": zoom_cls_token,
                "reg_token": zoom_reg_token,
            }
            conditioning_tokens = {
                key: conditioning_tokens[key] for key in self.args.conditioning_tokens
            }
        else:
            conditioning_tokens = {}

        # Select the patches to zoom into.
        rng = self.make_rng("patch_selection")
        keep_patch_indices = patch_selection(
            self.args,
            rng,
            zoom_map,
            attn_maps,
        )  # shape: (bs, k)

        # Forward pass for student.
        _, cls_embeds, reg_embeds, patch_embeds = self.student(
            images,
            k_patches=k_patches,
            keep_patch_indices=keep_patch_indices,
            conditioning_tokens=conditioning_tokens,
        )  # (bs, dim), (bs, num_registers, dim), (bs, num_keep_patches, dim)

        if self.args.patch_distill == "upsample":
            # Get indices to mask in upsampling. The first k are set to `num_patches` (invalid index, so unmasked).
            # The remaining ones pass their actual indices to mask.
            num_selected_patches = keep_patch_indices.shape[1]
            patch_indices = jnp.arange(num_selected_patches)
            indices_to_mask = jnp.where(
                patch_indices >= k_patches,
                patch_indices,
                num_selected_patches,  # Not masked
            )[None, :].repeat(zoom_map.shape[0], axis=0)
            patch_embeds = upsample_patch_embeds(
                self.args, keep_patch_indices, patch_embeds, masked_ids=indices_to_mask
            )  # (bs, 37*37, dim)

        sampled_loss, sampled_metrics = self.compute_distillation_loss(
            cls_embeds,
            reg_embeds,
            patch_embeds,
            teacher_cls_embeds,
            teacher_reg_embeds,
            teacher_patch_embeds,
            zoom_map,
            attn_maps,
            keep_patch_indices,
        )

        sampled_metrics["loss"] = sampled_loss

        if return_embeds:
            sampled_metrics["cls_embeds"] = cls_embeds

        return sampled_metrics
