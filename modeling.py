import torch.nn as nn
import torch
import timm
from einops import rearrange
import math
import torch.nn.functional as F
import types
from timm.layers.patch_embed import resample_patch_embed
from timm.models.vision_transformer import _create_vision_transformer
from timm.layers import SwiGLUPacked

from utils import interpolate_pos_encoding, upsample_grid_nn
    

class Extractor(nn.Module):
    def __init__(self, pretrained_params, lw_type, img_size, device):
        super().__init__()
        assert lw_type in ["franca", "dinov2"]
        patch_size = 14
        self.lw_type = lw_type

        if lw_type == "dinov2":
            self.model = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                num_classes=0,
                pretrained=False,
                img_size=img_size,
                patch_size=patch_size,
            )
        elif lw_type == "franca":
            embed_dim = 768
            num_heads = 12
            depth = 12
            model_args = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, init_values=1e-5, mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, img_size=img_size)
            self.model = _create_vision_transformer('vit_base_patch14_dinov2', pretrained=False, **dict(model_args))
            num_patches = (img_size // patch_size) ** 2
            self.model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        if img_size != 518:
            # we pre-trained at high_res=518x518, we thus interpolate position encoding when using other resolutions
            pretrained_params["pos_embed"] = interpolate_pos_encoding(
                patch_pos_embed=pretrained_params["pos_embed"],
                height=img_size,
                width=img_size,
                patch_size=patch_size
            )

        self.model.load_state_dict(pretrained_params)
        self.grid_size = img_size // patch_size
        self.to(device)

    def forward(self, x, selector_prefix_tokens, keep_patch_indices, return_only_cls=False):
        x = self.model.patch_embed(x)  # (bs, num_patches, dim)
        x = x + self.model.pos_embed  # (bs, num_patches, dim)

        # keep_patch_indices: (bs, k)
        batch_range = torch.arange(x.shape[0], device = x.device)[:, None]
        x = x[batch_range, keep_patch_indices] # (bs, k, dim)
                
        num_prefix = selector_prefix_tokens.shape[1]

        x = torch.cat([
            selector_prefix_tokens,
            x
        ], dim=1)  # (bs, num_prefix + num_patches, dim)

        x = self.model.blocks(x)
        x = self.model.norm(x)

        if return_only_cls:
            return x[:, 0, :]
        
        x_patches = x[:, num_prefix:, :]  # (bs, num_patches, dim)
        return upsample_grid_nn(
            all_keep_ids=keep_patch_indices,
            all_keep_values=x_patches,
            grid_size=self.grid_size,
        )


class SelectorHead(nn.Module):
    def __init__(
        self,
        dim,
        num_output,
    ):
        super().__init__()
        hidden_dim = int(4*dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, num_output)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Selector(nn.Module):
    def __init__(self, pretrained_params, lw_type, hr_size, device):
        super().__init__()
        assert lw_type in ["franca", "dinov2"]
        depth = 3  # our selector is shallow / fast!
        patch_size = 14
        img_size = 154  # hard-coded because we don't fine-tune the selector thus it will be bad if we use other img_size
        self.lw_type = lw_type

        if lw_type == "dinov2":
            self.model = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                depth=depth,
                num_classes=0,
                pretrained=False,
                img_size=img_size,
            )
        elif lw_type == "franca":
            embed_dim = 768
            num_heads = 12
            num_patches = (img_size // patch_size) **2
            model_args = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, init_values=1e-5, mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, img_size=img_size)
            self.model = _create_vision_transformer('vit_base_patch14_dinov2', pretrained=False, **dict(model_args))
            self.model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.model.load_state_dict(pretrained_params["backbone"])

        self.img_size = img_size
        self.input_grid_size = int(self.img_size / patch_size)
        self.target_grid_size = int(hr_size / patch_size)
        self.resolution_multiplier = math.ceil(37 / self.input_grid_size)  # (518/14)^2=37
        num_output = int(self.resolution_multiplier * self.resolution_multiplier)

        self.head = SelectorHead(dim=self.model.embed_dim, num_output=num_output)
        self.head.load_state_dict(pretrained_params["head"])
        
        self.to(device)
    
    def forward(self, x):
        # x is shape (bs, C, high_res_H, high_res_W)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.model.patch_embed(x)  # (bs, num_patches, dim)
        x = x + self.model.pos_embed

        if self.lw_type == "dinov2":
            x_prefix = torch.cat([
                self.model.cls_token.expand(x.shape[0], -1, -1),
                self.model.reg_token.expand(x.shape[0], -1, -1),
            ], dim=1)  # (bs, 1 + num_registers, dim)
        elif self.lw_type == "franca":
            x_prefix = self.model.cls_token.expand(x.shape[0], -1, -1)  # (bs, 1, dim)

        num_prefix = x_prefix.shape[1]

        x = torch.cat([
            x_prefix,
            x
        ], dim=1)  # (bs, num_prefix + num_patches, dim)

        x = self.model.blocks(x)
        x = self.model.norm(x)

        prefix_tokens = x[:, :num_prefix, :]  # (bs, num_prefix, dim)
        patch_tokens = x[:, num_prefix:, :]  # (bs, num_patches, dim)

        selector_map = self.head(patch_tokens)  # (bs, num_patches, num_output)
        selector_map = rearrange(
            selector_map,
            "b (h w) (i j) -> b 1 (h i) (w j)",
            h=self.input_grid_size,
            w=self.input_grid_size,
            i=self.resolution_multiplier,
            j=self.resolution_multiplier
        )
        selector_map = F.interpolate(selector_map, size=(self.target_grid_size, self.target_grid_size), mode='bilinear', align_corners=False)
        selector_map = rearrange(selector_map, "b 1 h w -> b (h w)")
        
        return {
            "selector_map": selector_map,
            "prefix_tokens": prefix_tokens,
            "patch_tokens": patch_tokens
        }


class LookWhereDownstream(nn.Module):
    def __init__(self, pretrained_params_path, high_res_size, num_classes, k, is_cls, device):
        super().__init__()
        # supports classification (1 prediction per image) and segmentation (1 prediction per patch)

        last_part_of_path = pretrained_params_path.split("/")[-1]
        assert last_part_of_path in ["lookwhere_dinov2.pt", "lookwhere_franca.pt"]
        self.lw_type = last_part_of_path.split("_")[-1].replace(".pt", "")  # either dinov2 or franca
        print(f"Using LookWhere type: {self.lw_type}")

        all_pretrained_params = torch.load(pretrained_params_path, map_location="cpu", weights_only=True)

        self.selector = Selector(
            pretrained_params=all_pretrained_params["selector"],
            lw_type=self.lw_type,
            hr_size=high_res_size,
            device=device
        )
        self.extractor = Extractor(
            pretrained_params=all_pretrained_params["extractor"],
            lw_type=self.lw_type,
            img_size=high_res_size,
            device=device
        )
        self.k = k
        self.is_cls = is_cls
        if num_classes == 0:
            # just extract features
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.selector.model.embed_dim, num_classes).to(device)
        
        self.high_res_grid_size = high_res_size // 14  # patch size is 14

    def forward(self, images, k=None):
        with torch.no_grad():
            # we do not fine-tune the selector
            selector_dict = self.selector(images)

        k = k if k is not None else self.k

        keep_patch_indices = torch.topk(selector_dict["selector_map"], k=k, sorted=True).indices

        if self.is_cls:
            x_cls = self.extractor(
                x=images,
                selector_prefix_tokens=selector_dict["prefix_tokens"],
                keep_patch_indices=keep_patch_indices,
                return_only_cls=True
            )
            return self.head(x_cls)
        else:
            x_patches = self.extractor(
                x=images,
                selector_prefix_tokens=selector_dict["prefix_tokens"],
                keep_patch_indices=keep_patch_indices,
                return_only_cls=False
            )
            x_patches = rearrange(x_patches, "b h w c -> b (h w) c")
            logits = self.head(x_patches)
            logits = rearrange(logits, "b (h w) c -> b c h w", h=self.high_res_grid_size, w=self.high_res_grid_size)
            return logits
