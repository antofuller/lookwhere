import math
import torch.nn as nn
import torch

def upsample_grid_nn(all_keep_ids, all_keep_values, grid_size, K=5, distance_power=1):
    batch_size, n_keep = all_keep_ids.shape
    channels = all_keep_values.shape[-1]

    device = all_keep_ids.device
    rows = torch.arange(grid_size, device=device)
    cols = torch.arange(grid_size, device=device)

    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
    full_coords = torch.stack((grid_rows, grid_cols), dim=-1).view(-1, 2)
    total_patches = full_coords.shape[0]

    r = all_keep_ids // grid_size
    c = all_keep_ids % grid_size
    known_coords = torch.stack([r, c], dim=-1)

    full_coords_exp = full_coords.unsqueeze(0).unsqueeze(2)
    known_coords_exp = known_coords.unsqueeze(1)
    
    diff = full_coords_exp - known_coords_exp
    dists = torch.sqrt(torch.sum(diff ** 2, dim=-1))
    
    if distance_power != 1:
        dists = dists ** distance_power

    sorted_indices = torch.argsort(dists, dim=-1)[:, :, :K]
    nearest_dists = torch.gather(dists, dim=-1, index=sorted_indices)
    
    epsilon = 1e-8
    weights = 1.0 / (nearest_dists + epsilon)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(-1, total_patches, K)
    neighbor_values = all_keep_values[batch_indices, sorted_indices]
    
    weights_exp = weights.unsqueeze(-1)
    filled = torch.sum(neighbor_values * weights_exp, dim=2)

    filled_full = filled.view(batch_size, grid_size, grid_size, channels)
    return filled_full


def interpolate_pos_encoding(patch_pos_embed: torch.Tensor, height: int, width: int, patch_size: int = 14) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_positions = patch_pos_embed.shape[1]
        dim = patch_pos_embed.shape[-1]

        new_height = height // patch_size
        new_width = width // patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed


def adjust_learning_rate(step, sched_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < sched_config["warmup_steps"]:
        lr = sched_config["max_lr"] * step / sched_config["warmup_steps"]
    else:
        lr = sched_config["min_lr"] + (
            sched_config["max_lr"] - sched_config["min_lr"]
        ) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (step - sched_config["warmup_steps"])
                / (sched_config["total_steps"] - sched_config["warmup_steps"])
            )
        )
    return lr


def get_learning_rates(optimizer):
    lr_dict = {}
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "default")
        lr_dict[group_name] = param_group["lr"]
    return lr_dict
