import jax
import optax
import jax.numpy as jnp
from utils import (
    modified_lamb,
    upsample_grid_nn,
)

def cosine_similarity_loss(x, y):
    # Unit norm the inputs
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
    # Compute cosine similarity
    cos_sim = jnp.sum(x * y, axis=-1)
    return 1 - cos_sim

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
    "mse": lambda x, y: optax.squared_error(x, y).mean(-1),
    "cosine": lambda x, y: cosine_similarity_loss(x, y).mean(-1),
    "kl": lambda x, y: optax.losses.kl_divergence_with_log_targets(
        # Normalize the logits to enforce probability distribution
        jax.nn.log_softmax(x),
        jax.nn.log_softmax(y),
    ).mean(-1),
}

OPTIMIZER_COLLECTION = {
    "adamw": optax.adamw,
    "lamb": modified_lamb,
}


def patch_selection(
    args, rng, zoom_map, attn_maps
):
    """Returns the indices of the patches to keep (optionally conditioned on the zoom_map)."""
    bs, num_patches = zoom_map.shape
    top_k_size = args.top_k_range.max if args.top_k_range is not None else args.top_k

    patch_selection_method = args.patch_selection_method
    match patch_selection_method:
        case "random":
            # Randomly select args.top_k patches to keep without replacement.
            keep_patch_indices = jax.vmap(
                lambda rng_i: jax.random.choice(
                    rng_i, num_patches, shape=(top_k_size,), replace=False
                )
            )(jax.random.split(rng, bs))  # uniformly sample k indices

        case "topk-zoomer":
            # Select the top args.top_k patches based on zoom_map.
            keep_patch_indices = jnp.argsort(zoom_map, axis=1)[
                :, -top_k_size:
            ]  # (bs, k)
            keep_patch_indices = keep_patch_indices[:, ::-1]  # (bs, k)

        case "topk-oracle":
            # Select the top args.top_k patches based on zoom_map.
            keep_patch_indices = jnp.argsort(attn_maps, axis=1)[
                :, -top_k_size:
            ]  # (bs, k)
            keep_patch_indices = keep_patch_indices[:, ::-1]  # (bs, k)

        case _:
            raise ValueError(f"Unknown patch selection type: {type}")
    return keep_patch_indices  # (bs, k)


def upsample_patch_embeds(
    args, keep_patch_indices, student_patch_embeds, masked_ids=None
):
    if args.upsample_features.type == "NN":
        # needs to be formatted "NN-K"
        bs, _, dim = student_patch_embeds.shape

        student_patch_embeds = upsample_grid_nn(
            all_keep_ids=keep_patch_indices,
            all_keep_values=student_patch_embeds,
            grid_size=int(args.image_size / args.patch_size),  # hardcode for DINO-V2
            K=args.upsample_features.K,
            distance_power=args.upsample_features.distance_power,
            masked_ids=masked_ids,
        ).reshape(
            bs, -1, dim
        )  # same reshape method as PatchEmbed: shape (bs, 37*37, dim)
        return student_patch_embeds

    else:
        raise ValueError(f"Unknown upsample features type: {args.upsample_features}")
