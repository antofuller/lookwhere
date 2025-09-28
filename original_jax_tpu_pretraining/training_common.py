from __future__ import annotations
from flax.traverse_util import flatten_dict
import json

import argparse
from functools import partial
import copy

import flax
import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from jax.tree_util import tree_map_with_path

import modeling
import modeling_franca
from utils import (
    get_layer_index_fn,
    timm_to_flax,
    get_franca_flax,
)

from training_util import (
    CRITERION_COLLECTION,
    OPTIMIZER_COLLECTION,
)
from training import TrainModule as SupervisedTrainModule


def summarize_params(params):
    flat = flatten_dict(params)  # flattens nested dict using '/' as separator
    stats = {}

    for key_tuple, value in flat.items():
        key = "/".join(key_tuple)  # Convert tuple keys to string path
        stats[key] = {
            "min": float(jnp.min(value)),
            "max": float(jnp.max(value)),
            "mean": float(jnp.mean(value)),
        }

    return stats


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey
    patch_selection_rng: PRNGKey
    teacher_params: ArrayTree

    batch_stats: ArrayTree | None = None

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        patch_selection_rng, new_patch_selection_rng = jax.random.split(
            self.patch_selection_rng
        )

        rngs = {
            "mixup": mixup_rng,
            "dropout": dropout_rng,
            "patch_selection": patch_selection_rng,
        }
        updates = {
            "mixup_rng": new_mixup_rng,
            "dropout_rng": new_dropout_rng,
            "patch_selection_rng": new_patch_selection_rng,
        }
        return rngs, updates

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
            patch_selection_rng=shard_prng_key(self.patch_selection_rng),
            teacher_params=flax.jax_utils.replicate(self.teacher_params),
        )


@partial(jax.pmap, axis_name="batch", donate_argnums=0)
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            *batch,
            rngs=rngs,
            teacher_params=state.teacher_params,
            mutable=["batch_stats"],
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return metrics["loss"], (metrics, new_model_state["batch_stats"])

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_util.tree_map(
            lambda g: g / state.micro_in_mini, state.grad_accum
        )
        return state.apply_gradients(
            grads=jax.lax.pmean(grads, axis_name="batch"),
            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )

    rngs, updates = state.split_rngs()
    (_, (metrics, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    state = state.replace(
        batch_stats=batch_stats,
    )

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
    else:
        state = state.replace(
            grad_accum=jax.tree_util.tree_map(
                lambda ga, g: ga + g, state.grad_accum, grads
            ),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state.replace(**updates), metrics | state.opt_state.hyperparams


def create_train_state(args: argparse.Namespace) -> TrainState:
    ##### Create student and teacher ViTs #####
    if "franca" in args.timm_model_name:
        student = modeling_franca.ViT(
            args=args,
        )
        timm_params = get_franca_flax()
    else:
        student = modeling.ViT(
            layers=args.layers,
            dim=args.dim,
            heads=args.heads,
            num_registers=args.num_registers,
            labels=args.labels,
            patch_size=args.patch_size,
            image_size=args.image_size,
            args=args,
        )
        timm_params = timm_to_flax(args.timm_model_name)


    teacher_params = copy.deepcopy(timm_params)
    head_params = {
        "kernel": jax.random.normal(jax.random.PRNGKey(0), (args.dim, 1000))
        / jnp.sqrt(args.dim),
        "bias": jnp.zeros(1000),
    }
    teacher_params["head"] = head_params

    ##### Create zoomer ViTs #####
    if "franca" in args.timm_model_name:
        zoomer = modeling_franca.Zoomer(
            dim=args.dim,
            heads=args.heads,
            labels=args.labels,
            patch_size=args.patch_size,
            image_size=args.zoomer_image_size,
        )
    else:
        args.target_grid_size = int(args.image_size // args.patch_size)
        assert args.target_grid_size == 37
        zoomer = modeling.Zoomer(
            args=args,
            layers=args.zoomer_depth,
            dim=args.dim,
            heads=args.heads,
            num_registers=args.num_registers,
            labels=args.labels,
            patch_size=args.patch_size,
            image_size=args.zoomer_image_size,
        )

    module = SupervisedTrainModule(
        args=args,
        student=student,
        zoomer=zoomer,
        criterion=CRITERION_COLLECTION[args.criterion],
        zoom_map_criterion=CRITERION_COLLECTION[args.zoom_map_criterion],
    )

    example_inputs = {
        "images": jnp.zeros((1, 3, args.image_size, args.image_size), dtype=jnp.uint8),
        "labels": jnp.zeros((1), dtype=jnp.uint8),
        "k_patches": jnp.full(
            (1),
            args.top_k if args.top_k is not None else args.top_k_range.min,
            dtype=jnp.int32,
        ),
        "teacher_params": teacher_params,
    }

    key1, key2 = jax.random.split(jax.random.PRNGKey(args.init_seed), 2)
    init_rngs = {
        "params": key1,
        "dropout": key2,
    }
    # print(module.tabulate(init_rngs, **example_inputs))

    # mutable False needed to load BN parameters.
    init_params_ = module.init(init_rngs, **example_inputs)
    init_params = init_params_["params"]

    student_params = copy.deepcopy(timm_params)
    student_params["head"] = init_params["student"]["head"]

    zoomer_params = init_params["zoomer"]

    # Init from a ViT
    zoomer_params = copy.deepcopy(timm_params)
    zoomer_params["head"] = init_params["zoomer"]["head"]

    print(f"zoomer keys before removing: {zoomer_params.keys()}")
    keys_to_remove = []
    init_layers = args.get("init_layers", None)
    if init_layers is not None:
        assert len(init_layers) == args.zoomer_depth
    for key in list(zoomer_params.keys()):
        if "layer" in key:
            layer = int(key.split("_")[-1])
            # print(layer, init_layers)
            if init_layers is not None:
                if layer not in init_layers:
                    keys_to_remove.append(key)
            elif layer >= args.zoomer_depth:
                keys_to_remove.append(key)

    for key in keys_to_remove:
        del zoomer_params[key]


    if init_layers is not None:
        # Map the layers to 0, 1, 2
        layer_map = {}
        sorted_layers = sorted(init_layers)
        for i, layer in enumerate(sorted_layers):
            layer_map[layer] = i

        for key in list(zoomer_params.keys()):
            if "layer" in key:
                layer = int(key.split("_")[-1])
                new_key = key.replace(f"layer_{layer}", f"layer_{layer_map[layer]}")
                zoomer_params[new_key] = zoomer_params[key]
                if new_key != key:
                    del zoomer_params[key]

    print(f"zoomer keys after removing: {zoomer_params.keys()}")

    # Combine student and zoomer parameters into one dictionary.
    combined_params = {"student": student_params, "zoomer": zoomer_params}

    summ = summarize_params(combined_params)
    json_summary = json.dumps(summ, indent=2)

    # Save the summary to a file
    # with open("params_summary.json", "w") as f:
    #     f.write(json_summary)

    if args.grad_accum > 1:
        grad_accum = jax.tree_util.tree_map(jnp.zeros_like, combined_params)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
        learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = OPTIMIZER_COLLECTION[args.optimizer.name](
            learning_rate=learning_rate,
            b1=args.optimizer.betas[0],
            b2=args.optimizer.betas[1],
            eps=float(args.optimizer.eps),
            weight_decay=args.optimizer.weight_decay,
            mask=partial(tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        if args.optimizer.lr_decay < 1.0:
            layerwise_scales = {
                i: optax.scale(args.optimizer.lr_decay ** (args.layers - i))
                for i in range(args.layers + 1)
            }
            label_fn = partial(get_layer_index_fn, num_layers=args.layers)
            label_fn = partial(tree_map_with_path, label_fn)
            tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if args.optimizer.clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.optimizer.clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=float(args.optimizer.lr),
        warmup_steps=args.warmup_steps,
        decay_steps=args.training_steps,
        end_value=1e-6,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=combined_params,
        batch_stats=init_params.get("batch_stats", {"not_used": jnp.zeros((1,))}),
        tx=create_optimizer_fn(learning_rate),
        mixup_rng=jax.random.PRNGKey(args.mixup_seed + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed + jax.process_index()),
        patch_selection_rng=jax.random.PRNGKey(
            args.patch_select_seed + jax.process_index()
        ),
        micro_step=0,
        micro_in_mini=args.grad_accum,
        grad_accum=grad_accum if args.grad_accum > 1 else None,
        teacher_params=teacher_params,
    )
