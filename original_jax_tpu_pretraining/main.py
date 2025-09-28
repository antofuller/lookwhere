from __future__ import annotations

import os
import json
import argparse
import warnings
import re
import random

import jax
import jax.numpy as jnp
import webdataset as wds
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
from flax.serialization import msgpack_serialize

from dataset import create_dataloaders
from training_common import create_train_state, training_step
from utils import AverageMeter, save_checkpoint_in_background
from misc import load_config

warnings.filterwarnings("ignore")


def main(args: argparse.Namespace):
    train_dataloader, valid_dataloader = create_dataloaders(args)
    train_dataloader_iter = iter(train_dataloader)
    state = create_train_state(args).replicate()
    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args)
    average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0

    for step in tqdm.trange(1, args.training_steps + 1, dynamic_ncols=True):
        k_step = (
            random.randint(args.top_k_range.min, args.top_k_range.max + 1)
            if args.top_k_range is not None
            else args.top_k
        )
        k_patches = shard(
            jnp.full((jax.local_device_count(),), k_step, dtype=jnp.int32)
        )

        for _ in range(args.grad_accum):
            batch = shard(
                jax.tree_util.tree_map(np.asarray, next(train_dataloader_iter))
            )
            state, metrics = training_step(
                state,
                batch=(
                    *batch,
                    k_patches
                ),
            )
            average_meter.update(**unreplicate(metrics))

        if (
            jax.process_index() == 0
            and args.log_interval > 0
            and step % args.log_interval == 0
        ):
            metrics = average_meter.summary(prefix="train/")
            metrics["processed_samples"] = step * args.train_batch_size
            wandb.log(metrics, step)

    if jax.process_index() == 0:
        params_bytes = msgpack_serialize(unreplicate(state.params))
        batch_state_bytes = msgpack_serialize(unreplicate(state.batch_stats))
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.name)

        if args.output_dir.startswith("gs://"):
            config_path = os.path.join(args.output_dir, f"{safe_name}-config.msgpack")

            save_checkpoint_in_background(
                args, params_bytes, postfix="last", safe_run_name=safe_name
            )
            save_checkpoint_in_background(
                args,
                batch_state_bytes,
                postfix="batch_state-last",
                safe_run_name=safe_name,
            )

        else: 
            os.makedirs(args.output_dir, exist_ok=True)
            config_path = os.path.join(args.output_dir, f"{safe_name}-config.msgpack")
            save_checkpoint_in_background(
                args, params_bytes, postfix="last", safe_run_name=safe_name
            )
            save_checkpoint_in_background(
                args,
                batch_state_bytes,
                postfix="batch_state-last",
                safe_run_name=safe_name,
            )

        with wds.gopen(config_path, "wb") as fp:
            fp.write(json.dumps(args.as_dict()).encode("utf-8"))

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="config/imagenet.yaml")
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--valid-dataset-shards")

    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    parser.add_argument("--output-dir")
    config_in = parser.parse_args()

    args = load_config(
        config_path=os.path.abspath(config_in.config_path),
    )
    # Update args with command line arguments
    args.train_dataset_shards = config_in.train_dataset_shards
    args.valid_dataset_shards = config_in.valid_dataset_shards
    args.name = config_in.name
    args.ipaddr = config_in.ipaddr
    args.hostname = config_in.hostname
    args.output_dir = config_in.output_dir

    print(args)
    main(args)
