import os
import argparse
from itertools import product
from copy import deepcopy

from main import main
from misc import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LookWhere Training"
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)

    parser.add_argument("--config-path", default="config/imagenet_franca.yaml")
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--valid-dataset-shards")

    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default=".")
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
    config_in = parser.parse_args()

    # Main sweep
    learning_rates = [1.0e-5, 2.0e-5, 5.0e-5]
    weight_decays = [0.02, 0.05, 0.1]

    cartesian = product(
        learning_rates,
        weight_decays,
    )
    total_exp = len(list(deepcopy(cartesian)))

    start = config_in.start
    end = config_in.end
    print(f"Running experiments {start} to {end}. Total:", total_exp)

    for exp_idx, (
        exp_learning_rate,
        exp_weight_decay,
    ) in enumerate(list(cartesian)[start:end]):
        args.optimizer.lr = exp_learning_rate
        args.optimizer.weight_decay = exp_weight_decay

        sweep_idx = exp_idx + start
        args.name = f"swp={sweep_idx}_LR={exp_learning_rate}_WD={exp_weight_decay}"

        print("Running experiment", exp_idx + start, "of", total_exp)
        print("Args", args)

        main(args)
