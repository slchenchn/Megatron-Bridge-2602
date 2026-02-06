#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
    Single GPU:
        torchrun --nproc_per_node=1 00_quickstart_pretrain.py

    Multiple GPUs (automatic data parallelism):
        torchrun --nproc_per_node=8 00_quickstart_pretrain.py

The script uses sensible defaults and mock data for quick testing.
For custom configurations through YAML and Hydra-style overrides, see 02_pretrain_with_yaml.py
For multi-node training, see launch_with_sbatch.sh or 04_launch_slurm_with_nemo_run.py
"""

import argparse
from pathlib import Path

from megatron.bridge.recipes.moonlight import moonlight_16b_finetune_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune Qwen3-8B with LoRA",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Moonlight-16B-A3B",
        help="Model name",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint in Megatron format",
    )
    parser.add_argument(
        "--packed-sequence",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to use packed sequence",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Sequence length",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="default",
        help="Experiment name",
    )
    parser.add_argument(
        "--debugpy",
        action="store_true",
        help="Whether to use debugpy",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=1000,
        help="Training iterations (used when --train-epochs is not set).",
    )
    parser.add_argument(
        "--train-epochs",
        type=float,
        default=2,
        help="Train by epochs (converted to train_samples using HF dataset length). Overrides --train-iters.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size per GPU (default: 1).",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="Tensor model parallel size (must match the pretrained checkpoint TP).",
    )
    # Activation recomputation (a.k.a. activation checkpointing) to reduce memory.
    # Mirrors Megatron-Core transformer config fields.
    parser.add_argument(
        "--enable-recompute",
        action="store_true",
        help="Enable activation recomputation (activation checkpointing).",
    )
    parser.add_argument(
        "--cross-entropy-fusion-impl",
        type=str,
        default="native",
        choices=["native", "te"],
        help=(
            "Cross entropy implementation when cross_entropy_loss_fusion=True. "
            "'native' uses Megatron-Core fused vocab-parallel CE (allocates an FP32 logits buffer). "
            "'te' uses TransformerEngine parallel cross entropy (often lower peak VRAM for TP=1)."
        ),
    )
    args = parser.parse_args()
    args.packed_sequence = bool(args.packed_sequence)
    return args


def main() -> None:
    args = parse_args()
    a800_path = f"/nfs/FM/chenshuailin/checkpoints/moonshotai/{args.model_name}"
    h100_path = f"/home/admin/csl/checkpoints/moonshotai/{args.model_name}"
    if Path(a800_path).exists():
        hf_model_path = a800_path
        dataset_root = "/nfs/FM/datasets/olmo3-post-training/allenai__Dolci-Instruct-SFT-No-Tools.filtered.shuffle"
        precision_config = "bf16_mixed"
        print(f">> auto detect a800 env, {dataset_root=}, {hf_model_path=}, {precision_config=}")
    elif Path(h100_path).exists():
        hf_model_path = h100_path
        dataset_root = "/mnt/md0/Dataset/olmo3-post-training/allenai__Dolci-Instruct-SFT-No-Tools.filtered.shuffle"
        precision_config = "bf16_with_fp8_subchannel_scaling_mixed"
        print(f">> auto detect h100 env, {dataset_root=}, {hf_model_path=}, {precision_config=}")
    else:
        raise ValueError(f"Checkpoint not found in {a800_path} or {h100_path}")

    # Load the base recipe configuration
    config = moonlight_16b_finetune_config()

    # OPTIONAL: Customize key settings here
    # Uncomment and modify as needed:

    # For a quick test run:
    config.train.train_iters = 10
    config.scheduler.lr_warmup_iters = 2

    # Use your own data:
    # config.data.data_path = "/path/to/your/dataset"

    # Adjust batch sizes for your GPU memory:
    # config.train.global_batch_size = 256
    # config.train.micro_batch_size = 2

    # Change checkpoint save frequency:
    # config.train.save_interval = 500

    # Start pretraining
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
