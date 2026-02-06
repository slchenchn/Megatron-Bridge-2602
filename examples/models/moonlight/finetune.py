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
from typing import Any, Optional, cast

import torch
from patch_utils import apply_monkey_patches

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig, ProcessExampleOutput
from megatron.bridge.recipes.moonlight.moonlight_16b import MoonlightFinetuneKwargs, _moonlight_finetune_common
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.utils.common_utils import get_world_size_safe


def calculate_train_samples(train_size: int, train_epochs: float, ddp_size: int, packing_factor: float = 1.0) -> int:
    """Calculate the number of training samples (packs) based on epochs and data size."""
    return int(train_epochs * (train_size / packing_factor) / ddp_size) * ddp_size


def calculate_global_batch_size(base_gbs: int = 128, packing_factor: float = 1.0, round_to: int = 8) -> int:
    """Scale global batch size based on packing factor to maintain token throughput."""
    scaled_gbs = base_gbs / packing_factor
    return max(round_to, round(scaled_gbs / round_to) * round_to)


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
    args = parser.parse_args()
    args.packed_sequence = bool(args.packed_sequence)
    return args


def process_dolci_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> ProcessExampleOutput:
    """Convert a Dolci Instruct SFT record into the standard input/output pair.

    The dataset schema (from the local README and parquet inspection):
    - id: string
    - messages: list of {content: string, function_calls: string, functions: string, role: string}
    - source: string

    Uses the tokenizer's apply_chat_template to format the conversation properly.
    The prompt includes all messages up to (but not including) the last assistant message,
    and the target is the last assistant message content.
    """

    messages = example.get("messages")
    assert isinstance(messages, list), f"messages must be a list, but got {type(messages)}"
    assert len(messages) >= 1, f"messages must have at least one message, but got {len(messages)}"

    # Use tokenizer's apply_chat_template if available
    if tokenizer is not None and hasattr(tokenizer, "_tokenizer"):
        hf_tokenizer = tokenizer._tokenizer
        if hasattr(hf_tokenizer, "apply_chat_template"):
            prompt = hf_tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError("tokenizer must have a _tokenizer attribute with an apply_chat_template method")
    else:
        raise ValueError("tokenizer must be provided")

    return ProcessExampleOutput(input=prompt, output=messages[-1]["content"], original_answers=[])


def _count_jsonl_lines(path: Path) -> int:
    """Count number of lines in a .jsonl file."""
    # Each record is one JSON object per line.
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    args = parse_args()
    apply_monkey_patches()
    a800_path = f"/nfs/FM/chenshuailin/checkpoints/moonshotai/{args.model_name}"
    h100_path = f"/home/admin/csl/checkpoints/moonshotai/{args.model_name}"
    if Path(a800_path).exists():
        hf_model_path = a800_path
        dataset_root = "/nfs/FM/datasets/olmo3-post-training/allenai__Dolci-Instruct-SFT-No-Tools.filtered.shuffle"
        precision_config = "bf16_mixed"
        enable_deepep = False
        print(f">> auto detect a800 env, {dataset_root=}, {hf_model_path=}, {precision_config=}, {enable_deepep=}")
    elif Path(h100_path).exists():
        hf_model_path = h100_path
        dataset_root = "/mnt/md0/Dataset/olmo3-post-training/allenai__Dolci-Instruct-SFT-No-Tools.filtered.shuffle"
        precision_config = "bf16_with_fp8_subchannel_scaling_mixed"
        enable_deepep = True
        print(f">> auto detect h100 env, {dataset_root=}, {hf_model_path=}, {precision_config=}, {enable_deepep=}")
    else:
        raise ValueError(f"Checkpoint not found in {a800_path} or {h100_path}")

    # Load the base recipe configuration

    kwargs: MoonlightFinetuneKwargs = {
        "tokenizer_path": hf_model_path,
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "sequence_parallel": True,
        "recompute_granularity": "selective",
        "enable_deepep": False,
        "apply_rope_fusion": False,
        "peft": None,
        "finetune_lr": 5e-6,
        "name": args.exp_name,
        "packed_sequence": args.packed_sequence,
        "seq_length": args.seq_length,
    }
    config = _moonlight_finetune_common(**kwargs)

    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    config.checkpoint.save_interval = 5000

    config.model.cross_entropy_fusion_impl = "te"

    # ===== OPTIONAL CUSTOMIZATIONS =====
    dataset_cfg = cast(HFDatasetConfig, config.dataset)
    dataset_cfg.dataset_name = dataset_root
    dataset_cfg.process_example_fn = process_dolci_example
    dataset_cfg.dataset_root = dataset_root
    dataset_cfg.val_proportion = 0.0
    dataset_cfg.do_validation = False

    # Print/log every N iterations.
    config.logger.log_interval = 5
    # Increase PG/NCCL collective timeout for long tokenization.
    config.dist.distributed_timeout_minutes = 240

    # === Quick test run ===
    packing_factor = 6.82 if args.packed_sequence else 1.0
    world_size = get_world_size_safe()
    config.set_data_parallel_size()
    ddp_size = config.data_parallel_size
    config.train.global_batch_size = calculate_global_batch_size(base_gbs=128, packing_factor=packing_factor)
    config.train.micro_batch_size = args.micro_batch_size
    if args.train_epochs is not None:
        # Epoch-based training is implemented via sample-based training.
        train_jsonl = Path(dataset_root) / "training.jsonl"
        train_size = _count_jsonl_lines(train_jsonl)

        # Scale train_samples based on packing factor (1 sample = 1 pack)
        train_samples = calculate_train_samples(train_size, args.train_epochs, ddp_size, packing_factor)

        config.train.train_iters = None
        config.train.train_samples = train_samples

        # Convert any existing iter-based schedule to sample-based for compatibility.
        config.scheduler.lr_warmup_samples = train_samples // 10
        config.scheduler.lr_warmup_iters = 0
        config.scheduler.lr_decay_iters = None
        config.scheduler.lr_wsd_decay_iters = None
        print(
            f">> train_epochs={args.train_epochs}, train_size={train_size}, packing_factor={packing_factor}, train_samples={train_samples} (packs), GBS={config.train.global_batch_size}, World Size={world_size}, DDP Size={ddp_size}"
        )

        # # Ensure decay is sample-based as well.
        # config.scheduler.lr_decay_iters = None
        # config.scheduler.lr_decay_samples = train_samples
    else:
        config.train.train_iters = args.train_iters
        config.scheduler.lr_warmup_iters = args.train_iters // 10

    # === Adjust learning rate ===
    config.optimizer.lr = 5e-6
    config.mixed_precision = precision_config

    # Start finetuning
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
