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

"""Moonlight pretraining launcher for Megatron-Bridge."""

import argparse
from pathlib import Path
from typing import cast

from megatron.bridge.recipes.moonlight.moonlight_16b import _moonlight_common
from megatron.bridge.training.config import GPTDatasetConfig, TokenizerConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.common_utils import get_world_size_safe, print_rank_0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain Moonlight-16B-A3B with Megatron-Bridge",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="Moonlight-16B-A3B", help="Model name for logging only.")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="HuggingFace tokenizer path. If unset, auto-detects common local Moonlight paths.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint directory. Loads model weights only (no optim/rng).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        required=False,
        help=(
            "Megatron indexed dataset prefix list. Supports weighted format, e.g. "
            "--data-path 30 /path/a 70 /path/b"
        ),
    )
    parser.add_argument("--mock-data", action="store_true", help="Use mock GPT data for debugging.")
    parser.add_argument("--optimizer-type", type=str, default="dist_muon", choices=["adam", "dist_muon"])
    parser.add_argument("--precision-config", type=str, default="bf16_mixed")
    parser.add_argument("--exp-name", type=str, default="moonlight_pretrain")

    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.5e-6)
    parser.add_argument("--lr-warmup-iters", type=int, default=100)
    parser.add_argument(
        "--lr-decay-style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "inverse-square-root", "WSD"],
    )
    parser.add_argument("--lr-wsd-decay-iters", type=int, default=None)

    parser.add_argument("--tp", "--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--cp", "--context-parallel-size", type=int, default=1)
    parser.add_argument(
        "--ep",
        "--expert-model-parallel-size",
        type=int,
        default=None,
        help="Expert model parallel size. Defaults to world_size // (tp*pp*cp).",
    )

    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--split", type=str, default="99,1,0")

    parser.add_argument("--save", type=str, default=None, help="Checkpoint save dir override.")

    parser.add_argument(
        "--token-drop",
        action="store_true",
        dest="token_drop",
        help="Enable MoE token drop (better expert load balance, may affect convergence). Default: False.",
    )
    parser.add_argument("--enable-recompute", action="store_true")
    return parser.parse_args()


def _auto_detect_tokenizer_path(model_name: str) -> str:
    candidates = [
        f"/nfs/FM/chenshuailin/checkpoints/moonshotai/{model_name}",
        f"/home/admin/csl/checkpoints/moonshotai/{model_name}",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise ValueError(
        "Failed to auto-detect tokenizer path. Please pass --tokenizer-path explicitly. "
        f"Checked: {candidates}"
    )


def _default_ep(tp: int, pp: int, cp: int) -> int:
    world_size = max(1, get_world_size_safe())
    divisor = max(1, tp * pp * cp)
    ep = max(1, world_size // divisor)
    if world_size % divisor != 0:
        print_rank_0(
            f"[WARN] world_size={world_size} is not divisible by tp*pp*cp={divisor}; "
            f"fallback ep={ep}. Please set --ep explicitly if needed."
        )
    return ep


def main() -> None:
    args = parse_args()

    if not args.mock_data and not args.data_path:
        raise ValueError("--data-path is required unless --mock-data is set")

    tokenizer_path = args.tokenizer_path or _auto_detect_tokenizer_path(args.model_name)
    expert_model_parallel_size = args.ep or _default_ep(args.tp, 1, args.cp)

    cfg = _moonlight_common(
        name=args.exp_name,
        data_paths=args.data_path,
        mock=args.mock_data,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=args.cp,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=args.tp > 1,
        recompute_granularity="selective",
        apply_rope_fusion=True,
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_iters=args.lr_warmup_iters,
        optimizer_type=args.optimizer_type,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        precision_config=args.precision_config,
    )
    cfg.scheduler.lr_decay_style = args.lr_decay_style
    if args.lr_decay_style == "WSD":
        if args.lr_wsd_decay_iters is None:
            raise ValueError("--lr-wsd-decay-iters is required when --lr-decay-style WSD")
        cfg.scheduler.lr_wsd_decay_iters = args.lr_wsd_decay_iters
    else:
        cfg.scheduler.lr_wsd_decay_iters = None
    cfg.model.cross_entropy_fusion_impl = "te"

    if args.token_drop:
        apply_moe_token_drop(cfg.model)
        cfg.model.moe_router_force_load_balancing = False
    else:
        apply_moe_token_drop(
            cfg.model,
            moe_expert_capacity_factor=-1.0,
            moe_pad_expert_input_to_capacity=False,
        )

    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_path,
        hf_tokenizer_kwargs={"trust_remote_code": True},
    )
    dataset_cfg = cast(GPTDatasetConfig, cfg.dataset)
    dataset_cfg.split = args.split
    cfg.logger.log_interval = args.log_interval
    cfg.logger.log_throughput = True

    if args.save is not None:
        cfg.checkpoint.save = args.save

    if args.pretrained_checkpoint is not None:
        cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
        cfg.checkpoint.load = args.pretrained_checkpoint
        cfg.checkpoint.finetune = True
        cfg.checkpoint.load_optim = False
        cfg.checkpoint.load_rng = False

    print_rank_0(
        "[moonlight-pretrain] "
        f"optimizer={args.optimizer_type}, precision={args.precision_config}, "
        f"tp={args.tp}, pp=1, cp={args.cp}, ep={expert_model_parallel_size}, "
        f"train_iters={args.train_iters}, gbs={args.global_batch_size}, mbs={args.micro_batch_size}, "
        f"seq_len={args.seq_length}, token_drop={args.token_drop}, mock_data={args.mock_data}, "
        f"data_path={args.data_path}, pretrained={args.pretrained_checkpoint}"
    )

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
