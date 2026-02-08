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

from megatron.bridge.recipes.moonlight import moonlight_16b_pretrain_config
from megatron.bridge.training.config import TokenizerConfig
from megatron.bridge.training.gpt_step import forward_step
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
        "--data-path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Megatron indexed dataset prefix list. Supports weighted format, e.g. "
            "--data-path 30 /path/a 70 /path/b"
        ),
    )
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

    parser.add_argument("--tp", "--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pp", "--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--cp", "--context-parallel-size", type=int, default=1)
    parser.add_argument(
        "--ep",
        "--expert-model-parallel-size",
        type=int,
        default=None,
        help="Expert model parallel size. Defaults to world_size // (tp*pp*cp).",
    )

    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--split", type=str, default="99,1,0")

    parser.add_argument("--save", type=str, default=None, help="Checkpoint save dir override.")
    parser.add_argument("--load", type=str, default=None, help="Checkpoint load dir override.")

    parser.add_argument("--enable-recompute", action="store_true")
    parser.add_argument("--apply-rope-fusion", action="store_true")
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

    tokenizer_path = args.tokenizer_path or _auto_detect_tokenizer_path(args.model_name)
    expert_model_parallel_size = args.ep or _default_ep(args.tp, args.pp, args.cp)

    cfg = moonlight_16b_pretrain_config(
        name=args.exp_name,
        data_paths=args.data_path,
        mock=False,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=args.tp > 1,
        recompute_granularity="selective" if args.enable_recompute else "none",
        apply_rope_fusion=args.apply_rope_fusion,
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

    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_path,
        hf_tokenizer_kwargs={"trust_remote_code": True},
    )
    cfg.dataset.split = args.split
    cfg.logger.log_interval = args.log_interval

    if args.save is not None:
        cfg.checkpoint.save = args.save
    if args.load is not None:
        cfg.checkpoint.load = args.load

    print_rank_0(
        "[moonlight-pretrain] "
        f"optimizer={args.optimizer_type}, precision={args.precision_config}, "
        f"tp={args.tp}, pp={args.pp}, cp={args.cp}, ep={expert_model_parallel_size}, "
        f"train_iters={args.train_iters}, gbs={args.global_batch_size}, mbs={args.micro_batch_size}, "
        f"seq_len={args.seq_length}, data_path={args.data_path}"
    )

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
