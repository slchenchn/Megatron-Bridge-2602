#!/usr/bin/env python3
import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "3rdparty" / "Megatron-LM"
sys.path.insert(0, str(MEGATRON_ROOT))

from datasets import load_dataset  # noqa: E402
from megatron.core.datasets import indexed_dataset  # noqa: E402
from megatron.core.tokenizers.text.utils.build_tokenizer import (  # noqa: E402
    build_tokenizer as build_new_tokenizer,
)
from megatron.training.arguments import _add_tokenizer_args  # noqa: E402


class Encoder:
    tokenizer = None

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def initializer(self) -> None:
        Encoder.tokenizer = build_new_tokenizer(self.args)

    def encode(self, row: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, list[int]], int]:
        ids: dict[str, list[int]] = {}
        lens: dict[str, list[int]] = {}
        for key in self.args.json_keys:
            text = row[key]
            sentences = text if isinstance(text, list) else [text]
            doc_ids: list[int] = []
            sentence_lens: list[int] = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if sentence_ids:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if doc_ids and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        # Keep MB/s metric meaningful for streaming dict rows.
        return ids, lens, len(json.dumps(row, ensure_ascii=False))


def print_processing_stats(count: int, start_t: float, total_bytes_processed: int, interval: int) -> None:
    if count % interval != 0:
        return
    elapsed = time.time() - start_t
    mbs = total_bytes_processed / max(elapsed, 1e-9) / 1024 / 1024
    print(f"Processed {count} documents ({count / max(elapsed, 1e-9)} docs/s, {mbs} MB/s).", file=sys.stderr)


def load_rows(input_path: str, data_format: str):
    if os.path.isdir(input_path):
        try:
            return load_dataset(data_format, data_dir=input_path, split="train", streaming=True)
        except Exception:
            return load_dataset(data_format, data_dir=input_path, split="validation", streaming=True)
    try:
        return load_dataset(data_format, data_files=input_path, split="train", streaming=True)
    except Exception:
        return load_dataset(data_format, data_files=input_path, split="validation", streaming=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize dataset directory via HF streaming loader")
    parser = _add_tokenizer_args(parser)
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output-prefix", type=str, required=True, help="Output prefix without suffix")
    parser.add_argument("--json-keys", nargs="+", default=["text"], help="JSON keys to tokenize")
    parser.add_argument("--workers", type=int, required=True, help="Number of tokenizer workers")
    parser.add_argument("--append-eod", action="store_true", help="Append EOD token")
    parser.add_argument("--log-interval", type=int, default=1000, help="Logging interval")
    parser.add_argument("--data-format", type=str, default="json", help="HF dataset format")
    args = parser.parse_args()

    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.keep_empty = False
    return args


def main() -> None:
    args = parse_args()
    print("Opening", args.input)
    rows = load_rows(args.input, args.data_format)

    startup_start = time.time()
    tokenizer = build_new_tokenizer(args)
    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_rows = pool.imap(encoder.encode, rows, 32)

    output_bin_files: dict[str, str] = {}
    output_idx_files: dict[str, str] = {}
    builders: dict[str, indexed_dataset.IndexedDatasetBuilder] = {}
    for key in args.json_keys:
        output_bin_files[key] = f"{args.output_prefix}_{key}_document.bin"
        output_idx_files[key] = f"{args.output_prefix}_{key}_document.idx"
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

    startup_end = time.time()
    print("Time to startup:", startup_end - startup_start)
    proc_start = time.time()
    total_bytes_processed = 0

    try:
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_rows, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            print_processing_stats(i, proc_start, total_bytes_processed, args.log_interval)
    finally:
        pool.close()
        pool.join()

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

