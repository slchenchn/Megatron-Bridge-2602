#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "3rdparty" / "Megatron-LM"
sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.indexed_dataset import (  # noqa: E402
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively merge indexed dataset shards")
    parser.add_argument("--input", type=str, required=True, help="Directory to scan for .idx/.bin pairs")
    parser.add_argument("--output-prefix", type=str, required=True, help="Merged output prefix")
    parser.add_argument("--multimodal", action="store_true", help="Treat inputs as multimodal datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.input):
        raise ValueError(f"Input is not a directory: {args.input}")

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir and not os.path.isdir(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")

    prefixes: list[str] = []
    for root, _, files in os.walk(args.input):
        basenames = {os.path.splitext(name)[0] for name in files}
        for base in sorted(basenames):
            if f"{base}.idx" in files and f"{base}.bin" in files:
                prefixes.append(os.path.join(root, base))

    if not prefixes:
        print(f"ERROR: no indexed dataset files found under {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(prefixes)} indexed dataset prefixes", file=sys.stderr)
    first = IndexedDataset(prefixes[0], multimodal=args.multimodal)
    builder = IndexedDatasetBuilder(
        get_bin_path(args.output_prefix),
        dtype=first.index.dtype,
        multimodal=args.multimodal,
    )
    del first

    for prefix in prefixes:
        builder.add_index(prefix)

    builder.finalize(get_idx_path(args.output_prefix))


if __name__ == "__main__":
    main()

