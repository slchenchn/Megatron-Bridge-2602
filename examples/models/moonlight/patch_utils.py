import collections
import datetime
import json
import logging
import os
import time
from functools import partial
from typing import Any, cast

import numpy as np
from tqdm.auto import tqdm, trange


def apply_monkey_patches():
    """Apply all performance and stability patches to Megatron-Bridge.
    Failures are explicit (no try-except).
    """
    # --- 1. Patch utils.build_index_files (Logging fix) ---
    # Avoid crash when safe_map returns None for failed workers.
    from megatron.bridge.data.datasets import utils as ds_utils

    def build_index_files_patched(
        dataset_paths,
        newline_int,
        workers=None,
        build_index_fn=ds_utils.build_index_from_memdata,
        index_mapping_dir: str | None = None,
    ):
        # Keep compatibility with callers that pass a single string path.
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        if len(dataset_paths) < 1:
            raise ValueError("files_list must contain at leat one file name")

        if workers is None:
            workers = max(1, (os.cpu_count() or 1) // 2)

        ds_utils.logger.info(f"Processing {len(dataset_paths)} data files using {workers} workers")
        start_time = time.time()
        build_status = ds_utils.safe_map(
            partial(
                ds_utils._build_memmap_index_files,
                newline_int,
                build_index_fn,
                index_mapping_dir=index_mapping_dir,
            ),
            dataset_paths,
            workers=workers,
        )
        # Only count successful builds; failed workers yield None.
        built = sum(1 for x in build_status if x is True)
        ds_utils.logger.info(
            f"Time building {built} / {len(build_status)} "
            f"mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
        )

    setattr(
        ds_utils,
        "build_index_files",
        cast(Any, build_index_files_patched),
    )
    ds_utils.logger.warning("[PATCH] Applied build_index_files logging fix")

    # --- 2. Patch packing_utils (Algorithm Optimization & TQDM) ---
    # Add progress bars and keep packing logic compatible with upstream.
    from megatron.bridge.data.datasets import packing_utils

    def first_fit_optimized(seqlens, pack_size):
        # First-fit packing with a tqdm loop for visibility on large datasets.
        res = []
        bin_remaining = []
        for s in tqdm(seqlens, desc="Packing sequences (first_fit)", mininterval=1, leave=False):
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if remaining >= s:
                    res[i].append(s)
                    bin_remaining[i] -= s
                    placed = True
                    break
            if not placed:
                res.append([s])
                bin_remaining.append(pack_size - s)
        return res

    def create_hist_patched(dataset, truncate_seq_len):
        # Build histogram with a progress bar to show tokenization scan progress.
        packing_utils.logger.info("Creating histogram from tokenized dataset...")
        sequences = collections.defaultdict(list)
        for item_dict in tqdm(dataset, desc="Creating histogram", mininterval=1, leave=False):
            seq_len = len(item_dict["input_ids"]) - 1
            sequences[seq_len].append(item_dict)
        histogram = []
        for seq_len in range(truncate_seq_len + 1):
            histogram.append(len(sequences[seq_len]))
        return sequences, histogram

    packing_utils.first_fit = cast(Any, first_fit_optimized)
    packing_utils.create_hist = cast(Any, create_hist_patched)
    packing_utils.logger.warning("[PATCH] Applied packing_utils algorithm optimization and progress bars")

    # --- 3. Patch packed_sequence (Tokenize Cache & TQDM) ---
    # Cache tokenized outputs and show tqdm progress during tokenization.
    from megatron.core.msc_utils import MultiStorageClientFeature  # type: ignore[import-not-found]

    from megatron.bridge.data.datasets import packed_sequence as ps

    def tokenize_dataset_patched(
        path,
        tokenizer,
        max_seq_length,
        seed,
        dataset_kwargs=None,
        pad_seq_to_mult=1,
    ):
        # Preserve dataset kwargs behavior while adding tokenization cache/progress.
        from megatron.bridge.data.datasets.sft import create_sft_dataset

        if not dataset_kwargs:
            dataset_kwargs = {}
        ts = dataset_kwargs.get("tool_schemas")
        if ts and not isinstance(ts, str):
            dataset_kwargs["tool_schemas"] = json.dumps(ts)
        chat_template = dataset_kwargs.pop("chat_template", None)
        if chat_template and hasattr(tokenizer, "_tokenizer"):
            tokenizer._tokenizer.chat_template = chat_template

        if pad_seq_to_mult is not None and pad_seq_to_mult <= 0:
            raise ValueError("pad_seq_to_mult must be a positive integer when provided.")

        pad_seq_length_to_mult = 1 if pad_seq_to_mult is None else max(1, pad_seq_to_mult)

        dataset = create_sft_dataset(
            path=path,
            tokenizer=tokenizer,
            seq_length=max_seq_length,
            seed=seed,
            is_test=True,
            pad_seq_length_to_mult=pad_seq_length_to_mult,
            **dataset_kwargs,
        )
        pad_id = dataset.tokenizer.eod
        pad_seq_length_to_mult = dataset.pad_seq_length_to_mult
        max_seq_length = dataset.max_seq_length
        iterator = trange(len(dataset), desc="Tokenizing dataset", mininterval=1, leave=False)
        dataset = np.array([dataset[i] for i in iterator])

        if pad_seq_to_mult > 1:

            def pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id):
                # Pre-pad sequences to the next multiple to improve packing efficiency.
                assert max_seq_length >= max_length_to_pad
                for key, val in data.items():
                    if key in {"input_ids", "context_ids"}:
                        if len(val) <= max_length_to_pad:
                            val = val + [pad_id] * (max_length_to_pad - len(val) + 1)
                        elif len(val) > max_seq_length:
                            logging.info(
                                "Sequence length %d is larger than max_seq_length %d; truncating for packing.",
                                len(val),
                                max_seq_length,
                            )
                            val = val[:max_seq_length]
                        data[key] = val
                return

            ceil_to_nearest = lambda n, m: (n + m - 1) // m * m
            for data in dataset:
                max_length_to_pad = min(
                    max_seq_length, ceil_to_nearest(len(data["input_ids"]), pad_seq_length_to_mult)
                )
                pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id)

        return dataset

    def prepare_packed_sequence_data_patched(
        input_path,
        output_path,
        output_metadata_path,
        packed_sequence_size,
        tokenizer,
        max_seq_length,
        seed=0,
        packing_algorithm="first_fit_shuffle",
        dataset_kwargs=None,
    ):
        # Reuse cached tokenization to avoid recomputation across retries.
        logger = logging.getLogger(ps.__name__)
        logger.info(f"Preparing packed sequence from {input_path}")
        cache_path = output_path.with_name(f"{output_path.stem}.tokenized_cache.npy")

        if cache_path.exists():
            logger.info(f"Loading cached tokenized data from {cache_path}")
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                dataset = msc.numpy.load(cache_path, allow_pickle=True)
            else:
                dataset = np.load(cache_path, allow_pickle=True)
        else:
            dataset = ps.tokenize_dataset(input_path, tokenizer, max_seq_length, seed, dataset_kwargs)
            logger.info(f"Saving tokenized data cache to {cache_path}")
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                msc.numpy.save(cache_path, cast(Any, dataset))
            else:
                np.save(cache_path, cast(Any, dataset))

        sequences, histogram = ps.create_hist(dataset, max_seq_length)
        assignments, packing_metadata = ps.create_packing_strategy(histogram, packed_sequence_size, packing_algorithm)
        output_data = ps.fill_packing_strategy(assignments, sequences, packed_sequence_size, tokenizer.eos_id)

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            msc.numpy.save(output_path, cast(Any, output_data))
        else:
            np.save(output_path, cast(Any, output_data))

        if output_metadata_path is not None:
            try:
                with output_metadata_path.open(mode="r") as f:
                    packing_metadata_file = json.load(f)
            except FileNotFoundError:
                packing_metadata_file = []
            packing_metadata_file.append(packing_metadata)
            with output_metadata_path.open(mode="w") as f:
                json.dump(packing_metadata_file, f)
        logger.info(f"Packed sequence is prepared and saved to {output_path}")

    ps.tokenize_dataset = cast(Any, tokenize_dataset_patched)
    ps.prepare_packed_sequence_data = cast(Any, prepare_packed_sequence_data_patched)
    ps.logger.warning("[PATCH] Applied packed_sequence tokenize cache and progress bars")

    # --- 4. Patch FinetuningDatasetBuilder (Missing validation.jsonl fix) ---
    # Skip validation packing/build if validation.jsonl is absent.
    from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
    from megatron.bridge.data.datasets import packed_sequence as ps

    def prepare_packed_data_patched(self) -> None:
        # Prepare train packing; only attempt validation if the source file exists.
        from megatron.bridge.utils.common_utils import print_rank_0

        if self.packed_sequence_size <= 0:
            return

        if not self.train_path_packed.is_file():
            print_rank_0(f"[PATCHED] Preparing packed training data at {self.train_path_packed}")
            ps.prepare_packed_sequence_data(
                input_path=self.train_path,
                output_path=self.train_path_packed,
                packed_sequence_size=self.packed_sequence_size,
                tokenizer=self.tokenizer,
                max_seq_length=self.seq_length,
                seed=self.seed,
                output_metadata_path=self.pack_metadata,
                dataset_kwargs=self.dataset_kwargs,
                pad_seq_to_mult=self._pad_seq_to_mult,
            )
        else:
            print_rank_0(f"[PATCHED] Packed training data already exists at {self.train_path_packed}")

        if not self.do_validation:
            return

        if self.validation_path_packed.is_file():
            return

        if not self.validation_path.is_file():
            self.do_validation = False
            print_rank_0("[PATCHED] validation.jsonl missing; disabling validation dataset/packing to avoid crash.")
            return

        print_rank_0(f"[PATCHED] Preparing packed validation data at {self.validation_path_packed}")
        ps.prepare_packed_sequence_data(
            input_path=self.validation_path,
            output_path=self.validation_path_packed,
            packed_sequence_size=self.packed_sequence_size,
            tokenizer=self.tokenizer,
            max_seq_length=self.seq_length,
            seed=self.seed,
            output_metadata_path=self.pack_metadata,
            dataset_kwargs=self.dataset_kwargs,
            pad_seq_to_mult=self._pad_seq_to_mult,
        )

    def _build_datasets_patched(self):
        # Guard validation build when packed/unpacked validation files are missing.
        from megatron.bridge.utils.common_utils import print_rank_0

        train_ds = self._create_dataset(
            self.train_path if self.packed_sequence_size <= 0 else self.train_path_packed,
            pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
            max_num_samples=self.max_train_samples,
            **self.dataset_kwargs,
        )

        valid_ds = None
        if self.do_validation:
            valid_path = self.validation_path if self.packed_sequence_size <= 0 else self.validation_path_packed
            if not valid_path.is_file():
                self.do_validation = False
                print_rank_0(f"[PATCHED] Validation file missing at {valid_path}; skipping validation dataset build.")
            else:
                valid_ds = self._create_dataset(
                    valid_path,
                    pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
                    is_test=True,
                    **self.dataset_kwargs,
                )

        if self.do_test:
            test_ds = self._create_dataset(
                self.test_path,
                is_test=True,
                **self.dataset_kwargs,
            )
        else:
            test_ds = None

        return [train_ds, valid_ds, test_ds]

    FinetuningDatasetBuilder.prepare_packed_data = cast(Any, prepare_packed_data_patched)
    FinetuningDatasetBuilder._build_datasets = cast(Any, _build_datasets_patched)
    logging.getLogger("megatron.bridge").warning(
        "[PATCH] Applied FinetuningDatasetBuilder missing validation.jsonl handling"
    )
