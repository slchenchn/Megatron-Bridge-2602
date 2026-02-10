#!/usr/bin/env python3
"""Copy chat_template from ref tokenizer_config into model tokenizer_config if missing."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Copy chat_template from ref tokenizer_config into model if missing."""
    parser = argparse.ArgumentParser(description="Validate/copy chat_template in tokenizer_config.json")
    parser.add_argument("model_dir", help="Directory containing tokenizer_config.json to update")
    parser.add_argument(
        "--ref_model_dir",
        default="/home/admin/csl/checkpoints/moonshotai/Moonlight-16B-A3B/",
        help="Reference directory with tokenizer_config.json (source of chat_template)",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model_dir) / "tokenizer_config.json"
    ref_config_path = Path(args.ref_model_dir) / "tokenizer_config.json"

    if not model_config_path.is_file():
        raise SystemExit(f"Model config not found: {model_config_path}")
    if not ref_config_path.is_file():
        raise SystemExit(f"Reference config not found: {ref_config_path}")

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    with open(ref_config_path, "r", encoding="utf-8") as f:
        ref_config = json.load(f)

    if "chat_template" in model_config:
        print("Model tokenizer_config already has chat_template, nothing to do.")
        return

    chat_template = ref_config.get("chat_template")
    if chat_template is None:
        raise SystemExit("Reference tokenizer_config has no chat_template to copy.")

    model_config["chat_template"] = chat_template
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    print(f"Copied chat_template from {ref_config_path} into {model_config_path}")


if __name__ == "__main__":
    main()
