import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from modelscope import snapshot_download
from modelscope.hub.api import HubApi
from modelscope.hub.constants import ModelVisibility


def _get_token(explicit_token: Optional[str], env_name: str) -> str:
    """Resolve a token from argument or environment."""
    if explicit_token:
        token = explicit_token
    else:
        token = os.getenv(env_name)
    if not token:
        raise ValueError(f"Environment variable {env_name} is not set")
    return token


def _ensure_modelscope_repo(api, repo_id: str, repo_type: str) -> str:
    """Create a public ModelScope repo if needed and return repo_id."""
    if "/" not in repo_id:
        repo_id = f"chenda/{repo_id}"
    if repo_id.count("/") != 1:
        raise ValueError("repo_id must be in the form 'owner/name'.")
    if api.repo_exists(repo_id=repo_id, repo_type=repo_type):
        return repo_id
    visibility = ModelVisibility.PUBLIC
    if repo_type == "dataset":
        api.create_dataset(repo_id, visibility=visibility)
    else:
        api.create_model(repo_id, visibility=visibility)
    print(f"Created ModelScope {repo_type} repo: {repo_id}")
    return repo_id


def download_from_modelscope(repo_id: str, local_dir: Path):
    """Download a ModelScope repo into a local directory."""
    local_dir.mkdir(exist_ok=True, parents=True)
    snapshot_download(repo_id, local_dir=str(local_dir))
    print(f"ModelScope download finished: {local_dir}")


def _push_folder_to_hub(api: HubApi, repo_id: str, folder_path: Path, message: str) -> None:
    """Upload a folder to ModelScope using HubApi."""
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder_path),
        commit_message=message,
    )


def _upload_extra_path(repo_root: Path, path: Path, label: str) -> Path:
    """Copy a log or TB path into the repo root."""
    if not path.exists():
        raise FileNotFoundError(f"{label} path not found: {path}")

    link_root = repo_root / "_logs"
    link_root.mkdir(exist_ok=True)

    link_path = link_root / path.name
    if link_path.exists() or link_path.is_symlink():
        return link_path
    if path.is_dir():
        shutil.copytree(path, link_path, dirs_exist_ok=True)
        return link_path
    shutil.copy2(path, link_path)
    return link_path


def _find_experiment_root(folder_path: Path) -> Path:
    """Find the experiment root containing a checkpoints dir."""
    current = folder_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "checkpoints").exists():
            return parent

    raise FileNotFoundError(f"Failed to locate experiment root with 'checkpoints' from: {folder_path}")


def _auto_detect_logs(experiment_root: Path) -> tuple[Path, Path]:
    """Detect the newest train log and TB directory."""
    log_candidates = sorted(experiment_root.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_candidates:
        raise FileNotFoundError("Failed to auto-detect train log (*.log).")
    train_log = log_candidates[0]

    tb_dir_candidates = []
    for candidate in experiment_root.iterdir():
        if candidate.is_dir() and ("tb" in candidate.name.lower() or "tensorboard" in candidate.name.lower()):
            tb_dir_candidates.append(candidate)
    if not tb_dir_candidates:
        tb_dir_candidates = sorted(
            [p.parent for p in experiment_root.glob("**/events.out.tfevents*")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not tb_dir_candidates:
        raise FileNotFoundError("Failed to auto-detect TensorBoard logs.")
    tb_log = tb_dir_candidates[0]
    return train_log, tb_log


def upload_to_modelscope(
    folder_path: Path,
    repo_id: str,
    token: str,
    api: HubApi,
    repo_type: str,
):
    """Upload model plus auto-detected logs to ModelScope."""
    repo_id = _ensure_modelscope_repo(api, repo_id, repo_type)
    experiment_root = _find_experiment_root(folder_path)
    train_log_path, tb_log_path = _auto_detect_logs(experiment_root)

    link_root = folder_path / "_logs"

    for log_path, label in ((train_log_path, "train log"), (tb_log_path, "tb log")):
        target_path = link_root / log_path.name
        if target_path.exists() or target_path.is_symlink():
            print(f"Skip copying {label}, target already exists: {target_path}")
            continue
        _upload_extra_path(folder_path, log_path, label)
    print(f"Uploading {folder_path} to ModelScope: {repo_id} ({repo_type})...")
    _push_folder_to_hub(api, repo_id, folder_path, "upload folder to repo")
    print("ModelScope upload finished.")


def main():
    """Parse CLI args and run download or upload."""
    parser = argparse.ArgumentParser(description="ModelScope download/upload helper.")
    parser.add_argument("repo_id", help="ModelScope repo id.")
    parser.add_argument("local_dir", help="Local directory to use.")
    parser.add_argument("--token", help="ModelScope token. Defaults to MODELSCOPE_TOKEN.")
    parser.add_argument("--repo-type", default="model", help="Repo type: model or dataset.")
    args = parser.parse_args()

    token = _get_token(args.token, "MODELSCOPE_TOKEN")
    api = HubApi()
    api.login(token)
    local_dir = Path(args.local_dir)
    if local_dir.exists():
        print("Branch: upload (local_dir exists) -> uploading to ModelScope")
        upload_to_modelscope(
            folder_path=local_dir,
            repo_id=args.repo_id,
            token=token,
            api=api,
            repo_type=args.repo_type,
        )
    else:
        print("Branch: download (local_dir not found) -> downloading from ModelScope")
        download_from_modelscope(repo_id=args.repo_id, local_dir=local_dir)


if __name__ == "__main__":
    main()
