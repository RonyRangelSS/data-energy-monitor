import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm")
HASH_CHUNK_BYTES = 8 * 1024 * 1024
GENERATED_REPORTS_PATHSPEC = ":(exclude)artifacts/experiments"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def describe_code_version() -> dict[str, Any]:
    status = run_git("status", "--porcelain", "--", ".", GENERATED_REPORTS_PATHSPEC)
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "has_uncommitted_changes": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str]:
    return {package: metadata.version(package) for package in TRACKED_PACKAGES}


def build_run_record(
    experiment: str,
    configuration: dict[str, Any],
    data_path: Path,
    seed: int | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "seed": seed,
        "code_version": describe_code_version(),
        "package_versions": package_versions(),
        "data_file": data_path.name,
        "data_sha256": hash_file(data_path),
        "metrics": metrics,
    }


def write_json(content: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, ensure_ascii=False, indent=2, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
