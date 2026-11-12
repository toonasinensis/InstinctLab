"""W&B model-artifact helpers: versioned download and versioned logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb


def download_artifact_cached(artifact: Any) -> Path:
    """Use W&B's own versioned download cache; no custom checksum machinery."""
    return Path(artifact.download())


def download_model_dir(run: Any) -> Path:
    """Local dir of a run's ``:latest`` model-checkpoint artifact (versioned)."""
    artifact = next(
        (a for a in run.logged_artifacts() if a.type == "model" and "latest" in a.aliases),
        None,
    )
    if artifact is None:
        raise RuntimeError(f"No 'model' artifact found for run {run.id}")
    return download_artifact_cached(artifact)


def _version_index(artifact: Any) -> int:
    return int(str(artifact.version).lstrip("v"))


def log_model_artifact(files: list[str], keep_last: int = 5, milestone_every: int = 25) -> None:
    """Log ``files`` as a new version of the run's ``model`` checkpoint artifact.

    Named ``model-{run.id}``: artifact names are project-level keys and run names
    may repeat. Keeps the newest ``keep_last`` versions, every
    ``milestone_every``-th version, and any aliased version; deletes the rest.
    """
    assert wandb.run is not None, "log_model_artifact needs an active wandb run"
    try:
        artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
        for path in files:
            artifact.add_file(path)
        wandb.run.log_artifact(artifact)
        collection = f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}"
        versions = sorted(wandb.Api().artifacts("model", collection), key=_version_index)
        for stale in versions[:-keep_last]:
            if _version_index(stale) % milestone_every != 0 and not stale.aliases:
                stale.delete()
    except Exception as e:  # wandb failures must never kill training
        print(f"[WARN] Checkpoint artifact log/prune failed: {type(e).__name__}: {e}")
