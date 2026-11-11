"""Bound checkpoint storage for manually stopped long-running experiments."""

from pathlib import Path


def retain_checkpoints(path, every, latest):
    """Prune only numbered checkpoints created in this new experiment directory."""
    files = sorted(Path(path).parent.glob("model_*.pt"), key=lambda p: int(p.stem.removeprefix("model_")))
    keep = set(files[:1] + files[-latest:])
    for checkpoint in files:
        if checkpoint not in keep and int(checkpoint.stem.removeprefix("model_")) % every:
            checkpoint.unlink()
