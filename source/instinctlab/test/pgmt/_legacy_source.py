"""Load comparison oracles from an explicitly selected external source snapshot."""

import importlib
import os
import sys
import types
from pathlib import Path


def legacy_module(module):
    root = Path(os.environ["PGMT_LEGACY_SOURCE"]) / "source/instinctlab/instinctlab/tasks/shadowing/pgmt"
    name = "instinctlab.tasks.shadowing._pgmt_legacy"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(name + "." + module)
