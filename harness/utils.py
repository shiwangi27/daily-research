"""Shared utilities: seeding, environment capture, hashing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from typing import Any


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "uncommitted"


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config dict, for tracking identical runs."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def capture_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_sha": git_sha(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["num_threads"] = torch.get_num_threads()
    except ImportError:
        env["torch"] = None
    return env


class Timer:
    """Context manager that records wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = round(time.perf_counter() - self._start, 2)
