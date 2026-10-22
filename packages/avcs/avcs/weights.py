"""AVCS Pre-Trained Weights Management.

Handles automatic discovery, local caching, and loading of pre-trained
Artificial Visual Cortex neural network checkpoints.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch

# There is NO public release of pre-trained AVCS cortex weights. The previous
# value here pointed at a repository URL that returns HTTP 404, so
# `download_weights()` could only ever fail (and, with a permissive caller,
# silently leave the cortex at random initialisation while appearing
# "configured"). Leave this unset unless a real, reachable release exists, and
# always train or supply a checkpoint explicitly.
DEFAULT_WEIGHTS_URL: str | None = None

# Standard filename
WEIGHTS_FILENAME = "avcs_cortex_weights.pt"


def get_default_cache_dir() -> Path:
    """Returns the user cache directory for AVCS neural weights."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        cache_dir = Path(base) / "AVCS" / "weights"
    else:
        cache_dir = Path(os.path.expanduser("~/.cache/avcs/weights"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_weights_path(custom_path: str | None = None) -> str | None:
    """Finds the path to pre-trained AVCS weights.

    Search priority:
    1. Explicit custom_path argument
    2. AVCS_WEIGHTS_PATH environment variable
    3. Local workspace runs directory: `runs/weights/avcs_cortex_weights.pt`
    4. Bundled package directory (if installed)
    5. User cache directory: `~/.cache/avcs/weights/avcs_cortex_weights.pt`
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_path = os.environ.get("AVCS_WEIGHTS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Workspace runs/ directory
    workspace_path = Path(__file__).resolve().parent.parent / "runs" / "weights" / WEIGHTS_FILENAME
    if workspace_path.is_file():
        return str(workspace_path)

    # Relative to current working directory
    cwd_path = Path.cwd() / "runs" / "weights" / WEIGHTS_FILENAME
    if cwd_path.is_file():
        return str(cwd_path)

    # Bundled within avcs package
    bundled_path = Path(__file__).resolve().parent / WEIGHTS_FILENAME
    if bundled_path.is_file():
        return str(bundled_path)

    # User cache directory
    cached_path = get_default_cache_dir() / WEIGHTS_FILENAME
    if cached_path.is_file():
        return str(cached_path)

    return None


def download_weights(
    dest_path: str | None = None,
    url: str | None = DEFAULT_WEIGHTS_URL,
    verbose: bool = True,
) -> str:
    """Download pre-trained AVCS weights from a release URL.

    There is currently no published release, so the default URL is unset and
    this raises rather than pretending to fetch weights that do not exist.
    """
    if not url:
        raise RuntimeError(
            "No AVCS weights release URL is configured; train a checkpoint "
            "(forge.vision.cortex.training.trainer) or pass an explicit url."
        )
    target = Path(dest_path) if dest_path else get_default_cache_dir() / WEIGHTS_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[AVCS] Downloading pre-trained weights from {url} to {target}...")

    urllib.request.urlretrieve(url, str(target))

    if verbose:
        print(f"[AVCS] Download complete ({target.stat().st_size:,} bytes).")

    return str(target)


def load_avcs_weights(
    cortex: torch.nn.Module,
    path: str | None = None,
    device: str = "cpu",
    strict: bool = False,
    auto_download: bool = False,
) -> bool:
    """Loads pre-trained weights into the cortex model if available.

    Returns True if weights were loaded, False otherwise.
    """
    resolved_path = get_weights_path(path)
    if resolved_path is None and auto_download:
        try:
            resolved_path = download_weights()
        except Exception as exc:
            print(f"[AVCS] Notice: Could not download weights ({exc}). Proceeding with baseline weights.")
            return False

    if resolved_path and os.path.isfile(resolved_path):
        try:
            state = torch.load(resolved_path, map_location=device)
            cortex.load_state_dict(state, strict=strict)
            return True
        except Exception as exc:
            print(f"[AVCS] Warning: Failed to load weights from {resolved_path}: {exc}")
            return False

    return False

