"""Local repository and Git context dynamic inferrer for zero-flag execution.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
Infers workspace_id, tenant_id, branch, and session_id from local git environment
without requiring any user-facing flags.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalWorkspaceContext:
    repo_root: Path
    branch: str
    tenant_id: str
    workspace_id: str
    session_id: str
    is_git: bool


def _git_root(directory: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip()).resolve()


def resolve_local_context(cwd: Path | str | None = None) -> LocalWorkspaceContext:
    """Infer workspace context from cwd, falling back to the installed source checkout."""
    working_dir = Path(cwd or Path.cwd()).resolve()

    configured_root = os.environ.get("SONIC_REPO_ROOT", "").strip()
    source_checkout = Path(__file__).resolve().parents[3]
    if configured_root:
        repo_root = _git_root(Path(configured_root).expanduser().resolve())
        if repo_root is None:
            raise ValueError("SONIC_REPO_ROOT must point to an existing Git checkout")
    else:
        if working_dir == Path.home().resolve():
            candidate_sonic = working_dir / "sonic"
            repo_root = (
                _git_root(source_checkout)
                or (_git_root(candidate_sonic) if candidate_sonic.exists() else None)
                or _git_root(working_dir)
            )
        else:
            repo_root = _git_root(working_dir) or _git_root(source_checkout)
    is_git = repo_root is not None
    if repo_root is None:
        repo_root = working_dir

    # 2. Resolve branch
    branch = "main"
    if is_git:
        b_proc = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if b_proc.returncode == 0 and b_proc.stdout.strip():
            branch = b_proc.stdout.strip()

    # 3. Infer tenant_id (Zero hardcoded values!)
    tenant_id = ""
    if is_git:
        # Try remote.origin.url
        try:
            r_proc = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            url = r_proc.stdout.strip()
            if url:
                # e.g. git@github.com:nandk/sonic.git or https://github.com/nandk/sonic
                match = re.search(r"[:/]([a-zA-Z0-9_\-]+)/[a-zA-Z0-9_\-]+(?:\.git)?$", url)
                if match:
                    tenant_id = f"tenant-{match.group(1).lower()}"
        except Exception:
            pass

        # Fallback to git user.email domain
        if not tenant_id:
            try:
                e_proc = subprocess.run(
                    ["git", "config", "--get", "user.email"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                email = e_proc.stdout.strip()
                if "@" in email:
                    domain = email.split("@")[-1].split(".")[0].lower()
                    tenant_id = f"tenant-{domain}"
            except Exception:
                pass

    if not tenant_id:
        repo_hash = hashlib.sha256(str(repo_root).encode()).hexdigest()[:8]
        tenant_id = f"tenant-{repo_hash}"

    # 4. Infer workspace_id
    workspace_hash = hashlib.sha256(str(repo_root).encode()).hexdigest()[:10]
    workspace_id = f"ws-{workspace_hash}"

    # 5. Infer unique session_id
    clean_branch = re.sub(r"[^a-zA-Z0-9_\-]", "-", branch)
    session_id = f"sess-{clean_branch}-{int(time.time())}"

    return LocalWorkspaceContext(
        repo_root=repo_root,
        branch=branch,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        session_id=session_id,
        is_git=is_git,
    )


def get_token(token: str = "") -> str:
    """Retrieve auth token from flag, env var, or ~/.sonic/token file."""
    if token:
        return token
    if os.environ.get("SONIC_TOKEN"):
        return os.environ["SONIC_TOKEN"].strip()
    token_file = Path.home() / ".sonic" / "token"
    if token_file.exists():
        try:
            return token_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""
