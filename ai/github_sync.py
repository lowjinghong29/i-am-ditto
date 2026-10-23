"""
GitHub Asset Discovery & Automatic Repository Sync for i-am-ditto.

1. Searches public GitHub repositories for pre-made character face masks / textures.
2. Automatically commits and pushes newly discovered/generated avatars in `avatars/` to
   https://github.com/lowjinghong29/i-am-ditto in a non-blocking background thread.
"""

import json
import subprocess
import threading
import urllib.parse
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
_git_lock = threading.Lock()


def search_github_mask_urls(character_name: str, limit: int = 4) -> List[str]:
    """
    Use GitHub CLI (`gh api`) to search public GitHub repositories for PNG mask/face assets
    matching `character_name`. Converts matching blob URLs into raw.githubusercontent.com URLs.
    """
    urls: List[str] = []
    try:
        clean = character_name.strip().replace('"', "")
        query = urllib.parse.quote(f"{clean} mask OR face extension:png")
        cmd = [
            "gh",
            "api",
            f"search/code?q={query}&per_page={limit}",
            "--jq",
            ".items[].html_url",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("https://github.com/") and "/blob/" in line:
                    raw_url = line.replace("https://github.com/", "https://raw.githubusercontent.com/").replace(
                        "/blob/", "/"
                    )
                    urls.append(raw_url)
    except Exception:
        pass
    return urls


def auto_push_avatar_async(slug: str, character_name: str, on_status=None) -> None:
    """
    Automatically stage, commit, and push a newly cached avatar in `avatars/<slug>` to GitHub
    in a background thread so the user's GitHub repo builds a shared cloud library automatically.
    """
    def _worker():
        with _git_lock:
            try:
                rel_path = f"avatars/{slug}"
                subprocess.run(
                    ["git", "add", rel_path],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    timeout=8.0,
                )
                status = subprocess.run(
                    ["git", "status", "--porcelain", rel_path],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if not status.stdout.strip():
                    return

                commit_msg = f"feat(avatars): auto-sync '{character_name}' AR filter to library"
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    timeout=10.0,
                )
                push_proc = subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    timeout=15.0,
                )
                if push_proc.returncode == 0 and on_status:
                    on_status(f"Active: {character_name} (Synced to GitHub)")
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()
