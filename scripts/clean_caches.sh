#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

remove_dir() {
  local path="$1"
  if [ -e "$path" ]; then
    rm -rf "$path"
  fi
}

# Snapshot tracked cache paths first so we can restore them after deleting
# generated caches. This keeps the working tree clean even if some pyc or
# checkpoint files are still version-controlled in the repository.
tracked_pyc_paths="$(git ls-files -- '*.pyc' || true)"
tracked_checkpoint_paths="$(git ls-files -- '*.ipynb_checkpoints/*' || true)"

# Remove shared cache directories first.
remove_dir ".pytest_cache"
remove_dir "analysis_cache"
remove_dir "__pycache__"

# Remove notebook checkpoint folders that Jupyter may recreate.
find notebooks -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

# Remove Python bytecode caches anywhere in the repo.
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Restore any tracked pyc files that were removed above. If the repository
# still tracks cache artifacts, this prevents the cleanup from leaving deletions
# behind in git status.
if [ -n "$tracked_pyc_paths" ]; then
  printf '%s\n' "$tracked_pyc_paths" | xargs -r git restore --source=HEAD --worktree --staged --
fi
if [ -n "$tracked_checkpoint_paths" ]; then
  printf '%s\n' "$tracked_checkpoint_paths" | xargs -r git restore --source=HEAD --worktree --staged --
fi

echo "Cache cleanup complete."
