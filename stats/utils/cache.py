from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd


CACHE_VERSION = "v2"


def _hash_dict(d: Dict[str, Any]) -> str:
    """Create a stable short hash for cache key parameters."""
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def cache_dir(day_dir: Path) -> Path:
    """Return the per-day cache directory, creating it if needed."""
    d = Path(day_dir) / "cache" / CACHE_VERSION
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(day_dir: Path, kind: str, params: Dict[str, Any], ext: str = "parquet") -> Path:
    """Build a deterministic per-day cache path for one artifact kind and parameter set."""
    key = _hash_dict({"kind": kind, "params": params, "cache_version": CACHE_VERSION})
    return cache_dir(day_dir) / f"{kind}_{key}.{ext}"


def analysis_cache_dir(cache_root: Path, namespace: Optional[str] = None) -> Path:
    """Return the shared analysis cache directory, optionally under a namespace."""
    d = Path(cache_root) / "analysis_cache" / CACHE_VERSION
    if namespace:
        d = d / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d


def analysis_cache_path(
    cache_root: Path,
    kind: str,
    params: Dict[str, Any],
    *,
    namespace: Optional[str] = None,
    ext: str = "parquet",
) -> Path:
    """Build a deterministic shared-analysis cache path."""
    key = _hash_dict(
        {
            "kind": kind,
            "params": params,
            "namespace": namespace,
            "cache_version": CACHE_VERSION,
        }
    )
    return analysis_cache_dir(cache_root, namespace=namespace) / f"{kind}_{key}.{ext}"


def has_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    """Return whether a dataframe contains every required column."""
    return set(required_columns) <= set(frame.columns)


def load_or_build_parquet(
    path: Path,
    *,
    build: Callable[[], pd.DataFrame],
    force: bool = False,
    required_columns: Iterable[str] | None = None,
    validator: Callable[[pd.DataFrame], bool] | None = None,
    index: bool = False,
) -> pd.DataFrame:
    """Load a cached parquet if valid, otherwise rebuild and overwrite it.

    Callers may provide a required column set, a custom validator, or both.
    Any read failure or validation failure triggers a rebuild.
    """
    required_columns = tuple(required_columns or ())

    def is_valid(frame: pd.DataFrame) -> bool:
        if required_columns and not has_required_columns(frame, required_columns):
            return False
        if validator is not None and not validator(frame):
            return False
        return True

    if not force and path.exists():
        try:
            cached = pd.read_parquet(path)
        except Exception:
            cached = None
        else:
            if is_valid(cached):
                return cached

    built = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    built.to_parquet(path, index=index)
    return built
