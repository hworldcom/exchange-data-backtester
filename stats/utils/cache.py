from __future__ import annotations
from pathlib import Path
import json
import hashlib
from typing import Any, Dict, Optional


CACHE_VERSION = "v2"


def _hash_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def cache_dir(day_dir: Path) -> Path:
    d = Path(day_dir) / "cache" / CACHE_VERSION
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(day_dir: Path, kind: str, params: Dict[str, Any], ext: str = "parquet") -> Path:
    key = _hash_dict({"kind": kind, "params": params, "cache_version": CACHE_VERSION})
    return cache_dir(day_dir) / f"{kind}_{key}.{ext}"


def analysis_cache_dir(cache_root: Path, namespace: Optional[str] = None) -> Path:
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
    key = _hash_dict(
        {
            "kind": kind,
            "params": params,
            "namespace": namespace,
            "cache_version": CACHE_VERSION,
        }
    )
    return analysis_cache_dir(cache_root, namespace=namespace) / f"{kind}_{key}.{ext}"
