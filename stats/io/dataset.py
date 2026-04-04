from __future__ import annotations

import csv
import gzip
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd


class DatasetError(RuntimeError):
    pass


class DatasetIntegrityError(DatasetError):
    pass


@dataclass(frozen=True)
class DayMetadata:
    exchange: Optional[str]
    symbol: Optional[str]
    day: Optional[str]
    schema_version: Optional[int]
    created_utc: Optional[str]


@dataclass(frozen=True)
class InstrumentMetadata:
    exchange: Optional[str]
    symbol: Optional[str]
    base_asset: Optional[str]
    quote_asset: Optional[str]
    asset_source: Optional[str]
    tick_size: Optional[str]
    tick_size_source: Optional[str]


DEFAULT_BINANCE_QUOTE_ASSETS: tuple[str, ...] = (
    "USDT",
    "USDC",
    "FDUSD",
    "BUSD",
    "TUSD",
    "DAI",
    "BTC",
    "ETH",
    "BNB",
    "TRY",
    "EUR",
    "GBP",
    "BRL",
    "AUD",
    "RUB",
    "ZAR",
    "IDRT",
    "NGN",
    "UAH",
    "VAI",
)


@dataclass(frozen=True)
class DatasetPaths:
    schema_path: Path
    events_path: Optional[Path]
    gaps_path: Optional[Path]
    book_path: Optional[Path]
    trades_path: Optional[Path]
    diff_paths: tuple[Path, ...]
    snapshot_csv_paths: tuple[Path, ...]
    snapshot_json_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LedgerEvent:
    event_id: int
    recv_time_ms: int
    recv_seq: int
    run_id: int
    typ: str
    epoch_id: int
    details: dict[str, Any]


@dataclass(frozen=True)
class TradeRow:
    event_time_ms: int
    recv_time_ms: int
    recv_seq: int
    run_id: int
    trade_id: int
    trade_time_ms: int
    price: float
    qty: float
    is_buyer_maker: int
    side: Optional[str]
    ord_type: Optional[str]
    exchange: Optional[str]
    symbol: Optional[str]


@dataclass(frozen=True)
class ReplaySegment:
    index: int
    tag: str
    event_id: int
    recv_seq: int
    epoch_id: int
    snapshot_csv_path: Path
    snapshot_json_path: Optional[Path]
    end_recv_seq: Optional[int]


def infer_meta_from_day_dir(day_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    parts = day_dir.resolve().parts
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    return None, None, None


def _resolve_under_day(day_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return day_dir / path


def _resolve_recorded_path(day_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates: list[Path] = [day_dir / path]
    parts = path.parts
    if parts and parts[0] == "data":
        if len(day_dir.parents) >= 2:
            exchange_root = day_dir.parents[1]
            candidates.append(exchange_root / Path(*parts[1:]))
        data_root = day_dir
        while data_root.name != "data" and data_root.parent != data_root:
            data_root = data_root.parent
        if data_root.name == "data":
            candidates.append(data_root / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return day_dir / path


def _glob_sorted(day_dir: Path, pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(day_dir.glob(pattern)))


def _normalize_schema_files(day_dir: Path, schema: dict[str, Any]) -> DatasetPaths:
    files = schema.get("files") or {}
    events_path: Optional[Path] = None
    gaps_path: Optional[Path] = None
    book_path: Optional[Path] = None
    trades_path: Optional[Path] = None
    diff_paths: tuple[Path, ...] = ()
    snapshot_csv_paths: tuple[Path, ...] = ()
    snapshot_json_paths: tuple[Path, ...] = ()

    if isinstance(files, dict):
        role_map = {
            "events_csv": "events_path",
            "gaps_csv": "gaps_path",
            "orderbook_ws_depth_csv": "book_path",
            "trades_ws_csv": "trades_path",
        }
        resolved: dict[str, Optional[Path]] = {
            "events_path": None,
            "gaps_path": None,
            "book_path": None,
            "trades_path": None,
        }
        for role, target in role_map.items():
            entry = files.get(role)
            if isinstance(entry, dict) and entry.get("path"):
                resolved[target] = _resolve_under_day(day_dir, str(entry["path"]))
        events_path = resolved["events_path"]
        gaps_path = resolved["gaps_path"]
        book_path = resolved["book_path"]
        trades_path = resolved["trades_path"]

        diff_entry = files.get("depth_diffs_ndjson_gz")
        if isinstance(diff_entry, dict) and diff_entry.get("path"):
            diff_paths = (_resolve_under_day(day_dir, str(diff_entry["path"])),)

    if events_path is None:
        hits = _glob_sorted(day_dir, "events_*.csv.gz")
        events_path = hits[0] if hits else None
    if gaps_path is None:
        hits = _glob_sorted(day_dir, "gaps_*.csv.gz")
        gaps_path = hits[0] if hits else None
    if book_path is None:
        hits = _glob_sorted(day_dir, "orderbook_ws_depth_*.csv.gz")
        book_path = hits[0] if hits else None
    if trades_path is None:
        hits = _glob_sorted(day_dir, "trades_ws_*.csv.gz")
        trades_path = hits[0] if hits else None
    if not diff_paths:
        diff_paths = _glob_sorted(day_dir / "diffs", "*.ndjson*") if (day_dir / "diffs").exists() else ()

    snapshot_csv_paths = _glob_sorted(day_dir / "snapshots", "*.csv") if (day_dir / "snapshots").exists() else ()
    snapshot_json_paths = _glob_sorted(day_dir / "snapshots", "*.json") if (day_dir / "snapshots").exists() else ()

    return DatasetPaths(
        schema_path=day_dir / "schema.json",
        events_path=events_path,
        gaps_path=gaps_path,
        book_path=book_path,
        trades_path=trades_path,
        diff_paths=diff_paths,
        snapshot_csv_paths=snapshot_csv_paths,
        snapshot_json_paths=snapshot_json_paths,
    )


def _parse_event_details(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetIntegrityError(f"Invalid details_json: {exc}") from exc
    if isinstance(parsed, dict):
        return parsed
    return {}


def _read_tabular(path: Optional[Path], *, label: str) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DatasetIntegrityError(f"Failed reading {label} {path}: {exc}") from exc


class DayDataset:
    def __init__(
        self,
        day_dir: Path,
        metadata: DayMetadata,
        paths: DatasetPaths,
        instrument: InstrumentMetadata | None = None,
    ) -> None:
        self.day_dir = day_dir
        self.metadata = metadata
        self.paths = paths
        self.instrument = instrument

    @property
    def exchange(self) -> Optional[str]:
        return self.metadata.exchange

    @property
    def symbol(self) -> Optional[str]:
        return self.metadata.symbol

    @property
    def day(self) -> Optional[str]:
        return self.metadata.day

    def load_book(self) -> Optional[pd.DataFrame]:
        return _read_tabular(self.paths.book_path, label="book csv")

    def load_trades(self) -> Optional[pd.DataFrame]:
        return _read_tabular(self.paths.trades_path, label="trades csv")

    def load_events(self) -> Optional[pd.DataFrame]:
        return _read_tabular(self.paths.events_path, label="events csv")

    def load_gaps(self) -> Optional[pd.DataFrame]:
        return _read_tabular(self.paths.gaps_path, label="gaps csv")

    def iter_events(self) -> Iterator[LedgerEvent]:
        path = self.paths.events_path
        if path is None:
            return
        if not path.exists():
            raise FileNotFoundError(f"events csv not found: {path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header:
                    return
                for row in reader:
                    if not row:
                        continue
                    if len(row) < 6:
                        raise DatasetIntegrityError(f"Malformed event row in {path}: {row!r}")
                    details = _parse_event_details(row[6]) if len(row) > 6 else {}
                    yield LedgerEvent(
                        event_id=int(row[0]),
                        recv_time_ms=int(row[1]),
                        recv_seq=int(row[2]),
                        run_id=int(row[3]),
                        typ=str(row[4]),
                        epoch_id=int(row[5]),
                        details=details,
                    )
        except DatasetIntegrityError:
            raise
        except Exception as exc:
            raise DatasetIntegrityError(f"Failed reading events {path}: {exc}") from exc

    def iter_depth_diffs(self) -> Iterator[dict[str, Any]]:
        if not self.paths.diff_paths:
            return
        for path in self.paths.diff_paths:
            opener = gzip.open if path.suffix == ".gz" else open
            try:
                with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
                    for lineno, line in enumerate(handle, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise DatasetIntegrityError(f"Invalid diff JSON at {path}:{lineno}: {exc}") from exc
                        if not isinstance(payload, dict):
                            raise DatasetIntegrityError(f"Diff payload must be object at {path}:{lineno}")
                        yield payload
            except DatasetIntegrityError:
                raise
            except Exception as exc:
                raise DatasetIntegrityError(f"Failed reading diffs {path}: {exc}") from exc

    def iter_trades(self) -> Iterator[TradeRow]:
        path = self.paths.trades_path
        if path is None:
            return
        if not path.exists():
            raise FileNotFoundError(f"trades csv not found: {path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    return
                for row in reader:
                    if not row:
                        continue
                    side_raw = row.get("side")
                    side = str(side_raw).lower() if side_raw not in (None, "") else None
                    if side is None and row.get("is_buyer_maker") not in (None, ""):
                        side = "sell" if int(row["is_buyer_maker"]) == 1 else "buy"
                    yield TradeRow(
                        event_time_ms=int(row["event_time_ms"]),
                        recv_time_ms=int(row["recv_time_ms"]),
                        recv_seq=int(row["recv_seq"]),
                        run_id=int(row["run_id"]),
                        trade_id=int(row["trade_id"]),
                        trade_time_ms=int(row["trade_time_ms"]),
                        price=float(row["price"]),
                        qty=float(row["qty"]),
                        is_buyer_maker=int(row["is_buyer_maker"]),
                        side=side,
                        ord_type=(str(row["ord_type"]) if row.get("ord_type") not in (None, "") else None),
                        exchange=(str(row["exchange"]) if row.get("exchange") not in (None, "") else None),
                        symbol=(str(row["symbol"]) if row.get("symbol") not in (None, "") else None),
                    )
        except Exception as exc:
            raise DatasetIntegrityError(f"Failed reading trades {path}: {exc}") from exc

    def build_segments(self) -> list[ReplaySegment]:
        events = list(self.iter_events())
        resync_starts = sorted(event.recv_seq for event in events if event.typ == "resync_start")
        snapshot_starts = sorted(event.recv_seq for event in events if event.typ == "snapshot_loaded")

        segments: list[ReplaySegment] = []
        for event in events:
            if event.typ != "snapshot_loaded":
                continue
            tag = str(event.details.get("tag", "snapshot"))
            snapshot_csv_path = self._resolve_snapshot_csv_path(event.event_id, tag, event.details)
            snapshot_json_path = self._resolve_snapshot_json_path(event.event_id, tag, event.details, snapshot_csv_path)

            next_resync = None
            next_snapshot = None
            idx_resync = bisect_right(resync_starts, event.recv_seq)
            if idx_resync < len(resync_starts):
                next_resync = resync_starts[idx_resync]
            idx_snapshot = bisect_right(snapshot_starts, event.recv_seq)
            if idx_snapshot < len(snapshot_starts):
                next_snapshot = snapshot_starts[idx_snapshot]
            candidates = [value for value in (next_resync, next_snapshot) if value is not None]
            end_recv_seq = min(candidates) if candidates else None

            segments.append(
                ReplaySegment(
                    index=len(segments),
                    tag=tag,
                    event_id=event.event_id,
                    recv_seq=event.recv_seq,
                    epoch_id=event.epoch_id,
                    snapshot_csv_path=snapshot_csv_path,
                    snapshot_json_path=snapshot_json_path,
                    end_recv_seq=end_recv_seq,
                )
            )

        return segments

    def _resolve_snapshot_csv_path(self, event_id: int, tag: str, details: dict[str, Any]) -> Path:
        if details.get("path"):
            return _resolve_recorded_path(self.day_dir, str(details["path"]))
        return self.day_dir / "snapshots" / f"snapshot_{event_id:06d}_{tag}.csv"

    def _resolve_snapshot_json_path(
        self,
        event_id: int,
        tag: str,
        details: dict[str, Any],
        snapshot_csv_path: Path,
    ) -> Optional[Path]:
        raw_path = details.get("raw_path")
        if raw_path:
            resolved = _resolve_recorded_path(self.day_dir, str(raw_path))
            if resolved.exists():
                return resolved
        inferred = snapshot_csv_path.with_suffix(".json")
        if inferred.exists():
            return inferred
        return None


def load_day(day_dir: Path) -> DayDataset:
    day_dir = Path(day_dir).resolve()
    schema_path = day_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.json not found under {day_dir}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetIntegrityError(f"Invalid schema.json in {day_dir}: {exc}") from exc

    exchange, symbol, day = infer_meta_from_day_dir(day_dir)
    metadata = DayMetadata(
        exchange=exchange,
        symbol=symbol,
        day=day,
        schema_version=(int(schema["schema_version"]) if schema.get("schema_version") is not None else None),
        created_utc=(str(schema["created_utc"]) if schema.get("created_utc") is not None else None),
    )
    instrument = None
    instrument_raw = schema.get("instrument")
    if isinstance(instrument_raw, dict):
        instrument = InstrumentMetadata(
            exchange=(str(instrument_raw["exchange"]) if instrument_raw.get("exchange") is not None else None),
            symbol=(str(instrument_raw["symbol"]) if instrument_raw.get("symbol") is not None else None),
            base_asset=(str(instrument_raw["base_asset"]) if instrument_raw.get("base_asset") is not None else None),
            quote_asset=(str(instrument_raw["quote_asset"]) if instrument_raw.get("quote_asset") is not None else None),
            asset_source=(str(instrument_raw["asset_source"]) if instrument_raw.get("asset_source") is not None else None),
            tick_size=(str(instrument_raw["tick_size"]) if instrument_raw.get("tick_size") is not None else None),
            tick_size_source=(
                str(instrument_raw["tick_size_source"]) if instrument_raw.get("tick_size_source") is not None else None
            ),
        )
    paths = _normalize_schema_files(day_dir, schema)
    return DayDataset(day_dir=day_dir, metadata=metadata, paths=paths, instrument=instrument)
