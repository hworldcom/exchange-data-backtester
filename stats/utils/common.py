from __future__ import annotations

from pathlib import Path

import pandas as pd

from stats.io import DayDataset, load_day


def ensure_dataset(dataset_or_day_dir: DayDataset | Path) -> DayDataset:
    """Accept either a loaded dataset or a day-directory path and return a dataset."""
    if isinstance(dataset_or_day_dir, DayDataset):
        return dataset_or_day_dir
    return load_day(dataset_or_day_dir)


def to_utc_datetime_ms(values: pd.Series | pd.Index) -> pd.Series | pd.DatetimeIndex:
    """Convert millisecond timestamps to UTC datetimes without changing shape."""
    return pd.to_datetime(values.astype("int64"), unit="ms", utc=True)
