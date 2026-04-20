"""Пример обработчика истории для ветростанции."""

from __future__ import annotations

import pandas as pd


def build_history_dataframe(station) -> pd.DataFrame:
    """Собирает историю из CSV-файлов в папке station.auto_history_folder."""
    from pathlib import Path

    root = Path(getattr(station, "auto_history_folder", "") or "")
    if not root.exists():
        return pd.DataFrame(columns=["ds", "power_kw"])

    frames = []
    for file in sorted(root.glob("*.csv")):
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        if {"ds", "power_kw"}.issubset(df.columns):
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["ds", "power_kw"])

    out = pd.concat(frames, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    return out
