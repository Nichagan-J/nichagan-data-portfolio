# src/utils.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# -------------------------
# Config
# -------------------------
@dataclass
class ProjectPaths:
    project_root: Path

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def outputs(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def models(self) -> Path:
        return self.outputs / "models"

    def ensure(self) -> None:
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Assumes this file is project_3_churn_prediction/src/utils.py
    so project root = parents[1].
    """
    return Path(__file__).resolve().parents[1]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def infer_target_column(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> str:
    """
    Try to locate churn target column.
    Default candidates: ["Churn", "churn", "Exited", "exit", "Target", "target"]
    """
    if candidates is None:
        candidates = ["Churn", "churn", "Exited", "exit", "Target", "target", "Attrition", "attrition"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Target column not found. Please rename your target to 'Churn' "
        "or update candidates in infer_target_column()."
    )


def normalize_target_to_binary(series: pd.Series) -> pd.Series:
    """
    Normalize target to 0/1.
    Supports:
      - Yes/No, Y/N, True/False
      - 0/1 already
      - strings '0','1'
    """
    s = series.copy()

    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(s):
        # convert to int 0/1 if possible
        uniq = set(pd.Series(s.dropna().unique()).astype(float).tolist())
        if uniq.issubset({0.0, 1.0}):
            return s.astype(int)

    # Convert to string lower for mapping
    s_str = s.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0
    }

    mapped = s_str.map(mapping)
    if mapped.isna().any():
        # Try direct factorize if still unknown (last resort)
        # But ensure binary
        vals = s_str.dropna().unique()
        if len(vals) == 2:
            # consistent ordering: map first to 0 second to 1
            v0, v1 = vals[0], vals[1]
            mapping2 = {v0: 0, v1: 1}
            mapped2 = s_str.map(mapping2)
            return mapped2.astype(int)
        raise ValueError(
            "Cannot normalize target to binary. "
            "Found unexpected labels in target column."
        )
    return mapped.astype(int)


def basic_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning safe for most churn datasets.
    - Strip whitespace from column names
    - Convert blank strings to NaN
    """
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.replace(r"^\s*$", np.nan, regex=True)
    return out


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
