# src/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd

from .utils import (
    ProjectPaths,
    get_project_root,
    load_csv,
    save_csv,
    basic_clean_df,
    infer_target_column,
    normalize_target_to_binary,
    save_json,
)


def preprocess_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    metadata_json: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw churn dataset, clean, normalize target to 0/1, and save processed dataset.
    Returns processed df and metadata.
    """
    df = load_csv(input_csv)
    df = basic_clean_df(df)

    target_col = infer_target_column(df)
    df[target_col] = normalize_target_to_binary(df[target_col])

    # Drop ID-like columns if present (common in Telco)
    drop_candidates = ["customerID", "CustomerID", "customer_id", "customerId", "Row ID", "RowID"]
    dropped = []
    for c in drop_candidates:
        if c in df.columns:
            df = df.drop(columns=[c])
            dropped.append(c)

    # Some Telco datasets have TotalCharges as object with spaces; try numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Reorder: target last
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]

    meta = {
        "target_col": target_col,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dropped_columns": dropped,
    }

    save_csv(df, output_csv, index=False)
    save_json(meta, metadata_json)
    return df, meta


def main() -> None:
    root = get_project_root()
    paths = ProjectPaths(root)
    paths.ensure()

    input_csv = paths.data_raw / "telco_customer_churn.csv"
    output_csv = paths.data_processed / "churn_processed.csv"
    metadata_json = paths.data_processed / "metadata.json"

    df, meta = preprocess_dataset(input_csv, output_csv, metadata_json)
    print("✅ Preprocess done.")
    print(meta)
    print(df.head())


if __name__ == "__main__":
    main()
