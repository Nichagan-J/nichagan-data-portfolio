# src/train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .utils import ProjectPaths, get_project_root, load_csv, read_json, save_json


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )
    return preprocess


def evaluate_binary_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def train_baselines(
    X_train: pd.DataFrame, y_train: pd.Series, preprocess: ColumnTransformer
) -> Dict[str, Pipeline]:
    lr = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )),
    ])

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return {"logreg": lr, "rf": rf}


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, preprocess: ColumnTransformer
) -> Pipeline:
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_split": [2, 5],
    }

    # roc_auc is a good general metric for churn ranking
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def main() -> None:
    root = get_project_root()
    paths = ProjectPaths(root)
    paths.ensure()

    processed_csv = paths.data_processed / "churn_processed.csv"
    meta_path = paths.data_processed / "metadata.json"

    df = load_csv(processed_csv)
    meta = read_json(meta_path)
    target_col = meta["target_col"]

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = build_preprocess(X_train)

    # Train baselines
    models = train_baselines(X_train, y_train, preprocess)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_binary_model(model, X_test, y_test)

    # Tune RF
    best_rf = tune_random_forest(X_train, y_train, preprocess)
    results["rf_tuned"] = evaluate_binary_model(best_rf, X_test, y_test)

    # Choose best by roc_auc
    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    best_model = best_rf if best_name == "rf_tuned" else models[best_name]

    # Save model + metrics
    model_path = paths.models / "best_model.pkl"
    joblib.dump(best_model, model_path)

    metrics_path = paths.outputs / "metrics.json"
    save_json(results, metrics_path)

    summary = {
        "best_model": best_name,
        "best_model_path": str(model_path),
        "best_roc_auc": results[best_name]["roc_auc"],
        "best_pr_auc": results[best_name]["pr_auc"],
    }
    save_json(summary, paths.outputs / "summary.json")

    print("✅ Training done.")
    print(summary)


if __name__ == "__main__":
    main()
