
"""
Простая валидация только для DecisionTreeRegressor.

Поддерживает:
- hold-out
- CV (KFold)
- версионирование модели в простом реестре
- минимальный quality gate по порогам из config.yaml
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor

from utils import load_config


def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def quality_gate(report_metrics, thresholds):
    reasons = []

    max_mae = thresholds.get("max_mae")
    min_r2 = thresholds.get("min_r2")

    mae_value = report_metrics.get("mae", report_metrics.get("mae_mean"))
    r2_value = report_metrics.get("r2", report_metrics.get("r2_mean"))

    if max_mae is not None and mae_value is not None and mae_value > float(max_mae):
        reasons.append(f"mae>{max_mae}")

    if min_r2 is not None and r2_value is not None and r2_value < float(min_r2):
        reasons.append(f"r2<{min_r2}")

    return ("approved" if not reasons else "rejected", reasons)

def read_registry(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_registry(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def next_version(registry):
    versions = [int(x.get("version", 0)) for x in registry if str(x.get("version", "")).isdigit()]
    return max(versions, default=0) + 1

def validate_tree_model(config_path, model, X_train, y_train, X_test, y_test):
    cfg = load_config(config_path)

    validation_params = cfg["model_validation"]
    scheme       = str(validation_params.get("scheme", "holdout")).lower()
    n_splits     = int(validation_params.get("n_splits", 5))
    registry_dir = Path(validation_params.get("logs", "logs/") + "model_registry/tree")
    thresholds   = validation_params.get("thresholds", {})

    report_metrics = None

    if scheme == "holdout":
        report_metrics = calc_metrics(y_test, model.predict(X_test))

    elif scheme == "cv":
        cv_model = DecisionTreeRegressor(**model.get_params())
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("seed", 42)))

        scores = cross_validate(cv_model, X_train, y_train, cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        cv_mae = -scores["test_mae"]
        cv_mse = -scores["test_mse"]
        cv_r2 = scores["test_r2"]

        report_metrics = {
            "mae_mean": float(np.mean(cv_mae)),
            "mae_std": float(np.std(cv_mae, ddof=1)) if len(cv_mae) > 1 else 0.0,
            "mse_mean": float(np.mean(cv_mse)),
            "rmse_mean": float(np.mean(np.sqrt(cv_mse))),
            "r2_mean": float(np.mean(cv_r2)),
            "r2_std": float(np.std(cv_r2, ddof=1)) if len(cv_r2) > 1 else 0.0,
            "folds": int(n_splits),
        }

    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "registry.json"
    registry = read_registry(registry_path)
    version = next_version(registry)

    status, reasons = quality_gate(report_metrics, thresholds)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = registry_dir / f"v{version:03d}_{ts}"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model_dt.joblib"
    joblib.dump(model, model_path)

    record = {
        "version": version,
        "timestamp": ts,
        "model_type": "DecisionTreeRegressor",
        "scheme": scheme,
        "status": status,
        "model_path": str(model_path),
        "metrics": report_metrics,
        "quality_gate": {"reasons": reasons},
        "hparams": model.get_params(),
    }

    registry.append(record)
    write_registry(registry_path, registry)

    shutil.copy2(model_path, registry_dir / "latest.joblib")
    if status == "approved":
        shutil.copy2(model_path, registry_dir / "best.joblib")

    return record