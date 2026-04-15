'''

Обучение/дообучение модели. Функционал: построение моделей ML (LR,
kNN или дерево решений).

'''

import pandas as pd
import json
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import load_config


SUPPORTED_SCALERS = ['minmax', 'standard']
SUPPORTED_MODELS = ['lr', 'knn', 'dt']

HPARAMS = {
    'lr': {},
    'knn': {"n_neighbors": 5, "weights": "distance"},
    'dt': {"max_depth": 12, "min_samples_leaf": 5}
}


def _get_regressor(model_name: str, hparams: Dict[str, Any], seed: int):
    params = dict(hparams or {})
    if model_name == 'lr':
        params.setdefault("loss", "squared_error")
        params.setdefault("penalty", None)
        params.setdefault("random_state", seed)
        return SGDRegressor(**params)
    if model_name == 'knn':
        return KNeighborsRegressor(**params)
    if model_name == 'dt':
        params.setdefault("random_state", seed)
        return DecisionTreeRegressor(**params)
    raise ValueError(f"Unrecognized model name: {model_name}")


def _get_scaler(scaler_name: str):
    if scaler_name == "minmax":
        return MinMaxScaler()
    if scaler_name == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler name: {scaler_name}. Supported: {SUPPORTED_SCALERS}")


def _build_pipeline(model_name: str, scaler_name: str, X_train: pd.DataFrame, hparams: Dict[str, Any], seed: int) -> Pipeline:
    regressor = _get_regressor(model_name, hparams, seed)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", _get_scaler(scaler_name))
                    ]
                ),
                num_cols
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ]
                ),
                cat_cols
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    losses = {
        'train_mae': float(mean_absolute_error(y_train, train_preds)),
        'train_mse': float(mean_squared_error(y_train, train_preds)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_preds))),
        'train_r2': float(r2_score(y_train, train_preds)),
        'test_mae': float(mean_absolute_error(y_test, test_preds)),
        'test_mse': float(mean_squared_error(y_test, test_preds)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_preds))),
        'test_r2': float(r2_score(y_test, test_preds))
    }
    return losses


def fit_model(model_name: str, scaler_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                hparams: Dict[str, Any], seed: int) -> Tuple[Pipeline, Dict[str, float]]:
    model = _build_pipeline(model_name, scaler_name, X_train, hparams, seed)
    model.fit(X_train, y_train)
    losses = evaluate_model(model, X_train, X_test, y_train, y_test)
    return model, losses


def search_parameters(model_name: str, scaler_name: str, X_train: pd.DataFrame,
                      y_train: pd.Series, seed: int) -> Tuple[Dict[str, Any], float]:
    model = _build_pipeline(model_name, scaler_name, X_train, hparams={}, seed=seed)

    if model_name == "lr":
        param_grid = {
            "regressor__alpha": [1e-6, 1e-5, 1e-4],
            "regressor__eta0": [1e-3, 1e-2, 1e-1],
            "regressor__learning_rate": ["optimal", "adaptive"],
            "regressor__fit_intercept": [True, False]
        }
    elif model_name == "knn":
        param_grid = {
            "regressor__n_neighbors": [3, 5, 11, 21],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2]
        }
    elif model_name == "dt":
        param_grid = {
            "regressor__max_depth": [5, 10, 20, None],
            "regressor__min_samples_split": [2, 10, 20],
            "regressor__min_samples_leaf": [1, 5, 10]
        }
    else:
        raise ValueError(f"Unrecognized model name: {model_name}")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_params = {}
    for key, value in search.best_params_.items():
        best_params[key.replace("regressor__", "")] = value

    best_cv_mae = float(-search.best_score_)
    return best_params, best_cv_mae


def finetune_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    X_new = preprocessor.transform(X_train)
    y_new = np.asarray(y_train, dtype=float).reshape(-1)

    if hasattr(regressor, "partial_fit"):
        regressor.partial_fit(X_new, y_new)
        return model

    raise NotImplementedError(
        f"Regressor {type(regressor).__name__} does not support partial_fit. "
        "Use model 'lr' (SGD-based) for incremental finetune."
    )


def load_params(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def save_model(model: Pipeline, path: str) -> str:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return str(save_path)


def train_model(config, X_train, X_test, y_train, y_test):
    cfg = load_config(config)

    model_name = cfg.train_model.model
    scaler_name = cfg.train_model.scaler
    log_dir = cfg.logging.folder
    hparams = cfg.train_model.hparams
    
    search_mode = cfg.model.search_mode
    prev_path = cfg.train_model.prev_path
    seed = cfg.seed

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")
    if scaler_name not in SUPPORTED_SCALERS:
        raise ValueError(f"Unsupported scaler '{scaler_name}'. Supported: {SUPPORTED_SCALERS}")

    if not hparams:
        hparams = dict(HPARAMS.get(model_name, {}))

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if search_mode:
        best_params, best_cv_mae = search_parameters(model_name, scaler_name, X_train, y_train, seed)
        hparams.update(best_params)
    else:
        best_cv_mae = None

    if prev_path:
        model = load_model(prev_path)
        model = finetune_model(model, X_train, y_train)
        losses = evaluate_model(model, X_train, X_test, y_train, y_test)
    else:
        model, losses = fit_model(model_name, scaler_name, X_train, X_test, y_train, y_test, hparams, seed)

    if best_cv_mae is not None:
        losses["best_cv_mae"] = best_cv_mae

    output_dir = Path(log_dir) / f"train_report_{train_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_model(model, str(output_dir / f"model_{model_name}.joblib"))

    summary = {
        "run_time": train_ts,
        "scaler": scaler_name,
        "model": model_name,
        "search_mode": search_mode,
        "metrics": losses,
        "hparams": hparams
    }
    summary_path = output_dir / "train_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "summary_path": str(summary_path),
        "model_path": model_path
    }, ensure_ascii=False))

    return model, losses

