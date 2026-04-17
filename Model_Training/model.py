'''

Обучение/дообучение модели. Функционал: построение моделей ML (LR,
kNN или дерево решений).

'''

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from utils import load_config


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


def fit_model(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                hparams: Dict[str, Any], seed: int) -> Tuple[Pipeline, Dict[str, float]]:
    model = _get_regressor(model_name, hparams, seed)
    model.fit(X_train, y_train)
    losses = evaluate_model(model, X_train, X_test, y_train, y_test)
    return model, losses


def search_parameters(model_name: str, X_train: pd.DataFrame,
                      y_train: pd.Series, seed: int) -> Tuple[Dict[str, Any], float]:
    model = _get_regressor(model_name, hparams={}, seed=seed)

    if model_name == "lr":
        param_grid = {
            "alpha": [1e-6, 1e-5, 1e-4],
            "eta0": [1e-3, 1e-2, 1e-1],
            "learning_rate": ["optimal", "adaptive"],
            "fit_intercept": [True, False]
        }
    elif model_name == "knn":
        param_grid = {
            "n_neighbors": [3, 5, 11, 21],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    elif model_name == "dt":
        param_grid = {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10]
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
    best_params = dict(search.best_params_)
    best_cv_mae = float(-search.best_score_)
    return best_params, best_cv_mae


def finetune_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    X_new = X_train if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    y_new = np.asarray(y_train, dtype=float).reshape(-1)

    if hasattr(model, "partial_fit"):
        model.partial_fit(X_new, y_new)
        return model

    if isinstance(model, DecisionTreeRegressor):
        X_array = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
        if X_array.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Feature mismatch for DecisionTreeRegressor: "
                f"model expects {model.n_features_in_}, got {X_array.shape[1]}"
            )
        leaf_ids = model.apply(X_new)
        unique_leaf_ids = np.unique(leaf_ids)
        for leaf_id in unique_leaf_ids:
            mask = (leaf_ids == leaf_id)
            cnt_new = int(mask.sum())
            if cnt_new == 0:
                continue
            sum_new = float(y_new[mask].sum())

            old_cnt = int(model.tree_.n_node_samples[leaf_id])
            old_mean = float(model.tree_.value[leaf_id, 0, 0])
            total_cnt = old_cnt + cnt_new
            new_mean = (old_mean * old_cnt + sum_new) / total_cnt

            model.tree_.value[leaf_id, 0, 0] = new_mean
            model.tree_.n_node_samples[leaf_id] = total_cnt
            model.tree_.weighted_n_node_samples[leaf_id] = (
                model.tree_.weighted_n_node_samples[leaf_id] + float(cnt_new)
            )
        return model

    if isinstance(model, KNeighborsRegressor):
        old_X = model._fit_X
        old_y = np.asarray(model._y, dtype=float).reshape(-1)
        X_array = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
        all_X = np.vstack([old_X, X_array])
        all_y = np.concatenate([old_y, y_new])
        model.fit(all_X, all_y)
        return model

    raise NotImplementedError(
        f"Finetune is not implemented for {type(model).__name__}."
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

    model_name = cfg['train_model']['model']
    log_dir = cfg['logging']['folder']
    hparams = cfg['train_model']['hparams']
    
    search_mode = cfg['train_model']['search_mode']
    prev_path = cfg['train_model']['prev_path']
    seed = cfg['seed']

    save_path = cfg["train_model"]["save_path"]

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")

    if not hparams:
        hparams = dict(HPARAMS.get(model_name, {}))

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if search_mode:
        best_params, best_cv_mae = search_parameters(model_name, X_train, y_train, seed)
        hparams.update(best_params)
    else:
        best_cv_mae = None

    if prev_path:
        model = load_model(prev_path)
        model = finetune_model(model, X_train, y_train)
        losses = evaluate_model(model, X_train, X_test, y_train, y_test)
    else:
        model, losses = fit_model(model_name, X_train, X_test, y_train, y_test, hparams, seed)

    if best_cv_mae is not None:
        losses["best_cv_mae"] = best_cv_mae

    output_dir = Path(log_dir) / f"train_report_{train_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_model(model, str(output_dir / f"model_{model_name}.joblib"))
    model_path = save_model(model, save_path + f"/model_{model_name}.joblib")

    summary = {
        "run_time": train_ts,
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
