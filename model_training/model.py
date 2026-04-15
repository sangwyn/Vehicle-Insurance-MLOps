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
from sklearn.linear_model import LinearRegression
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

SUPPORTED_SCALERS = ['minmax', 'standard']
SUPPORTED_MODELS = ['lr', 'knn', 'dt']
DEFAULT_MODELS = SUPPORTED_MODELS
TARGET_COL = 'CLAIM_PAID'

HPARAMS = {
    'lr': {},
    'knn': {"n_neighbors": 5, "weights": "distance"},
    'dt': {"max_depth": 12, "min_samples_leaf": 5}
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/finetune model (LR, KNN, DT)")

    parser.add_argument("--input", type=str, required=True, help="Path to .csv file or dir with .csv files containing data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to dir where reports will be saved.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    def_str = ",".join(DEFAULT_MODELS)
    parser.add_argument("--models", type=str, default=def_str, help=f"Regressors to train.")
    parser.add_argument("--scaler", type=str, default='minmax', help=f"Scaler for data.")
    parser.add_argument("--test_size", type=float, default=0.25, help="Size of test sample.")
    parser.add_argument("--search", action="store_true", help="Hyperparameters optimization mode.")
    parser.add_argument("--params_path", type=str, default=None, help="Optional path to JSON with model hyperparameters.")
    parser.add_argument("--prev_model", type=str, default=None, help="Optional path to an existing model for retraining.")

    return parser.parse_args()


def _parse_model_names(models_raw: str) -> List[str]:
    model_names = [x.strip() for x in models_raw.split(",") if x.strip()]
    unknown = [x for x in model_names if x not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"Unsupported model names: {unknown}. Supported: {SUPPORTED_MODELS}")
    return model_names


def _resolve_input_file(path: str) -> Path:
    input_path = Path(path)
    if input_path.is_file():
        return input_path
    if input_path.is_dir():
        candidates = sorted(input_path.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No .csv files found in directory: {input_path}")

        preferred_order = ["clean_data.csv", "motor_data11-14lats.csv", "ex.csv"]
        for preferred_name in preferred_order:
            for candidate in candidates:
                if candidate.name == preferred_name:
                    return candidate
        return candidates[0]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _maybe_convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            parsed = pd.to_datetime(out[col], format="%d-%b-%y", errors="coerce")
            if parsed.notna().mean() > 0.8:
                out[col] = parsed

    datetime_cols = out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        out[col] = (out[col] - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    return out


def _get_regressor(model_name: str, hparams: Dict[str, Any], seed: int):
    params = dict(hparams or {})
    if model_name == 'lr':
        return LinearRegression(**params)
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


def train_model(model_name: str, scaler_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
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
            "regressor__fit_intercept": [True, False],
            "regressor__positive": [False, True]
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
    model.fit(X_train, y_train)
    return model


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


def prepare_data(path: str, test_size: float = 0.25, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    input_file = _resolve_input_file(path)
    df = pd.read_csv(input_file)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)

    X = df.drop(columns=[TARGET_COL])
    X = _maybe_convert_dates(X)
    y = df[TARGET_COL]

    return train_test_split(X, y, test_size=test_size, random_state=seed)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    model_names = _parse_model_names(args.models)

    if args.scaler not in SUPPORTED_SCALERS:
        raise ValueError(f"Unsupported scaler '{args.scaler}'. Supported: {SUPPORTED_SCALERS}")

    X_train, X_test, y_train, y_test = prepare_data(args.input, args.test_size, args.seed)
    user_params = load_params(args.params_path)

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"train_report_{train_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models: Dict[str, Pipeline] = {}
    all_losses: Dict[str, Dict[str, float]] = {}

    for model_name in model_names:
        hparams = dict(HPARAMS.get(model_name, {}))
        if isinstance(user_params.get(model_name), dict):
            hparams.update(user_params[model_name])

        if args.search:
            best_params, best_cv_mae = search_parameters(model_name, args.scaler, X_train, y_train, args.seed)
            hparams.update(best_params)
        else:
            best_cv_mae = None

        if args.prev_model and len(model_names) == 1:
            model = load_model(args.prev_model)
            model = finetune_model(model, X_train, y_train)
            losses = evaluate_model(model, X_train, X_test, y_train, y_test)
        else:
            model, losses = train_model(model_name, args.scaler, X_train, X_test, y_train, y_test, hparams, args.seed)

        if best_cv_mae is not None:
            losses["best_cv_mae"] = best_cv_mae
        losses["model"] = model_name
        losses["hparams"] = hparams
        all_models[model_name] = model
        all_losses[model_name] = losses

        save_model(model, str(output_dir / f"model_{model_name}.joblib"))

    best_model_name = min(all_losses.keys(), key=lambda k: all_losses[k]["test_mae"])
    best_model_path = save_model(all_models[best_model_name], str(output_dir / "best_model.joblib"))

    summary = {
        "run_time": train_ts,
        "input": args.input,
        "target": TARGET_COL,
        "scaler": args.scaler,
        "models": model_names,
        "best_model": best_model_name,
        "metrics": all_losses
    }
    summary_path = output_dir / "train_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "summary_path": str(summary_path),
        "best_model_path": best_model_path
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
