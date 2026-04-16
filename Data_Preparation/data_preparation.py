# preprocessing.py
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler


from utils import *

TARGET_COL = "CLAIM_PAID"

DROP_COLS = [
    "OBJECT_ID",
    "insertion_date",
    "source_name",
    "INSR_BEGIN",
    "INSR_END",
]

NUMERIC_COLS = [
    "INSURED_VALUE",
    "PREMIUM",
    "PROD_YEAR",
    "SEATS_NUM",
    "CARRYING_CAPACITY",
    "CCM_TON",
    "EFFECTIVE_YR",
    "insr_duration",
    "prem_insr_ratio",
]

CATEGORICAL_COLS = [
    "SEX",
    "INSR_TYPE",
    "TYPE_VEHICLE",
    "MAKE",
    "USAGE",
    "insr_begin_season",
    "insr_end_season",
]


SUPPORTED_SCALERS = ["minmax", "standard"]

def _get_scaler(scaler_name: str):
    if scaler_name == "minmax":
        return MinMaxScaler()
    if scaler_name == "standard":
        return StandardScaler()
    raise ValueError(
        f"Unsupported scaler name: {scaler_name}. Supported: {SUPPORTED_SCALERS}"
    )

def build_preprocessor(scaler_name: str = "standard"):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", _get_scaler(scaler_name)),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def prepare(config):
    preparation_params = load_config(config)["data_preparation"]
    clean_db          = preparation_params["clean_db"]
    clean_table       = preparation_params["clean_table"]
    output_path       = preparation_params["output_path"]
    preprocessor_path = preparation_params["preprocessor_path"]
    scaler_name       = preparation_params.get("scaler_name", "standard")

    df = load_data_from_db(clean_db, table=clean_table)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Таргет '{TARGET_COL}' не найден в {clean_table}")

    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])
    X = X.replace([np.inf, -np.inf], np.nan)

    preprocessor = build_preprocessor(scaler_name=scaler_name)
    X_transformed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_prepared = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    X_prepared[TARGET_COL] = y.values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    X_prepared.to_parquet(output_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Препроцессинг: {X_prepared.shape[0]} строк, {X_prepared.shape[1]-1} фич")
    print(f"  данные    → {output_path}")
    print(f"  препроц.  → {preprocessor_path}")

    return X_prepared, preprocessor
