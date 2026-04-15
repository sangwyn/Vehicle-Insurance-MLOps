# preprocessing.py
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
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
    clean_db           = preparation_params["clean_db"]
    clean_table        = preparation_params["clean_table"]
    output_path        = preparation_params["output_path"]
    preprocessor_path  = preparation_params["preprocessor_path"]

    df = load_data_from_db(clean_db, table=clean_table)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Таргет '{TARGET_COL}' не найден в {clean_table}")

    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])
    X = X.replace([np.inf, -np.inf], np.nan)

    preprocessor = build_preprocessor()
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
