# preprocessing.py
import logging
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# Таргет: формулируем как бинарную классификацию — был ли страховой случай
RAW_TARGET_COL = "CLAIM_PAID"
TARGET_COL = "claim_occurred"

DROP_COLS = [
    "OBJECT_ID",   
    "insertion_date",
    "source_name",
    "INSR_BEGIN",
    "INSR_END",
]

DATE_COLS = ["INSR_BEGIN", "INSR_END"]

NUMERIC_COLS = [
    "INSURED_VALUE",
    "PREMIUM",
    "PROD_YEAR",
    "SEATS_NUM",
    "CARRYING_CAPACITY",
    "CCM_TON",
    "EFFECTIVE_YR",
]

CATEGORICAL_COLS = [
    "SEX",
    "INSR_TYPE",
    "TYPE_VEHICLE",
    "MAKE",
    "USAGE",
]

def load_raw_from_db(db_path: str, table: str = "raw_data") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    return df

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Превращает CLAIM_PAID в бинарный таргет claim_occurred.
    """
    df = df.copy()
    df[TARGET_COL] = (df[RAW_TARGET_COL].fillna(0) > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET_COL])
    return df


# ---------- Препроцессор для деревьев ----------

def build_preprocessor() -> ColumnTransformer:
    """
    Препроцессор под деревья (DecisionTree / RandomForest / GradientBoosting).

    Отличия от варианта под LR:
      - нет StandardScaler: деревьям масштабирование не нужно;
      - вместо OneHotEncoder используется OrdinalEncoder: для MAKE с сотнями
        категорий это на порядок компактнее и дереву нормально.
    """

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


# ---------- Основная функция ----------

def prepare_data(
    db_path: str,
    raw_table: str = "raw_data",
    output_path: str = "storage/prepared.parquet",
    preprocessor_path: str = "storage/preprocessor.joblib",
):
    logger.info("=== Начало подготовки данных ===")

    df = load_raw_from_db(db_path, raw_table)

    # 2) сформировать таргет
    df = make_target(df)

    # 3) выкинуть служебные и неинформативные
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    logger.info(f"Фичи: {X.shape[1]} колонок, {X.shape[0]} строк")
    logger.info(f"Баланс классов: {y.value_counts().to_dict()}")
    logger.info(f"Пропусков в сырых фичах: {int(X.isna().sum().sum())}")

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_prepared = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    X_prepared[TARGET_COL] = y.values

    logger.info(f"После препроцессинга: {X_prepared.shape[1] - 1} фич")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    X_prepared.to_parquet(output_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Подготовленные данные сохранены в {output_path}")
    logger.info(f"Препроцессор сохранён в {preprocessor_path}")
    logger.info("=== Подготовка завершена ===")

    return X_prepared, preprocessor


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    prepare_data(db_path="storage/raw_data.db")