import yaml
from pathlib import Path
import pandas as pd
import sqlite3

def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No such config: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    return cfg

def read_csv_to_pd(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_data_from_db(db_path: str, table: str = "raw_data") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    for col in ["INSR_BEGIN", "INSR_END"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%d-%b-%y")
    return df

def save_data_to_db(df: pd.DataFrame, db_path: str, table: str = "clean_data") -> None:
    df_to_save = df.copy()

    dt_cols = df_to_save.select_dtypes(include=["datetime64[ns]"]).columns
    for col in dt_cols:
        df_to_save[col] = df_to_save[col].dt.strftime("%Y-%m-%d")

    cat_cols = df_to_save.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df_to_save[col] = df_to_save[col].astype(str)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df_to_save.to_sql(table, conn, if_exists="replace", index=False)


def load_data(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.is_file():
        source_file = input_path
    elif input_path.is_dir():
        candidates = []
        for ext in ("*.csv", "*.xls", "*.xlsx"):
            candidates.extend(sorted(input_path.glob(ext)))
        if not candidates:
            raise FileNotFoundError(f"No tabular files found in directory: {input_path}")

        preferred = [p for p in candidates if p.name == "motor_data11-14lats.csv"]
        source_file = preferred[0] if preferred else candidates[0]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    df = read_csv_to_pd(source_file)
    for col in ["INSR_BEGIN", "INSR_END"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%d-%b-%y")
    return df


def save_data(df: pd.DataFrame, path: str) -> str:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "clean_data.csv"

    df_to_save = df.copy()
    datetime_cols = df_to_save.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        df_to_save[col] = df_to_save[col].dt.strftime("%Y-%m-%d")

    df_to_save.to_csv(save_path, index=False)
    return str(save_path)