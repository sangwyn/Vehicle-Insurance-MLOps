import json
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np

def read_csv_to_pd(csv_path):
    df = pd.read_csv(csv_path)
    return df

def get_another_batch(df, batch_size = 100):
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end].copy()
        yield batch

def store_batch_to_bd(batch, db, table = "raw_data"):
    if batch is None or batch.empty:
        raise ValueError("Empty batch!!!")

    batch = batch.copy()
    batch["insertion_date"] = datetime.now().isoformat()

    with sqlite3.connect(db) as conn:
        batch.to_sql(
            name=table,
            con=conn,
            if_exists="append",
            index=False,
        )

    return len(batch)

def compute_batch_meta(batch, batch_id):
    numeric_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = batch.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_stats = {}
    for col in numeric_cols:
        s = batch[col]
        numeric_stats[col] = {
            "min": float(s.min()) if not s.empty else None,
            "max": float(s.max()) if not s.empty else None,
            "mean": float(s.mean()) if not s.empty else None,
            "std": float(s.std()) if not s.empty else None,
        }

    categorical_stats = {}
    for col in categorical_cols:
        if s.empty:
            categorical_stats[col] = {"n_unique": 0, "top": None, "top_freq": 0.0}
            continue
        top = s.mode().iloc[0]
        categorical_stats[col] = {
            "n_unique": int(s.nunique()),
            "top": str(top),
            "top_freq": float((s == top).mean()),
        }

    meta = {
        "batch_id": batch_id,
        "computed_at": datetime.utcnow().isoformat(),
        "n_rows": int(len(batch)),
        "n_cols": int(batch.shape[1]),
        "missing_per_col": {c: int(batch[c].isna().sum()) for c in batch.columns},
        "missing_total": int(batch.isna().sum().sum()),
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
    }
    return meta


def store_batch_meta(meta, db, table = "batch_meta"):
    row = {
        "batch_id": meta["batch_id"],
        "computed_at": meta["computed_at"],
        "n_rows": meta["n_rows"],
        "n_cols": meta["n_cols"],
        "missing_total": meta["missing_total"],
        "missing_per_col": json.dumps(meta["missing_per_col"]),
        "numeric_stats": json.dumps(meta["numeric_stats"]),
        "categorical_stats": json.dumps(meta["categorical_stats"]),
    }
    df_meta = pd.DataFrame([row])

    with sqlite3.connect(db) as conn:
        df_meta.to_sql(table, conn, if_exists="append", index=False)