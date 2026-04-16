'''

Анализ данных. Функционал: оценка качества данных (Data Quality), авто-
матический EDA, контроль выполнения критериев качества, очистка данных.

python3 analyze_data.py --input data/motor_data11-14lats.csv --output_dir reports

'''

import json
import numpy as np
import argparse
import os
from typing import Dict, Any, Tuple
from scipy.stats import kstest, ks_2samp
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from utils import *

def analyze_and_clean(config):
    """
    Полный второй этап: загрузка из raw_db, анализ, очистка,
    сохранение в clean_db + отчёт в output_dir.
    """
    analisys_params = load_config(config)["data_analisys"]
    raw_db         = analisys_params["raw_db"]
    raw_table      = analisys_params["raw_table"]
    clean_db       = analisys_params["clean_db"]
    clean_table    = analisys_params["clean_table"]
    logs           = analisys_params["logs"]
    prev_data_path = analisys_params["prev_data_path"]
    

    df = load_data_from_db(raw_db, raw_table)
    print(df)
    df_clean, report = analyze(df)

    ts_dir = Path(logs) / f"anal_report_{datetime.now().isoformat().replace(':', '-')}"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # clean → SQLite
    save_data_to_db(df_clean, clean_db, table=clean_table)

    # EDA + отчёт
    eda_plot_path = str(ts_dir / "eda_top_categories.png")
    report["eda"] = DataQualityEvaluator.EDA(df_clean, eda_plot_path)

    # data drift, если есть предыдущий датасет
    if prev_data_path:
        prev_df = load_data(prev_data_path)
        drift_results = monitor_drift(prev_df, df, str(ts_dir))
        with open(ts_dir / "drift_report.json", "w", encoding="utf-8") as f:
            json.dump(drift_results, f, ensure_ascii=False, indent=2, default=_json_default)

    report_path = ts_dir / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"Очистка: {report['rows_before']} → {report['rows_after']} строк")
    print(f"  clean data → {clean_db}")
    print(f"  отчёт      → {report_path}")

class DataQualityEvaluator:
    @staticmethod
    def completeness(df) -> Dict[str, float]:
        df_bin = df.isna()
        cols_nan_ratio = df_bin.sum(axis=0) / df.shape[0]
        rows_nan_ratio = df_bin.sum(axis=1) / df.shape[1]
        d = {
            "full": df_bin.sum().sum() / np.prod(df.shape),
            "cols_max": cols_nan_ratio.max(),
            "rows_max": rows_nan_ratio.max()
        }
        return d
    
    @staticmethod
    def validity(df) -> Dict[str, float]:
        d = {}
        for p in ["PREMIUM", "INSURED_VALUE", "CLAIM_PAID", "SEATS_NUM"]:
            if p in df.columns:
                d[p] = not ((df[p] < 0).any())
        return d
    
    @staticmethod
    def timeliness(df) -> Dict[str, float]:
        if "INSR_BEGIN" not in df.columns:
            return {"delta_time_max": None}

        inrs_begin = df["INSR_BEGIN"]
        if not np.issubdtype(inrs_begin.dtype, np.datetime64):
            inrs_begin = pd.to_datetime(inrs_begin, errors="coerce", format="%d-%b-%y")

        unique_dates = np.sort(inrs_begin.dropna().unique())
        if len(unique_dates) < 2:
            return {"delta_time_max": 0}

        max_time_lap = np.diff(unique_dates).max()
        max_time_lap = max_time_lap.astype("timedelta64[D]").astype(int)
        d = {"delta_time_max": max_time_lap}
        return d
    
    @staticmethod
    def EDA(df, output_path: str) -> Dict[str, Any]:
        if "TYPE_VEHICLE" not in df.columns or "MAKE" not in df.columns:
            return {"warning": "TYPE_VEHICLE or MAKE columns are not available for EDA plot."}

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(13, 3))
        type_count = df["TYPE_VEHICLE"].value_counts()
        type_count.sort_values().tail(20).plot(kind="bar", ax=axes[0], title="Top TYPE_VEHICLE")
        
        make_count = df["MAKE"].value_counts()
        make_count.sort_values().tail(20).plot(kind="bar", ax=axes[1], title="Top MAKE")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)

        return {
            "top_vehicle_types": type_count.head(10).to_dict(),
            "top_makes": make_count.head(10).to_dict()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check data quality, auto EDA")

    parser.add_argument("--input", type=str, required=True, help="Path to .csv file or dir with .csv files containing data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to dir where reports will be saved.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument("--prev_data", type=str, help="Previous dataset. Required to monitor data drift")

    return parser.parse_args()


def kolmogorov_smirnov(prev_df:pd.DataFrame, df: pd.DataFrame, col: str, output_path: str = None) -> Tuple[float, float, float]:
    sample1, sample2 = prev_df[col].tolist(), df[col].tolist()
    ks_statistic, p_value = ks_2samp(sample1, sample2, alternative='two-sided', mode='auto')

    alpha = 0.05
    drift_detected = p_value < alpha

    if output_path is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(sample1, bins=25, density=True, alpha=0.6, label="Old data")
        plt.hist(sample2, bins=25, density=True, alpha=0.6, label="Current data")
        plt.title("Two-Sample Kolmogorov-Smirnov Test")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path + f'/kstest_{col}.png', dpi=120)
        plt.close()

    return ks_statistic, p_value, drift_detected


def monitor_drift(prev_df: pd.DataFrame, df: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
    test_cols = ['INSURED_VALUE', 'PREMIUM', 'CLAIM_PAID', 'USAGE', 'TYPE_VEHICLE']
    drift_results = {}
    for col in test_cols:
        ks_statistic, p_value, drift_detected = kolmogorov_smirnov(prev_df, df, col, output_path)
        drift_results[col] = {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }

    return drift_results

 
def feature_engineering(df: pd.DataFrame):
    # Insurance duration (days)
    df['insr_duration'] = (df['INSR_END'] - df['INSR_BEGIN']).dt.days
    # Insurance season
    df['insr_begin_season'] = pd.cut(
        (df['INSR_BEGIN'].dt.month % 12) + 1, 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Autumn']
    )
    df['insr_end_season'] = pd.cut(
        (df['INSR_END'].dt.month % 12) + 1, 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Autumn']
    )
    # 1000 * Premium / Insured value ratio
    df['prem_insr_ratio'] = 1000 * df['PREMIUM'] / df['INSURED_VALUE']

    return df


def analyze(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    old_features = set(df.columns.tolist())
    df = feature_engineering(df)
    new_features = set(df.columns.tolist()) - old_features
    new_features = list(new_features)
    
    quality_before = {
        "completeness": DataQualityEvaluator.completeness(df),
        "validity": DataQualityEvaluator.validity(df),
        "timeliness": DataQualityEvaluator.timeliness(df)
    }

    thresholds = {
        "completeness_full_max": 0.10,
        "completeness_cols_max": 0.95,
        "completeness_rows_max": 0.65,
        "timeliness_delta_time_max_days": 31
    }

    checks = {
        "completeness_full_ok": bool(quality_before["completeness"]["full"] <= thresholds["completeness_full_max"]),
        "completeness_cols_ok": bool(quality_before["completeness"]["cols_max"] <= thresholds["completeness_cols_max"]),
        "completeness_rows_ok": bool(quality_before["completeness"]["rows_max"] <= thresholds["completeness_rows_max"]),
        "validity_ok": all(quality_before["validity"].values()) if quality_before["validity"] else True,
        "timeliness_ok": bool(
            quality_before["timeliness"]["delta_time_max"] is not None and
            quality_before["timeliness"]["delta_time_max"] <= thresholds["timeliness_delta_time_max_days"]
        )
    }
    checks["all_ok"] = all(checks.values())

    df_clean = df.copy()
    rows_before = len(df_clean)

    # 1. Remove duplicates.
    df_clean = df_clean.drop_duplicates()

    # 2. Remove rows with invalid negative values for selected numeric fields.
    for col in ["PREMIUM", "INSURED_VALUE", "CLAIM_PAID", "SEATS_NUM"]:
        if col in df_clean.columns:
            df_clean = df_clean[(df_clean[col].isna()) | (df_clean[col] >= 0)]

    # 3. Basic NA handling.
    if "INSR_BEGIN" in df_clean.columns:
        df_clean = df_clean[df_clean["INSR_BEGIN"].notna()]

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "CLAIM_PAID":
            df_clean[col] = df_clean[col].fillna(0.0)
        elif df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in non_numeric_cols:
        if df_clean[col].isna().any():
            mode = df_clean[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
            df_clean[col] = df_clean[col].fillna(fill_value)

    if "INSR_BEGIN" in df_clean.columns:
        df_clean = df_clean.sort_values("INSR_BEGIN").reset_index(drop=True)

    quality_after = {
        "completeness": DataQualityEvaluator.completeness(df_clean),
        "validity": DataQualityEvaluator.validity(df_clean),
        "timeliness": DataQualityEvaluator.timeliness(df_clean)
    }

    report = {
        "thresholds": thresholds,
        "new_features": new_features,
        "checks_before_cleaning": checks,
        "quality_before_cleaning": quality_before,
        "quality_after_cleaning": quality_after,
        "rows_before": rows_before,
        "rows_after": len(df_clean),
        "rows_removed": rows_before - len(df_clean)
    }
    return df_clean, report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    df = load_data(args.input)
    df_clean, report = analyze(df)

    output_dir = Path(args.output_dir + f'/anal_report_{str(datetime.now().isoformat())}')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prev_data is not None:
        prev_df = load_data(args.prev_data)
        drift_results = monitor_drift(prev_df, df, str(output_dir))
        drift_report_path = output_dir / "drift_report.json"
        with open(drift_report_path, "w", encoding="utf-8") as f:
            json.dump(drift_results, f, ensure_ascii=False, indent=2, default=_json_default)

    clean_data_path = save_data(df_clean, output_dir)
    eda_plot_path = str(output_dir / "eda_top_categories.png")
    report["eda"] = DataQualityEvaluator.EDA(df_clean, eda_plot_path)

    report_path = output_dir / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    print(json.dumps({
        "clean_data_path": clean_data_path,
        "report_path": str(report_path),
        "eda_plot_path": eda_plot_path
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
