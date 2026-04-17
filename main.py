from Data_Collection.data_collection import collect
from Data_Analysis.analyze_data import analyze_and_clean
from Data_Preparation.data_preparation import prepare
from Model_Training.model import train_model, load_model, finetune_model, evaluate_model, save_model
import numpy as np
from utils import load_config
import argparse
import json
from pathlib import Path
from datetime import datetime
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle insurance payment prediction, control script.")

    parser.add_argument("--mode", required=True, choices=["inference", "update", "summary"],
                        help="Modus operandi: inference/update/summary")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to configuration file.")
    parser.add_argument("--input", default="data/motor_data11-14lats.csv",
                        help="Path to .csv with inference/finetune data.")
    parser.add_argument("--model", default="model/model_dt.joblib",
                        help="Path to .joblib with serialized model.")
    parser.add_argument("--save_dir", default="results/",
                        help="Directory to save results in.")

    return parser.parse_args()


TARGET_COL = "CLAIM_PAID"


def _ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_data_pipeline(config: str, data_path: str, fit_preprocessor: bool):
    collect(config, override_csv=data_path)
    analyze_and_clean(config)
    X_prepared, _ = prepare(config, fit_preprocessor=fit_preprocessor)

    if TARGET_COL not in X_prepared.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing after preparation.")

    y = X_prepared[TARGET_COL].astype(float)
    X = X_prepared.drop(columns=[TARGET_COL])
    return X_prepared, X, y


def inference(config, model_path, data_path, save_dir):
    save_dir = _ensure_dir(save_dir)
    X_prepared, X, _ = _run_data_pipeline(config, data_path, fit_preprocessor=False)
    model = load_model(model_path)
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != int(X.shape[1]):
        raise ValueError(
            f"Feature mismatch: model expects {model.n_features_in_}, got {X.shape[1]}"
        )
    preds = model.predict(X)
    pred_df = X_prepared.copy()
    pred_df["predict"] = preds

    pred_path = save_dir / f"predictions_{Path(data_path).stem}_{_ts()}.csv"
    pred_df.to_csv(pred_path, index=False)
    return str(pred_path)


def update(config, model_path, data_path, save_dir):
    save_dir = _ensure_dir(save_dir)
    model_path = Path(model_path)
    try:
        _, X, y = _run_data_pipeline(config, data_path, fit_preprocessor=False)

        model = load_model(str(model_path))
        if hasattr(model, "n_features_in_") and int(model.n_features_in_) != int(X.shape[1]):
            raise ValueError(
                f"Feature mismatch: model expects {model.n_features_in_}, got {X.shape[1]}"
            )

        metrics_before = evaluate_model(model, X, X, y, y)
        model = finetune_model(model, X, y)
        metrics_after = evaluate_model(model, X, X, y, y)

        ts = _ts()
        finetuned_path = save_dir / f"{model_path.stem}_finetuned_{ts}.joblib"
        save_model(model, str(finetuned_path))
        save_model(model, str(model_path))

        report = {
            "timestamp": ts,
            "model_path": str(model_path),
            "finetuned_path": str(finetuned_path),
            "data_path": str(data_path),
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after
        }
        report_path = save_dir / f"update_report_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        err_path = save_dir / f"update_error_{_ts()}.log"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
        return False, None

    return True, str(report_path)


def summary(config, save_dir):
    cfg = load_config(config)
    save_dir = _ensure_dir(save_dir)
    logs_dir = Path(cfg.get("logging", {}).get("folder", "logs/"))

    analysis_rows = []
    for report_path in sorted(logs_dir.glob("anal_report_*/data_quality_report.json")):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        analysis_rows.append({
            "run": report_path.parent.name,
            "rows_before": report.get("rows_before"),
            "rows_after": report.get("rows_after"),
            "rows_removed": report.get("rows_removed"),
            "missing_full_after": (
                report.get("quality_after_cleaning", {})
                .get("completeness", {})
                .get("full")
            ),
        })

    training_rows = []
    for report_path in sorted(logs_dir.glob("train_report_*/train_report.json")):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        metrics = report.get("metrics", {})
        training_rows.append({
            "run_time": report.get("run_time"),
            "model": report.get("model"),
            "test_mae": metrics.get("test_mae"),
            "test_rmse": metrics.get("test_rmse"),
            "test_r2": metrics.get("test_r2"),
        })

    update_rows = []
    for report_path in sorted(Path(save_dir).glob("update_report_*.json")):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        update_rows.append({
            "timestamp": report.get("timestamp"),
            "model_path": report.get("model_path"),
            "finetuned_path": report.get("finetuned_path"),
            "before_test_mae": report.get("metrics_before", {}).get("test_mae"),
            "after_test_mae": report.get("metrics_after", {}).get("test_mae"),
        })

    best_train = None
    if training_rows:
        valid = [x for x in training_rows if x.get("test_mae") is not None]
        if valid:
            best_train = min(valid, key=lambda x: x["test_mae"])

    summary_payload = {
        "generated_at": _ts(),
        "sources": {
            "logs_dir": str(logs_dir),
            "results_dir": str(save_dir),
        },
        "counts": {
            "analysis_reports": len(analysis_rows),
            "training_reports": len(training_rows),
            "update_reports": len(update_rows),
        },
        "latest": {
            "analysis": analysis_rows[-1] if analysis_rows else None,
            "training": training_rows[-1] if training_rows else None,
            "update": update_rows[-1] if update_rows else None,
        },
        "best_training_by_test_mae": best_train,
        "history": {
            "analysis": analysis_rows,
            "training": training_rows,
            "updates": update_rows,
        }
    }

    summary_path = save_dir / f"summary_{_ts()}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    return str(summary_path)


def main():
    args = parse_args()

    config = args.config
    mode = args.mode
    data_path = args.input
    model_path = args.model
    save_dir = args.save_dir

    np.random.seed(load_config(config)['seed'])

    if mode == "inference":
        path = inference(config, model_path, data_path, save_dir)
        print(f"Inference successful: {path}")
    elif mode == "update":
        ok, report_path = update(config, model_path, data_path, save_dir)
        if ok:
            print(f"Update successful: {report_path}")
        else:
            print("Error while updating model")
    elif mode == "summary":
        path = summary(config, save_dir)
        print(f"Summary report saved: {path}")
    else:
        raise ValueError(f'Unrecognized execution mode: {mode}')


if __name__ == "__main__":
    main()
