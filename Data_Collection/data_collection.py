import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils import *

logger = logging.getLogger("collector")

def setup_logger(log_cfg):
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = log_cfg.get("folder", "logs/") + "collector.log"
    console = log_cfg.get("console", True)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(level)
    logger.handlers.clear()      # на случай повторного вызова (например в ноутбуке)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info(f"Логгер инициализирован (level={level_name}, file={log_file})")

def read_sources(sources_cfg):
    frames = {}
    for src in sources_cfg:
        name = src["name"]
        path = src["path"]
        try:
            frames[name] = read_csv_to_pd(path)
            logger.info(f"Источник '{name}' загружен: {len(frames[name])} строк из {path}")
        except FileNotFoundError:        logger.warning(f"Источник '{name}' не найден по пути {path} — пропускаем")
        except pd.errors.EmptyDataError: logger.warning(f"Источник '{name}' пустой ({path}) — пропускаем")
        except Exception:                logger.exception(f"Не удалось прочитать источник '{name}' ({path})")

    if not frames:
        logger.error("Ни один источник не удалось прочитать — прерываем сбор")
        raise RuntimeError("No sources loaded")
    return frames

def get_another_batch(df, batch_size = 100):
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end].copy()
        yield batch

def get_another_batch_multi(frames, batch_size=100, mix_sources=True):
    iterators = {name: get_another_batch(df, batch_size) for name, df in frames.items()}

    if not mix_sources:
        for name, it in iterators.items():
            for batch in it:
                yield name, batch
        return

    active = dict(iterators)
    while active:
        for name in list(active.keys()):
            try:
                batch = next(active[name])
                yield name, batch
            except StopIteration:
                del active[name]
                logger.info(f"Источник '{name}' исчерпан")

def store_batch_to_bd(batch, db, table="raw_data", source_name="unknown"):
    if batch is None or batch.empty:
        raise ValueError("Empty batch!!!")
    batch = batch.copy()
    batch["insertion_date"] = datetime.now().isoformat()
    batch["source_name"] = source_name
    with sqlite3.connect(db) as conn:
        batch.to_sql(
            name=table,
            con=conn,
            if_exists="append",
            index=False,
        )
    return len(batch)

def compute_batch_meta(batch, batch_id, source_name="unknown"):
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
        "source_name": source_name,
        "computed_at": datetime.now().isoformat(),
        "n_rows": int(len(batch)),
        "n_cols": int(batch.shape[1]),
        "missing_per_col": {c: int(batch[c].isna().sum()) for c in batch.columns},
        "missing_total": int(batch.isna().sum().sum()),
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
    }
    return meta

def store_batch_meta(meta, db, table="batch_meta"):
    row = {
        "batch_id": meta["batch_id"],
        "source_name": meta["source_name"],
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

def collect(config):
    cfg = load_config(config)
    setup_logger(cfg.get("logging", {}))

    logger.info("=== Начало сбора данных ===")

    try:
        sources_cfg = cfg["sources"]
        batch_sz    = cfg["stream"]["batch_size"]
        sleep_s     = cfg["stream"]["sleep_seconds"]
        max_batch   = cfg["stream"]["max_batches"]
        mix_sources = cfg["stream"].get("mix_sources", True)

        db_path    = cfg["storage"]["db_path"]
        raw_table  = cfg["storage"]["raw_table"]
        meta_table = cfg["storage"]["meta_table"]

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        frames = read_sources(sources_cfg)
        stream = get_another_batch_multi(frames, batch_size=batch_sz, mix_sources=mix_sources)

        processed = 0
        failed = 0

        for i, (source_name, batch) in enumerate(stream):
            if max_batch is not None and i >= max_batch:
                logger.info(f"Достигнут лимит max_batches={max_batch}, останавливаемся")
                break

            try:
                store_batch_to_bd(batch, db_path, table=raw_table, source_name=source_name)
                meta = compute_batch_meta(batch, batch_id=i, source_name=source_name)
                store_batch_meta(meta, db_path, table=meta_table)

                logger.info(
                    f"Батч #{i} [{source_name}]: {meta['n_rows']} строк, "
                    f"пропусков: {meta['missing_total']}"
                )
                processed += 1

            except Exception:
                # логируем с traceback, но не падаем — идём дальше
                logger.exception(f"Ошибка при обработке батча #{i} из '{source_name}'")
                failed += 1

            time.sleep(sleep_s)

        logger.info(f"=== Сбор завершён: обработано {processed}, с ошибкой {failed} ===")

    except Exception:
        logger.exception("Фатальная ошибка в collect_data")
        raise
