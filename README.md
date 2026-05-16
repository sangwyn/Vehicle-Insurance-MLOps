# Vehicle-Insurance-MLOps

MVP MLOps-конвейер для прогнозирования страховых выплат по автострахованию.

Реализованы этапы:
1. Сбор данных (batch-эмуляция потока, запись в SQLite).
2. Анализ и очистка данных (data quality, EDA, отчёты).
3. Подготовка данных (feature engineering + preprocessing + сохранение препроцессора).
4. Обучение/дообучение моделей (`lr`, `knn`, `dt`).
5. Валидация модели и реестр версий.
6. Сервисная часть (сериализация модели, артефакты в `model/`, `logs/`, `results/`).
7. Скрипт управления (`inference`, `update`, `summary`) через `main.py`.

## Документация для проверки

- Директория с документацией: [`doc/`](/home/oleg/coding/py/Vehicle-Insurance-MLOps/doc)
- План по баллам и статусам реализации: [`doc/grade`](/home/oleg/coding/py/Vehicle-Insurance-MLOps/doc/grade)
- Постановка задачи и описание технических решений: [`doc/task`](/home/oleg/coding/py/Vehicle-Insurance-MLOps/doc/task)

## Структура проекта

- `Data_Collection/` — этап 1.
- `Data_Analysis/` — этап 2.
- `Data_Preparation/` — этап 3.
- `Model_Training/` — этапы 4 + 6.
- `Model_Validation/` — этап 5.
- `main.py` — управление сценарием работы системы (этап 7).
- `full_pipeline.py` — полный запуск этапов 1→6.
- `configs/default.yaml` — конфигурация пайплайна.

## Требования

- Python 3.10+
- Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1) Полный запуск пайплайна (сбор → анализ → подготовка → обучение → валидация)

```bash
python full_pipeline.py --config configs/default.yaml
```

После запуска создаются:
- базы `raw_data.db`, `clean_data.db`;
- обученная модель (по умолчанию `model/model_dt.joblib`);
- отчёты в `logs/`.

### 2) Управление через `main.py` (этап 7)

Общий формат:

```bash
python main.py --mode <inference|update|summary> --config configs/default.yaml --input <path_to_csv> --model <path_to_model> --save_dir results
```

#### Inference

```bash
python main.py \
  --mode inference \
  --config configs/default.yaml \
  --input data/motor_data11-14lats.csv \
  --model model/model_dt.joblib \
  --save_dir results
```

Результат: CSV с колонкой `predict` в `results/`.

#### Update

```bash
python main.py \
  --mode update \
  --config configs/default.yaml \
  --input data/motor_data14-2018.csv \
  --model model/model_dt.joblib \
  --save_dir results
```

Результат:
- дообученная модель в `results/*_finetuned_*.joblib`;
- обновление основной модели по пути `--model`;
- отчёт `results/update_report_*.json` (метрики до/после).

#### Summary

```bash
python main.py --mode summary --config configs/default.yaml --save_dir results
```

Результат: агрегированный отчёт `results/summary_*.json`.

## Конфигурация

Основные параметры находятся в `configs/default.yaml`:
- `sources`, `stream`, `storage` — этап 1;
- `data_analisys` — этап 2;
- `data_preparation` — этап 3;
- `train_model` — этап 4;
- `model_validation` — этап 5;
- `logging` — путь и формат логов.
