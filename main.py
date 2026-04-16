from Data_Collection.data_collection import collect
from Data_Analysis.analyze_data import analyze_and_clean
from Data_Preparation.data_preparation import prepare
from Model_Training.model import train_model
from sklearn.model_selection import train_test_split
from Model_Validation.model_validation import validate_tree_model
import numpy as np
from utils import load_config


def main():
    config = "config.yaml"

    np.random.seed(load_config(config)['seed'])

    # Part 1: collect data
    collect(config)

    # Part 2: analyze data
    analyze_and_clean(config)

    # Part 3: prepare data
    X_prepared, _ = prepare(config)
    TARGET_COL = "CLAIM_PAID"

    y = X_prepared[TARGET_COL]
    X = X_prepared.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 20% в тест
        random_state=42,      # фиксируем для воспроизводимости
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")  

    # Part 4: train model
    model, losses = train_model(config, X_train, X_test, y_train, y_test)

    # Part 5: Validate model
    report = validate_tree_model(
        config_path=config,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(report)


if __name__ == "__main__":
    main()