from Data_Collection.data_collection import collect
from Data_Analysis.analyze_data import analyze_and_clean
from Data_Preparation.data_preparation import prepare
from sklearn.model_selection import train_test_split

def main():
    config = "config.yaml"

    collect(config)
    analyze_and_clean(config)
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

if __name__ == "__main__":
    main()