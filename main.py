from Data_Collection.data_collection import collect
from Data_Analysis.analyze_data import analyze_and_clean
from Data_Preparation.data_preparation import prepare

def main():
    config = "config.yaml"

    collect(config)
    analyze_and_clean(config)
    prepare(config)

if __name__ == "__main__":
    main()