import time
from data_collection import *

df = read_csv_to_pd("../data/motor_data11-14lats.csv")
db_path = "raw_data.db"


with sqlite3.connect("raw_data.db") as conn:
    print(pd.read_sql("SELECT batch_id, n_rows, missing_total FROM batch_meta", conn))