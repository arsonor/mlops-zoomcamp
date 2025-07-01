import pandas as pd
from datetime import datetime
from batch_q3 import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),         # valid
        (1, 1, dt(1, 2), dt(1, 10)),               # valid
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),      # too short (<1 min)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),          # too long (>60 min)
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    df_processed = prepare_data(df, categorical)

    expected_data = [
        {'PULocationID': '-1', 'DOLocationID': '-1'},  # originally None
        {'PULocationID': '1', 'DOLocationID': '1'},
    ]

    actual_data = df_processed[categorical].to_dict(orient='records')

    assert actual_data == expected_data
