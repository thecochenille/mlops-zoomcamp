from datetime import datetime
import pandas as pd
import batch



def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data(data):
    data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        ('-1', '-1', dt(2023, 1, 1, 0, 0), dt(2023, 1, 1, 0, 10), 10.0),  # Duration = 10 mins
        ('1', '1', dt(2023, 1, 1, 0, 2), dt(2023, 1, 1, 0, 10), 8.0),    # Duration = 8 mins
        ('1', '-1', dt(2023, 1, 1, 0, 2, 0), dt(2023, 1, 1, 0, 2, 59), 0.9833333333333333)  # Duration = 0.9833 mins
    ]

    expected_df = pd.DataFrame(expected_data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = batch.prepare_data(df, categorical)

    assert expected_df == actual_df