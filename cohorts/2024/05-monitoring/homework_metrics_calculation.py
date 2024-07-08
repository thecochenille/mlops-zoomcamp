import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric,ConflictPredictionMetric

from prefect import task, flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns int,
	share_missing_val float,
	num_unstable_predictions int,
	fare_quantile float)
"""
reference_data = pd.read_parquet('data/reference.parquet')
with open('models/lin_reg.bin','rb') as f_in:
	model=joblib.load(f_in)

raw_data= pd.read_parquet('data/green_tripdata_2024-03.parquet')

begin = datetime.datetime(2024,3,1,0,0)

num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features
)

report=Report(
    metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ConflictPredictionMetric(),
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)
        
    ]
)

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@flow
def calculate_metrics_postgresql(curr,i):
	current_data=raw_data[(raw_data.lpep_pickup_datetime <= (begin + datetime.timedelta(i))) &
		(raw_data.lpep_pickup_datetime <(begin + datetime.timedelta(i+1)))]

	current_data.fillna(0, inplace=True)
	current_data['prediction']=model.predict(current_data[num_features + cat_features])

	report.run(reference_data=reference_data, current_data=current_data,
			column_mapping=column_mapping)
	result=report.as_dict()

	prediction_drift=result['metrics'][0]['result']['drift_score']

	num_drifted_columns=result['metrics'][1]['result']['number_of_drifted_columns']

	share_missing_val=result['metrics'][2]['result']['current']['share_of_rows_with_missing_values']

	num_unstable_predictions= result['metrics'][3]['result']['current']['number_not_stable_prediction']

	fare_quantile=result['metrics'][4]['result']['current']['value']

	curr.execute(
		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_val, num_unstable_predictions, fare_quantile) values (%s, %s, %s, %s,%s,%s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_val, num_unstable_predictions, fare_quantile)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 31):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr,i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()