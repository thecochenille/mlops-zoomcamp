#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle

import pandas as pd

import mlflow
import uuid

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline


def generate_uuids(n):
    ride_ids=[]
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['ride_id'] = generate_uuids(len(df))
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
   
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts



def load_model(RUN_ID):
    logged_model = f'runs:/{RUN_ID}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    print(f'readning the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)
    
    print(f'applying the model')
    y_pred = model.predict(dicts)

    print(f'saving the results to {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']#there is no uniqueID in the df, can create a userid by combining unique columns, or use uuid
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PUlocationID']= df['PULocationID']
    df_result['DO_locationID']=df['DOLocationID']
    df_result['actual_duration']=df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] =df_result['actual_duration']-df_result['predicted_duration']
    df_result['model_version'] =run_id

    df_result.to_parquet(output_file, index=False)


def run():
    taxi_type= sys.argv[1] #'green'
    year= int(sys.argv[2]) #2021
    month= int(sys.argv[3]) #2
    run_id = sys.argv[4] #'6b736fe6c66e495e88a45b58a64eee67'


    input_file =f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file=f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    MLFLOW_TRACKING_URI ='http://127.0.0.1:5000'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    #RUN_ID = os.getenv('RUN_ID')
    #logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'

    apply_model(
        input_file=input_file, 
        run_id = run_id, 
        output_file=output_file
    )

if __name__ == '__main__':
    run()