#!/usr/bin/env python
# coding: utf-8


import sys
import pickle
import pandas as pd
import numpy as np
import datetime


def apply_model(input_file, categorical, output_file, year, month, dv, model):
    print(f' loading the data from {input_file}')
    df = read_data(input_file, categorical)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    print(f'predicting values')
    y_pred = model.predict(X_val)

    print(f'the standard deviation of the our predictions is {np.std(y_pred)}')
    print(f' the mean predicted duration is {np.mean(y_pred)}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predictions'] = y_pred
    df_result = df[['ride_id','predictions']] 

    print(f'saving results to {output_file}')
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )


def read_data(filename:str, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def run():

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    year = int(sys.argv[1]) #2023
    month = int(sys.argv[2]) #3
    taxi_type = 'yellow'

    input_file=f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    categorical = ['PULocationID', 'DOLocationID']

    apply_model(input_file=input_file, categorical=categorical, output_file=output_file, year=year, month=month, dv=dv, model = model)


if __name__ == '__main__':
    run()
