#!/usr/bin/env python3
"""
Preprocessing Data for Forcasting
"""
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def preprocess_data(data):
    # mpl.rcParams['figure.figsize'] = (8, 6)
    # mpl.rcParams['axes.grid'] = False
    # zip.path = tf.keras.utils.get_file(
    #     origin="coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip",
    #     fname="coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",
    #     extract=True)
    # name = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

    # csv_path,  _ = os.path.splitext(name)
    dfData = pd.read_csv(data)
    df = dfData.dropna()

    # set time: reset index and replace with update time
    df.reset_index(inplace=True, drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', origin='unix')

    # set time: reset the year, next in sequence
    df['year'] = pd.DatetimeIndex(df["Timestamp"]).year
    df_years = df[df['year'] >= 2016]
    df_years.reset_index(inplace=True, drop=True)

    # set time: What is included in the data.  What is seen.
    # convert data frame index to a dattime index
    df_config = df_years.set_index(pd.DatetimeIndex(df_years["Timestamp"]))
    df_config.drop("Timestamp", axis=1, inplace=True)  # axis=0 row, axis=1 col
    df_config.drop("year", axis=1, inplace=True)  # drop & inplace
    df_config.drop("Open", axis=1, inplace=True)
    df_config.drop("Close", axis=1, inplace=True)
    df_config.drop("High", axis=1, inplace=True)
    df_config.drop("Low", axis=1, inplace=True)

    # resample data at frequency of an hour
    df_config["Weighted_Price"].resample('H').mean
    df_config["Volume_(Currency)"].resample('H').sum
    df_config["Volume_(BTC)"].resample('H').sum
    df_config["High"].resample('H').max
    df_config["Low"].resample('H').min

    # features: what to consider when forecasting
    features_considered = ["Volume_(BTC)", "Volume_(Currency)",
                           "Weighted_Price"]
    features = df_config[features_considered]

    # Dataset: standardize using standard deviation and mean
    dataset = features.values
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    dataset = (dataset-data_mean)/data_std

    return dataset
