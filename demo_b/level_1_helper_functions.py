import itertools
import pandas as pd
from pyspark.sql.functions import col, when, size

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from ts.flint.summarizers import stddev

import demo_b.Configs as Configs


def clip_data(data_in=None):
    for column in data_in.columns:
        if column in list(Configs.CLIP_DICT.keys()):
            data_in[column] = data_in[column].clip(lower=Configs.CLIP_DICT[column][0],
                                                   upper=Configs.CLIP_DICT[column][1])
    return data_in

def clip(data_in, cols):
    for column in cols:
        if column in list(Configs.CLIP_DICT.keys()):
            data = data_in.withColumn(column, when(col(column) < Configs.CLIP_DICT[column][0], Configs.CLIP_DICT[column][0]))
            data_ = data.withColumn(column, when(col(column) > Configs.CLIP_DICT[column][1], Configs.CLIP_DICT[column][1]))
    return data_



def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def clean_data(data_in):
    data = data_in.fillna(method='ffill')
    data = data.fillna(method='bfill')
    return data


def get_split_prep_data(data, train_start, train_end,
                        column_name):
    # data = clean_data(vds_5)

    # minmax = MinMaxScaler(feature_range=(-1, 1))
    #
    # data[column_name] = minmax.fit_transform(data[column_name].values.reshape(-1, 1))
    #
    if column_name is not 'datetime':
        data = data.select(column_name).collect()
        result = []
        for index in range(train_start, train_end - Configs.LSTM_SEQUENCE_LENGTH):
            result.append(data[index: index + Configs.LSTM_SEQUENCE_LENGTH])
        result = np.array(result)  # shape (samples, sequence_length)
        result, result_mean = z_norm(result)

        train = result[train_start:train_end, :]
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train,


def prepare_data(data=None, column_name=None):
    x_train, y_train = get_split_prep_data(data, 0, data.count(), column_name)
    return x_train, y_train


def averageFunc(marksColumns):
    return (sum(x for x in marksColumns) / len(marksColumns))

def get_mean_of_cyl_values(dataframe):
    scavAirFireDetTemp_cols = [col(column) for column in dataframe.columns if column.startswith('scavAirFireDetTempN')]
    cylExhGasOutTemp_cols = [col(column) for column in dataframe.columns if column.startswith('cylExhGasOutTempN')]
    cylJCFWOutTemp_cols = [col(column) for column in dataframe.columns if column.startswith('cylJCFWOutTempNo')]
    cylPistonCOOutTemp_cols = [col(column) for column in dataframe.columns if column.startswith('cylPistonCOOutTempNo')]
    tcExhGasInTemp_cols = [col(column) for column in dataframe.columns if column.startswith('tcExhGasInTempN')]
    tcExhGasOutTemp_cols = [col(column) for column in dataframe.columns if column.startswith('tcExhGasOutTempN')]
    tcLOInLETPress_cols = [col(column) for column in dataframe.columns if column.startswith('tcLOInLETPressNo')]
    tcLOOutLETTemp_cols = [col(column) for column in dataframe.columns if column.startswith('tcLOOutLETTempNo')]
    coolingWOutLETTemp_cols = [col(column) for column in dataframe.columns if column.startswith('coolingWOutLETTemp')]
    tcRPM_cols = [col(column) for column in dataframe.columns if column.startswith('tcRPM')]

    print(scavAirFireDetTemp_cols)

    dataframe.withColumn('scavAirFireDetTempN', averageFunc(scavAirFireDetTemp_cols))
    dataframe.withColumn('cylExhGasOutTempN' ,averageFunc(cylExhGasOutTemp_cols))
    dataframe.withColumn('cylJCFWOutTempNo',averageFunc(cylJCFWOutTemp_cols))
    dataframe.withColumn('cylPistonCOOutTempNo', averageFunc(cylPistonCOOutTemp_cols))
    dataframe.withColumn('tcExhGasInTempN', averageFunc(tcExhGasInTemp_cols))
    dataframe.withColumn('tcExhGasOutTempN', averageFunc(tcExhGasOutTemp_cols))
    dataframe.withColumn('tcLOInLETPressNo',averageFunc(tcLOInLETPress_cols))
    dataframe.withColumn('tcLOOutLETTempNo',averageFunc(tcLOOutLETTemp_cols))
    dataframe.withColumn('coolingWOutLETTemp',averageFunc(coolingWOutLETTemp_cols))
    dataframe.withColumn('tcRPM',averageFunc(tcRPM_cols))

    # dataframe['cylJCFWOutTempNo'] = dataframe[cylJCFWOutTemp_cols].mean(axis=1)
    # dataframe['cylPistonCOOutTempNo'] = dataframe[cylPistonCOOutTemp_cols].mean(axis=1)
    # dataframe['tcExhGasInTempN'] = dataframe[tcExhGasInTemp_cols].mean(axis=1)
    # dataframe['tcExhGasOutTempN'] = dataframe[tcExhGasOutTemp_cols].mean(axis=1)
    # dataframe['tcLOInLETPressNo'] = dataframe[tcLOInLETPress_cols].mean(axis=1)
    # dataframe['tcLOOutLETTempNo'] = dataframe[tcLOOutLETTemp_cols].mean(axis=1)
    # dataframe['coolingWOutLETTemp'] = dataframe[coolingWOutLETTemp_cols].max(axis=1)
    # dataframe['tcRPM'] = dataframe[tcRPM_cols].mean(axis=1)
    #

    scavAirFireDetTemp_cols = [column for column in dataframe.columns if column.startswith('scavAirFireDetTempN')]
    cylExhGasOutTemp_cols = [column for column in dataframe.columns if column.startswith('cylExhGasOutTempN')]
    cylJCFWOutTemp_cols = [column for column in dataframe.columns if column.startswith('cylJCFWOutTempNo')]
    cylPistonCOOutTemp_cols = [column for column in dataframe.columns if column.startswith('cylPistonCOOutTempNo')]
    tcExhGasInTemp_cols = [column for column in dataframe.columns if column.startswith('tcExhGasInTempN')]
    tcExhGasOutTemp_cols = [column for column in dataframe.columns if column.startswith('tcExhGasOutTempN')]
    tcLOInLETPress_cols = [column for column in dataframe.columns if column.startswith('tcLOInLETPressNo')]
    tcLOOutLETTemp_cols = [column for column in dataframe.columns if column.startswith('tcLOOutLETTempNo')]
    coolingWOutLETTemp_cols = [column for column in dataframe.columns if column.startswith('coolingWOutLETTemp')]
    tcRPM_cols = [column for column in dataframe.columns if column.startswith('tcRPM')]

    dataframe.drop(*scavAirFireDetTemp_cols)
    dataframe.drop(*cylExhGasOutTemp_cols)
    dataframe.drop(*cylJCFWOutTemp_cols)
    dataframe.drop(*cylPistonCOOutTemp_cols)
    dataframe.drop(*tcExhGasInTemp_cols)
    dataframe.drop(*tcExhGasOutTemp_cols)
    dataframe.drop(*tcLOInLETPress_cols)
    dataframe.drop(*tcLOOutLETTemp_cols)
    dataframe.drop(*coolingWOutLETTemp_cols)
    dataframe.drop(*tcRPM_cols)
    return dataframe


# Function to normalise (standardise) PySpark dataframes
def standardize_train_test_data(train_df, columns):
    '''
    Add normalised columns to the input dataframe.
    formula = [(X - mean) / std_dev]
    Inputs : training dataframe, list of column name strings to be normalised
    Returns : dataframe with new normalised columns, averages and std deviation dataframes
    '''
    # Find the Mean and the Standard Deviation for each column
    aggExpr = []
    aggStd = []
    print(columns)
    for column in columns:
        print(column)
        aggExpr.append(np.mean(train_df[column]).alias(column))
        aggStd.append(stddev(train_df[column]).alias(column + '_stddev'))

    averages = train_df.agg(*aggExpr).collect()[0]
    std_devs = train_df.agg(*aggStd).collect()[0]

    # Standardise each dataframe, column by column
    for column in columns:
        # Standardise the TRAINING data
        train_df = train_df.withColumn(column + '_norm', ((train_df[column] - averages[column]) /
                                                          std_devs[column + '_stddev']))

        # Standardise the TEST data (using the training mean and std_dev)
        # test_df = test_df.withColumn(column + '_norm', ((test_df[column] - averages[column]) /
        #                                                 std_devs[column + '_stddev']))
    return train_df, averages, std_devs