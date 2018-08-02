# -*- coding: utf-8 -*-
"""
Test data utils

Created on Tue Jul 17 21:58:57 2018

@author: kkalla
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from data_utils import *

def get_test():
    x_sc = MinMaxScaler()
    y_sc = MinMaxScaler()
    print("reading dataset")
    df = pd.read_csv('../data/train_data_weekly.csv',parse_dates=['Date'])
    df.sort_values(by='Date',inplace=True)
    df = data_preprocess(df)
    train_X,train_Y,test_X,test_Y,data_dim,x_sc,y_sc = get_train_test(df,
                                        timesteps = 5,
                                        output_len = 1,
                                        target_col='target',
                                        x_scaler=x_sc,
                                        y_scaler=y_sc,
                                        test_len=40)
    print(test_Y)

def test_series_to_supervised():
    print('test series_to_supervised()...')
    x = [0,1,2,3,4,5,6]
    print(x,'to supervised(1,1)')
    print(series_to_supervised(x,1,1))
    x = np.random.randn(20,2)
    print(x,'to supervised(4,2)')
    print(series_to_supervised(x,4,2))
    df = pd.read_csv('../data/raw_data_merged.csv')
    df = df.iloc[:1000,:]
    result,_=series_to_supervised(df,20,20)
    print(result.columns)

if __name__ == "__main__":
    get_test()
    test_series_to_supervised()