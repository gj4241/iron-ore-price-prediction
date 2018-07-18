# -*- coding: utf-8 -*-
"""
데이터 관련 util 모음입니다.
series_to_supervised() 는 
'https://machinelearningmastery.com/
convert-time-series-supervised-learning-problem-python/'를 참고함.

Created on Tue Jul 17 14:54:19 2018

@author: kkalla
"""
import os

import pandas as pd


def train_test_split(data_dir):
    """
    'ironOre.csv' 데이터를 2018년 5월 이전과 이후로 나눠 저장합니다.
    """
    
    data_path = os.path.join(data_dir,'ironOre.csv')
    df = pd.read_csv(data_path)
    df[['날짜']] = pd.to_datetime(df['날짜'],format='%Y%m%d')
    df.set_index('날짜',inplace=True)
    before_may = df.loc[df.index < pd.datetime(2018,5,1),:]
    after_may = df.loc[df.index >= pd.datetime(2018,5,1),:]
    before_may.to_csv(os.path.join(data_dir,'train.csv'))
    after_may.to_csv(os.path.join(data_dir,'test.csv'))
    
def series_to_supervised(data,n_in=1,n_out=1,dropna=True):
    """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropna:
        agg.dropna(inplace=True)
    return agg
