# -*- coding: utf-8 -*-
"""
데이터 관련 util 모음입니다.
series_to_supervised() 는
'https://machinelearningmastery.com/
convert-time-series-supervised-learning-problem-python/'를 참고함.

Created on Tue Jul 17 14:54:19 2018

@author: kkalla
"""

import pandas as pd

def get_train_test(df,timesteps,output_len,target_col='target',
                      test_len=40,test_pct=None,
                      scaler=None):
    if test_len == None:
        assert test_pct==None, "test_len or test_pct must be specified"
        test_len = int(df.shape[0]*(1-test_pct))
    else:
        train_len = int(df.shape[0]-test_len)

    train_set = df.iloc[:train_len,:]
    test_set = df.iloc[train_len:,:]

    if type(target_col) == str:
        train_y = train_set[[target_col]]
        test_y = test_set[[target_col]]
    elif type(target_col) == int:
        train_y = train_set.iloc[:,target_col].values.reshape((-1,1))
        test_y = test_set.iloc[:,target_col].values.reshape((-1,1))
    else:
        assert "Check target_col"

    if scaler is not None:
        train_set = scaler.fit_transform(train_set)
        #train_y = scaler.fit_transform(train_y)
        #test_y = scaler.transform(test_y)

    train_sup,data_dim = series_to_supervised(train_set,timesteps,output_len)
    train_X = train_sup.iloc[:,:data_dim*timesteps]
    train_Y,_ = series_to_supervised(train_y,timesteps,output_len)
    train_Y = train_Y.iloc[:,timesteps:]

    test_sup,_ = series_to_supervised(test_set,timesteps,output_len)
    test_X = test_sup.iloc[:,:data_dim*timesteps]
    test_Y,_ = series_to_supervised(test_y,timesteps,output_len)
    test_Y = test_Y.iloc[:,timesteps:]

    return train_X,train_Y,test_X,test_Y,data_dim,scaler

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
    return agg, n_vars
