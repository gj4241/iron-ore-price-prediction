# -*- coding: utf-8 -*-
"""
데이터 관련 util 모음입니다.
series_to_supervised() 는
'https://machinelearningmastery.com/
convert-time-series-supervised-learning-problem-python/'를 참고함.

Created on Tue Jul 17 14:54:19 2018

@author: kkalla
"""
import datetime
import pandas as pd
import numpy as np


def data_preprocess(df):
    def index_corr(df,week):
        index_1=df[df.Date<=(df.Date.max()- datetime.timedelta(days=(week*7)))]
        index_1 = index_1.drop(["Date","target"], axis=1)
        index_1=index_1.reset_index(drop=True)

        if week == 0:
            target=df['target'][:]
        else :
            target=df['target'][week:]
        target=target.reset_index(drop=True)
        total=pd.concat([target,index_1],axis=1)
        total=total.reset_index(drop=True)
        corr_total=total.corr()
        corr_target=corr_total['target']
        return corr_target
    corr_t=[]
    tn=[]
    for i in range(0,30):
        t = index_corr(df,i)
        corr_t.append(t)
        tn.append('t-'+str(i)+'주')
    corr_t = pd.DataFrame(corr_t)
    corr_t.index=np.arange(0,30)
    corr_t=corr_t.drop('target',1)
    corr_t['시점']=tn
    cols = corr_t.columns.tolist()
    cols = cols[-1:] + cols[:-1] # 마지막 열을 앞에 열로 보내기
    corr_t=corr_t[cols]

    def check_t(corr_t,value):
        qq=[]
        for i in range(0,(len(corr_t.iloc[:,1])-1)):
            if abs(corr_t.iloc[:,value][i]) == max(abs(corr_t.iloc[:,value])):
                qq = ('t-'+str(i)+'주')
        return qq

    check=[]
    for k in range(1,len(corr_t.columns)):
        a= check_t(corr_t,k)
        if a == []:
            a = '선행시점없음'
        check.append(a)
    pre_corr=pd.DataFrame(check,corr_t.columns[1:],columns=['피어슨선행시점'])#### []공백으로 나오는건 마지막시점이 가장 max 일 경우 즉 쓸데없음
    pre_corr

    pre_feature = []
    pre_feature.extend(pre_corr.index[pre_corr.피어슨선행시점=='t-1주'].tolist())
    pre_feature.extend(pre_corr.index[pre_corr.피어슨선행시점=='t-2주'].tolist())
    pre_feature.extend(pre_corr.index[pre_corr.피어슨선행시점=='t-3주'].tolist())
    pre_feature.extend(pre_corr.index[pre_corr.피어슨선행시점=='t-4주'].tolist())
    pre_feature.extend(pre_corr.index[pre_corr.피어슨선행시점=='t-5주'].tolist())

    t_1=pre_corr.index[pre_corr.피어슨선행시점=='t-1주'].tolist()
    t_2=pre_corr.index[pre_corr.피어슨선행시점=='t-2주'].tolist()
    t_3=pre_corr.index[pre_corr.피어슨선행시점=='t-3주'].tolist()
    t_4=pre_corr.index[pre_corr.피어슨선행시점=='t-4주'].tolist()
    t_5=pre_corr.index[pre_corr.피어슨선행시점=='t-5주'].tolist()

    feature_1=df[t_1][1:]
    feature_2=df[t_2][2:]
    feature_3=df[t_3][3:]
    feature_4=df[t_4][4:]
    feature_5=df[t_5][5:]

    feature_1=feature_1.reset_index(drop=True)
    feature_2=feature_2.reset_index(drop=True)
    feature_3=feature_3.reset_index(drop=True)
    feature_4=feature_4.reset_index(drop=True)
    feature_5=feature_5.reset_index(drop=True)

    df_table=pd.concat(
            [df[['target']],
             feature_1,feature_2,feature_3,feature_4,feature_5],
             axis=1)

    a = df.target[1:]
    a = a.reset_index(drop=True)
    df_table['target_5'] = a
    df_table.dropna(inplace=True)
    df_table.reset_index(drop=True,inplace=True)

    ma1 = df_table.rolling(window=5).mean().fillna(df_table.rolling(window=5).mean().mean())['Bloomberg(DJ-UBS) Commodity Index1991=100']
    ma2 = df_table.rolling(window=5).mean().fillna(df_table.rolling(window=5).mean().mean())['Bloomberg Commodity Total Return1991=100']
    ma3 = df_table.rolling(window=5).mean().fillna(df_table.rolling(window=5).mean().mean())['전기동 [LME] 현물USD/ton']
    ma4 = df_table.rolling(window=5).mean().fillna(df_table.rolling(window=5).mean().mean())['니켈 [LME] 현물USD/ton']
    ma5 = df_table.rolling(window=5).mean().fillna(df_table.rolling(window=5).mean().mean())['Rogers International Commodities Index Metals1998.07.31=1000']

    df_table['ma1']= ma1
    df_table['ma2']= ma2
    df_table['ma3']= ma3
    df_table['ma4']= ma4
    df_table['ma5']= ma5

    result = df_table

    return result

def get_train_test(df,timesteps,output_len,target_col='target',
                      test_len=40,test_pct=None,
                      x_scaler=None,y_scaler=None):
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

    if x_scaler is not None:
        train_set = x_scaler.fit_transform(train_set)
        test_set = x_scaler.transform(test_set)
    if y_scaler is not None:
        train_y = y_scaler.fit_transform(train_y)
        test_y = y_scaler.transform(test_y)

    train_sup,data_dim = series_to_supervised(train_set,timesteps,output_len)
    train_X = train_sup.iloc[:,:data_dim*timesteps]
    train_Y,_ = series_to_supervised(train_y,timesteps,output_len)
    train_Y = train_Y.iloc[:,timesteps:]

    test_sup,_ = series_to_supervised(test_set,timesteps,output_len)
    test_X = test_sup.iloc[:,:data_dim*timesteps]
    test_Y,_ = series_to_supervised(test_y,timesteps,output_len)
    test_Y = test_Y.iloc[:,timesteps:]

    return train_X,train_Y,test_X,test_Y,data_dim,x_scaler,y_scaler

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
