#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:12:22 2018

@author: choigirhyo
"""
#============================================================================================================================================================
#변수로드
#============================================================================================================================================================
import argparse
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
data = pd.read_csv("data/dataset1.csv")
data = data.drop('Unnamed: 0',axis = 1)

parser = argparse.ArgumentParser()
parser.add_argument("--mode","-m",choices=['train','test'],
        default='train')

args = parser.parse_args()

#============================================================================================================================================================
#변수정제
#============================================================================================================================================================
Xtrain = data.drop('target',axis = 1)
Ytrain = data['target']

#맨뒤에 20%를 test-set으로(랜덤x)
def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_test, X_train = np.split(X, [i])
    y_test, y_train = np.split(y, [i])
    return X_train, X_test, y_train, y_test

X_test, X_train, y_test, y_train = non_shuffling_train_test_split(Xtrain,Ytrain)

#============================================================================================================================================================
#scale
#============================================================================================================================================================
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
X_train = scaler.fit_transform(X_train)
#X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
Y_train = scaler.fit_transform(DataFrame(y_train).values)

X_test = scaler.fit_transform(X_test)
Y_test = scaler.fit_transform(DataFrame(y_test).values)


#============================================================================================================================================================
#시점 이동
#============================================================================================================================================================
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)ñ
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
  if dropnan:
      agg.dropna(inplace=True)
  return agg

#12주차를 바탕으로 미래 예측
t = 12
n = 2 
X_train = series_to_supervised(X_train,t,n)
Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_train = series_to_supervised(Y_train,t,n)
Y_train = Y_train['var1(t+1)']

X_test = series_to_supervised(X_test,t,n)
Y_test = Y_test.reshape(Y_test.shape[0],1)
Y_test = series_to_supervised(Y_test,t,n)
Y_test = Y_test['var1(t+1)']

#3차원으로 변경
X_train = X_train.values.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.values.reshape(X_test.shape[0],X_test.shape[1],1)


#============================================================================================================================================================
#모델 빌드
#============================================================================================================================================================
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.utils import plot_model


model = Sequential()
model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
    input_dim = 50,
    output_dim = 100,
    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(
    output_dim=1))
model.add(Activation(PReLU()))############sigmoid,tanh,relu,Leakyrelu,LeakyRelu(alpha = 0.001),PReLU()
optimizer = Adam(lr=0.0001,decay=1e-6)
model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])

#============================================================================================================================================================
#fit에 사용되는 valid값 저장
#============================================================================================================================================================
def valid_increasing(model,X_train,Y_train,epochs,val_acc):
    epochs = epochs
    for i in range(epochs):
        val_split = i/100 +0.001
        history = model.fit(X_train,
                            Y_train,
                            epochs = 1,
                            batch_size = 64,
                            validation_split = val_split)
        if history.history['val_acc'][0] > val_acc:
            model_valid_X = history.validation_data[0]
            model_valid_Y = history.validation_data[1]
            pred = model.predict(model_valid_X)
    return model_valid_X,model_valid_Y,pred,model
if args.mode == 'train':
    model_valid_X,model_valid_Y,pred,model = valid_increasing(model,
            X_train,Y_train,100,0.0001)
    model.save("log/stock_price_pred_model.h5")
    plot_model(model,to_file='log/stock_price_pred_model.png')
#============================================================================================================================================================
#valid값을 바탕으로 linear_model 학습
#============================================================================================================================================================
def Linear_with_valid(model_valid_X,model_valid_Y,pred):
    from sklearn.linear_model import LinearRegression
    model_linear = LinearRegression()
    valid_X = DataFrame()
    valid_Y = DataFrame()
    valid_X = pd.concat([valid_X,
                         DataFrame(list(
                                 map(float,pred)))])
    valid_Y = pd.concat([valid_Y,
                         DataFrame(list(
                                 map(float,model_valid_Y)))])
    valid_X.index = range(0,len(valid_X))
    valid_Y.index = range(0,len(valid_Y))
    model_linear.fit(valid_X,valid_Y)    
    return model_linear

from sklearn.externals import joblib

if args.mode=='train':
    model_linear = Linear_with_valid(model_valid_X,model_valid_Y,pred)
    joblib.dump(model_linear,'log/model_linear.pkl')

if os.path.exists('log/stock_price_pred_model.h5'):
    model = load_model('log/stock_price_pred_model.h5')
else:
    assert "There is not a model saved. Train first"

X_test = model.predict(X_test)

#============================================================================================================================================================
#최종결과산출
#============================================================================================================================================================
def final_result(model_linear,X_test):
    result = model_linear.predict(X_test)
    return result
if args.mode == 'test':
    if os.path.exists("log/model_linear.pkl"):
        model_linear = joblib.load('log/model_linear.pkl')
    else:
        assert "There is not a linear model. Train first"
    result = final_result(model_linear,X_test)
    result = scaler.inverse_transform(DataFrame(result).values.reshape(-1,1))
    real = scaler.inverse_transform(DataFrame(Y_test).values.reshape(-1,1))



#============================================================================================================================================================
#결과확인 (rmse plot)
#============================================================================================================================================================
    from sklearn.metrics import mean_squared_error
    import math
    rmse = math.sqrt(mean_squared_error(result,real))

    import matplotlib.pyplot as plt
    plt.title('RMSE = {:3f}'.format(rmse))
    plt.plot(real,label='real')
    plt.plot(result,label='pred')
    plt.legend()
