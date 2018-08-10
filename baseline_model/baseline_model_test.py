# -*- coding: utf-8 -*-
"""
Evaludate baseline model

Created on Wed Jul 18 13:29:58 2018

@author: kkalla
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import train_test_split,series_to_supervised
from sklearn.metrics import mean_squared_error
from keras.models import load_model


data_dir = 'data'
log_dir = 'log'

def test():
    # load model
    model = load_model(os.path.join(log_dir,'baseline_model.h5'))
    
    # load testset
    if not os.path.exists(os.path.join(data_dir,'test.csv')):
        print('splitting train and test sets')
        train_test_split(data_dir)
            
    test_set_series = pd.read_csv(
            os.path.join(data_dir,'test.csv'),index_col=0)
    test_set = series_to_supervised(test_set_series,n_in=20,n_out=5)
    test_X = test_set.iloc[:,range(20)].values.reshape(-1,20,1)
    
    #print model overview
    print(model.summary())
    
    #predict & print
    yhat = model.predict(test_X)
    print('prediction: ')
    print(yhat)
    
    #flatten
    tmp = np.zeros((15,2))
    for i in range(yhat.shape[0]):
        for j in range(yhat.shape[1]):
            tmp[i+j,0] += yhat[i,j]
            tmp[i+j,1] += 1
            
    for i in range(tmp.shape[0]):
        tmp[i,0] = tmp[i,0]/tmp[i,1]
        
    yhat_flat = tmp[:,0]
    test_Y_flat = test_set_series.iloc[-15:,:].values.reshape(-1)
    print('yhat = ',yhat_flat)
    print('test_Y = ',test_Y_flat)
    #calculate rmse
    rmse = np.sqrt(mean_squared_error(yhat_flat,test_Y_flat))
    print('rmse = ',rmse)
    
    #plot predictions
    ##get index of test_set_series
    indices = test_set_series.index[-15:]
    yhat_df = pd.DataFrame(yhat_flat,index=indices)
    ## 
    plt.plot(test_set_series,label='test_set')
    plt.plot(yhat_df,label='predict')
    plt.legend()
    plt.show()
    
#    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#    # invert scaling for forecast
#    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
#    #inv_yhat = scaler.inverse_transform(inv_yhat)
#    inv_yhat = inv_yhat[:,0]
#    # invert scaling for actual
#    test_Y = test_Y.values.reshape((len(test_Y), 1))
#    inv_y = np.concatenate((test_Y, test_X[:, 1:]), axis=1)
#    inv_y = inv_y[:,0]
#    # calculate RMSE
#    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
#    print('Test RMSE: %.3f' % rmse)
#    
if __name__=='__main__':
    test()
