# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:00:55 2018

@author: user
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import series_to_supervised
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model
log_dir = 'log/iopf_model'

def test():
    test_set = pd.read_csv('data/testset1.csv',index_col=0)
    mmsc = MinMaxScaler()


    time_steps = 20
    data_dim = test_set.shape[1]

    test_sup = series_to_supervised(test_set,time_steps,1)
    test_X = test_sup.iloc[:,:200].values.reshape((-1,time_steps,data_dim))
    test_X = mmsc.fit_transform(test_X)
    test_y = test_sup.iloc[:,200].values

    print("test_X.shape = ",test_X.shape)
    print("test_y.shape = ",test_y.shape)

    #load model
    model = load_model(os.path.join(log_dir,'iopf_model.h5'))
    print(model.summary())
    pred_y = model.predict(test_X)
    print("prediction: ",pred_y)
    rmse = np.sqrt(mean_squared_error(pred_y,test_y))
    print("RMSE = ",rmse)

    plt.plot(test_y,label='test')
    plt.plot(pred_y,label='prediction')
    plt.legend()
    plt.show()

if __name__=="__main__":
    test()