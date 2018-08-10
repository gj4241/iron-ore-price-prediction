# -*- coding: utf-8 -*-
"""
xgb model을 위한 data관련 utils

Created on Thu Aug  9 15:59:48 2018

@author: kkalla, qwerty1434
"""

def train_valid_split(Training_data,train_percent):
    train_data_num = round(len(Training_data)*train_percent)
    Train_data = Training_data[:train_data_num]
    Valid_data = Training_data[~Training_data.index.isin(Train_data.index)]
    TrainX = Train_data.drop(['target'],axis = 1)
    TrainY = Train_data['target']
    ValidX = Valid_data.drop(['target'],axis = 1)
    ValidY = Valid_data['target']
    return TrainX,TrainY,ValidX,ValidY