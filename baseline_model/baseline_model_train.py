# -*- coding: utf-8 -*-
"""
training baseline model

Created on Tue Jul 17 22:23:37 2018

@author: kkalla
"""
from __future__ import absolute_import

import os

import pandas as pd
import matplotlib.pyplot as plt

from baseline_model import baseline_model
from utils.data_utils import train_test_split,series_to_supervised
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model

data_dir = 'data'
log_dir = 'log'

timeSteps = 20
data_dim = 1

learning_rate = 0.0005
epochs = 1000
batch_size = 32

hparams={
        'optimizer':Adam(lr=learning_rate,decay=0.01),
        'loss':'mae'}

def train(hparams=hparams):
    #Train data loading
    ## Train 없으면 Train_test_split실행
    if not os.path.exists(os.path.join(data_dir,'train.csv')):
        print('splitting train and test sets')
        train_test_split(data_dir)
            
    train_set = pd.read_csv(os.path.join(data_dir,'train.csv'),index_col=0)
    
    #seq to supervised
    train_set = series_to_supervised(train_set,n_in=20,n_out=5)
    # train_set to train_X,Y(85%) and valid set(15%)
    valid_indices = int(train_set.shape[0]*0.85)
    train_X = train_set.iloc[
            :valid_indices,range(20)].values.reshape(-1,timeSteps,data_dim)
    train_Y = train_set.iloc[
            :valid_indices,range(20,25)].values
    valid_X = train_set.iloc[
            valid_indices::,range(20)].values.reshape(-1,timeSteps,data_dim)
    valid_Y = train_set.iloc[
            valid_indices::,range(20,25)]
    
    #load model
    if os.path.exists(os.path.join(log_dir,'baseline_model.h5')):
        model = load_model(os.path.join(log_dir,'baseline_model.h5'))
    else:
        model = baseline_model(timeSteps,data_dim)
    
    #compile model with hparams
    optimizer = hparams['optimizer']
    loss = hparams['loss']
    model.compile(optimizer=optimizer,loss=loss)
    
    #fit model
    callbacks = [TensorBoard(log_dir=log_dir)]
    history = model.fit(train_X,train_Y,epochs=epochs,batch_size=batch_size,
              validation_data=(valid_X,valid_Y),callbacks=callbacks)
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='valid')
    plt.legend()
    plt.show()
    
    #save model
    model.save(os.path.join(log_dir,'baseline_model.h5'))
    
if __name__=='__main__':
    train()
