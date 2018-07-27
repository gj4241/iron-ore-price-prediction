# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 18:00:12 2018

@author: user
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

from iopf_model import iopf_model
from utils.data_utils import series_to_supervised
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

data_dir = 'data'
log_dir = 'log/iopf_model'

time_steps = 20
batch_size = 32
epochs = 200

hparams = {'optimizer':'adam',
           'loss':'mse',
           'learning_rate':0.001,
           'decay':0.01
           }

def train(hparams = hparams):


    # train / test split
    dataset = pd.read_csv(os.path.join(data_dir,'dataset1.csv'),index_col=0)
    train_len = int(dataset.shape[0]*0.8)
    train_set = dataset.iloc[:train_len,:]
    test_set = dataset.iloc[train_len:,:]

    # save test set
    test_set.to_csv(os.path.join(data_dir,'testset1.csv'))
    mmsc = MinMaxScaler()
    train_set = mmsc.fit_transform(train_set)

    data_dim = train_set.shape[1]

    train_sup = series_to_supervised(train_set,time_steps,1)
    train_X = train_sup.iloc[:,:200].values.reshape((-1,time_steps,data_dim))
    train_y = train_sup.iloc[:,200]

    #load model or create new
    if os.path.exists(os.path.join(log_dir,'iopf_model.h5')):
        print("loading pre-trained model...")
        model = load_model(os.path.join(log_dir,'iopf_model.h5'))
    else:
        print("Compile new model...")
        model = iopf_model(time_steps,1,data_dim)



    optimizer = hparams['optimizer']

    if optimizer == 'adam':
        optimizer = Adam(lr=hparams['learning_rate'],decay=hparams['decay'])
    loss = hparams['loss']
    model.compile(optimizer=optimizer,loss=loss)

    print(model.summary())

    callbacks = [TensorBoard(log_dir=log_dir)]
    history = model.fit(train_X,train_y,epochs=epochs,batch_size=batch_size,
                        callbacks=callbacks)
    plt.plot(history.history['loss'],label='train')
    plt.legend()
    plt.show()

    #save model
    model.save(os.path.join(log_dir,'iopf_model.h5'))

if __name__=="__main__":
    train(hparams)