# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 18:00:12 2018

@author: user
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

from utils.data_utils import series_to_supervised
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input,GRU,Dense,Dropout
from keras.models import load_model, Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

data_dir = 'data'
log_dir = 'log/iopf_model'

time_steps = 20
batch_size = 32
epochs = 100

hparams = {'optimizer':'adam',
           'loss':'mae',
           'learning_rate':0.001,
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
    train_X = train_sup.iloc[:,:200]
    train_y = train_sup.iloc[:,200]

    #load model or create new
    #if os.path.exists(os.path.join(log_dir,'iopf_model.h5')):
    #    model = load_model(os.path.join(log_dir,'iopf_model.h5'))
    #else:
    #model = iopf_model(time_steps,data_dim)

    main_input = Input(shape=(time_steps,data_dim),dtype='float32',
                       name = 'main_input')
    X = GRU(128,return_sequences=True,
            input_shape=(time_steps,data_dim))(main_input)
    X = GRU(128)(X)
    X = Dense(64,activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(32,activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(16,activation='relu')(X)
    X = Dropout(0.5)(X)

    main_output = Dense(1,name='main_output')(X)

    model = Model(inputs=main_input,outputs=main_output)


    optimizer = hparams['optimizer']
    if optimizer == 'adam':
        optimizer = Adam(lr=hparams['learning_rate'],decay=0.0)
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