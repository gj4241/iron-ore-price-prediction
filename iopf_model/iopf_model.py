# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:44:31 2018

@author: user
"""

from keras.layers import Input, GRU, Dense, Dropout
from keras.models import Model

def iopf_model(timesteps,outlen=1,data_dim=1):

    main_input = Input(shape=(timesteps,data_dim),dtype='float32',
                       name = 'main_input')
    X = GRU(128,return_sequences=True,
            input_shape=(timesteps,data_dim))(main_input)
    X = GRU(128)(X)
    X = Dense(64,activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(16,activation='relu')(X)
    X = Dropout(0.5)(X)

    main_output = Dense(outlen,name='main_output')(X)

    model = Model(inputs=main_input,outputs=main_output)

    return model
