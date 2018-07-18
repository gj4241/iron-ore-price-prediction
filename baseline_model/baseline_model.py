# -*- coding: utf-8 -*-
"""
baseline model using rnn with gru cells

5일*4주 -> input seq_length
5일 -> output_length

Created on Tue Jul 17 15:02:55 2018

@author: user
"""

from keras.layers import Input, GRU, Dense
from keras.models import Model

def baseline_model(timesteps=20,data_dim=1):
    """
    Arguments:
        timesteps: int, timesteps of input
        data_dim: int, dimensions of input
    """
    
    main_input = Input(shape=(timesteps,data_dim),dtype='float32',name='main_input')
    X = GRU(32,return_sequences=True,
            input_shape=(timesteps,data_dim))(main_input)
    X = GRU(32,return_sequences=True)(X)
    X = GRU(32)(X)
    X = Dense(16,activation='relu')(X)
    main_output = Dense(5,name='main_output')(X)
    
    model = Model(inputs=main_input,outputs=main_output)
    
    return model
    