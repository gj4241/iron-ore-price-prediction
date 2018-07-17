# -*- coding: utf-8 -*-
"""
training baseline model

Created on Tue Jul 17 22:23:37 2018

@author: kkalla
"""
from __future__ import absolute_import

import os

import pandas as pd

from baseline_model import baseline_model
from utils.data_utils import train_test_split,series_to_supervised

data_dir = 'data/baseline_model'

timeSteps = 20
data_dim = 1

def train():
    #Train data loading
    ## Train 없으면 Train_test_split실행
    if not os.path.exists(os.path.join(data_dir,'train.csv')):
        train_test_split(data_dir)
        
    train = pd.read_csv(os.path.join(data_dir,'train.csv'))
    
    
    model = baseline_model(timeSteps,data_dim)
    
    
if __name__=='__main__':
    train()