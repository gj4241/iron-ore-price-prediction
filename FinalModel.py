# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:19:28 2018

@author: user
"""

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

class FinalModel():
    def __init__(self,
                 booster,
                 objective,
                 max_depth,
                 subsample,
                 colsample_bytree,
                 silent,
                 eval_metric):
        self.booster = booster
        self.objective = objective
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.silent = silent
        self.eval_metric = eval_metric

    def train(self,trainX,trainY,num_round=5000):
        dtrain = xgb.DMatrix(trainX,trainY,feature_names=trainX.columns.values)
        params = {
                'booster':self.booster,
                'objective':self.objective,
                'max_depth':self.max_depth,
                'subsample':self.subsample,
                'colsample_bytree':self.colsample_bytree,
                'silent':self.silent,
                'eval_metric':self.eval_metric}
        eval_list = [(dtrain,'train')]
        print('training xgb model...')
        bst = xgb.train(params,dtrain,num_round,eval_list)
        self.bst = bst

    def evaluate(self,validX,validY):
        print("evaluating xgb model...")
        dvalid = xgb.DMatrix(validX,feature_names=validX.columns.values)
        predY = self.bst.predict(dvalid)
        validY = validY.values
        rmse = np.sqrt(mean_squared_error(predY,validY))


        plt.plot(predY, label='prediction')
        plt.plot(validY, label='true')
        plt.title('rmse = {:4f}'.format(rmse))
        plt.legend()
        plt.show()

    def inference(self,X):
        predY = self.bst.predict(X)
        print(predY)

    def set_bst(self,bst):
        self.bst=bst

