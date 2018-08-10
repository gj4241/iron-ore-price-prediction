# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 21:27:50 2018

@author: kkalla
"""

import argparse
import pickle

import pandas as pd
import xgboost as xgb
from FinalModel import FinalModel

parser = argparse.ArgumentParser()
parser.add_argument("--date","-d",type=str,default="2018/08/03",
                    help="yyyy/mm/dd format")

args = parser.parse_args()

def infer():
    # get data of the given dates
    t1_data = pd.read_csv('data/t1_data.csv',parse_dates=['Date'])
    input_date = pd.to_datetime(args.date,format="%Y/%m/%d")
    X = t1_data.drop(['Date','target'],axis=1).iloc[-7,:]

    # initialize model
    finalModel = FinalModel(
            booster='gbtree',
            objective='reg:linear',
            max_depth=7,
            subsample=0.83,
            colsample_bytree=0.8,
            silent=1,
            eval_metric='rmse')

    finalModel.set_bst(pickle.load(
                open("log/final_model.pkl","rb")))

    finalModel.inference(X,feature_names=X.index.values)



if __name__ == "__main__":
    infer()