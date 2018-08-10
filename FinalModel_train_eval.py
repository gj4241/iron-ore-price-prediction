# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:53:42 2018

@author: user
"""

from __future__ import absolute_import

import argparse,os,subprocess
import pickle
import pandas as pd

from FinalModel import FinalModel
from utils.data_utils import train_valid_split

parser = argparse.ArgumentParser()
parser.add_argument('--which_week','-ww',choices=['t1','t2','t3','t4','t5'],
                    default='t1')
parser.add_argument('--model_dir','-md',type=str)
parser.add_argument('--mode',choices=['train','eval'],default='train')

args = parser.parse_args()

def main():
    # initialize model
    finalModel = FinalModel(
            booster='gbtree',
            objective='reg:linear',
            max_depth=7,
            subsample=0.83,
            colsample_bytree=0.8,
            silent=1,
            eval_metric='rmse')

    # load dataset
    data_path = os.path.join('data',args.which_week+'_data.csv')
    if not os.path.exists(data_path):
        subprocess.call(["python","utils/make_t1_to_t5.py"])
    df = pd.read_csv(data_path,parse_dates=['Date'])
    train_data = df.drop('Date',axis=1)[:-5].dropna()

    # get trainX trainY validX validY
    trainX,trainY,validX,validY = train_valid_split(train_data,0.7)

    # train xgb model
    if args.mode == 'train':
        finalModel.train(trainX,trainY,num_round=6000)
        # pickle xgb model
        if args.model_dir is not None:
            print("saving xgb model...")
            pickle.dump(finalModel.bst,
                        open(os.path.join(args.model_dir,"final_model.pkl"),"wb"))


    # valid xgb model
    if args.mode == 'eval':
        assert args.model_dir is not None, "Model directory error!!"
        finalModel.set_bst(pickle.load(
                open(os.path.join(args.model_dir,"final_model.pkl"),"rb")))
    finalModel.evaluate(validX,validY)

if __name__=="__main__":
    main()