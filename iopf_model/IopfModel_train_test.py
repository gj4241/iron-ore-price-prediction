# -*- coding: utf-8 -*-
"""
Training IopfModel

Created on Mon Jul 30 20:37:32 2018

@author: kkalla
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IopfModel import IopfModel
from utils.data_utils import get_train_test,data_preprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",'-dp',type=str,
                    help="File path of dataset",
                    default = "data/train_data_weekly.csv")
parser.add_argument("--timesteps","-ts",type=int,
                    help="Input timesteps",default=25)
parser.add_argument("--output_len","-ol",type=int,
                    help="Output timesteps",default=20)
parser.add_argument('-mp','--model_path',type=str,
                    help="file path of pretrained model",
                    default="log/IopfModel.h5")
parser.add_argument("-ld","--log_dir",type=str,
                    help="Directory path to save logs",
                    default="log")
parser.add_argument("--mode",type=str,choices=['train','test'],
                    help="Choose 'train' or 'test'",default='train')
parser.add_argument("--is_pretrained",type=bool,default=False)

args = parser.parse_args()

def main():
    # Load trainX and trainY
    x_sc = MinMaxScaler()
    y_sc = MinMaxScaler()
    print("reading dataset")
    df = pd.read_csv(args.data_path,parse_dates=['Date'])
    df.sort_values(by='Date',inplace=True)
    df = data_preprocess(df)
    train_X,train_Y,test_X,test_Y,data_dim,x_sc,y_sc = get_train_test(df,
                                        timesteps = args.timesteps,
                                        output_len = args.output_len,
                                        target_col='target',
                                        x_scaler=x_sc,
                                        y_scaler=y_sc,
                                        test_len=40)
    train_X = train_X.values.reshape((-1,args.timesteps,data_dim))
    test_X = test_X.values.reshape((-1,args.timesteps,data_dim))

    # init IopfModel
    iopfModel = IopfModel(timesteps=args.timesteps,
                          out_len=args.output_len,
                          data_dim=data_dim,
                          model_path = args.model_path,
                          is_pretrained = args.is_pretrained,
                          decay = 0.01
                          )
    if args.mode=='train':
        print("begin training...")
        iopfModel.train(train_X,train_Y,log_dir=args.log_dir,
                        batch_size=int(train_X.shape[0]/5),epochs=500)

        plt.plot(iopfModel.history.history['loss'],label='train')
        plt.plot(iopfModel.history.history['val_loss'],label='valid')
        plt.legend()
        plt.show()
    elif args.mode=='test':
        print("begin test...")
        iopfModel = IopfModel(timesteps=args.timesteps,
                          out_len=args.output_len,
                          data_dim=data_dim,
                          model_path = args.model_path,
                          is_pretrained = True
                          )
        pred_Y = iopfModel.test(test_X,test_Y,y_sc)
        test_Y = y_sc.inverse_transform(test_Y.values.reshape((-1,1)))

        rmse = np.sqrt(mean_squared_error(test_Y,pred_Y))

        print("test_Y = \n",test_Y)
        print("pred_Y = \n",pred_Y)
        print("RMSE = ",rmse)

        plt.plot(test_Y,label='true')
        plt.plot(pred_Y,label='prediction')
        plt.title('rmse = {:3f}'.format(rmse))
        plt.legend()
        plt.show()


if __name__=="__main__":
    main()
