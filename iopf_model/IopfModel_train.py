# -*- coding: utf-8 -*-
"""
Training IopfModel

Created on Mon Jul 30 20:37:32 2018

@author: kkalla
"""

import argparse
import pandas as pd

from IopfModel import IopfModel
from utils.data_utils import get_trainX_trainY
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",'-dp',type=str,
                    help="File path of dataset",
                    default = "data/raw_data_merged.csv")
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

args = parser.parse_args()

def main():
    # Load trainX and trainY
    sc = MinMaxScaler()
    df = pd.read_csv(args.data_path)
    train_X,train_y,data_dim,sc = get_trainX_trainY(df,
                                        timesteps = args.timesteps,
                                        output_len = args.timesteps,
                                        scaler = sc)
    train_X = train_X.values.reshape((-1,args.timesteps,data_dim))

    # init IopfModel
    iopfModel = IopfModel(timesteps=args.timesteps,
                          out_len=args.output_len,
                          data_dim=data_dim,
                          )

    print("begin training...")
    iopfModel.train(train_X,train_y,log_dir=args.log_dir)




if __name__=="__main__":
    main()