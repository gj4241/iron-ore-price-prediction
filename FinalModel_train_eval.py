# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:53:42 2018

@author: user
"""

from FinalModel import FinalModel

def main():
    finalModel = FinalModel(
            booster='gbtree',
            objective='reg:linear',
            max_depth=7,
            subsample=0.83,
            colsample_bytree=0.8,
            silent=1,
            eval_metric='rmse')

    # get trainX trainY validX validY

    return 0

if __name__=="__main__":
    main()