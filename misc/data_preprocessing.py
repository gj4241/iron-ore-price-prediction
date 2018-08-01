# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:38:12 2018

@author: user
"""

import pandas as pd

def main():
    fvj = pd.read_csv('raw_data_daily/FVJ.SG.csv',parse_dates=['Date'])
    fmg = pd.read_csv('raw_data_daily/FMG.AX.csv',parse_dates=['Date'])
    pds = pd.read_csv('raw_data_daily/pds변수.csv',engine='python',skiprows=6,
                      parse_dates=['Date'])

    fvj_close = fvj[['Date','Close']].set_index('Date')
    fmg_close = fmg[['Date','Close']].set_index('Date')
    pds = pds.set_index('Date')

    fvj_weekly = fvj_close.resample('W-FRI').mean()
    fmg_weekly = fmg_close.resample('W-FRI').mean()
    pds_weekly = pds.resample('W-FRI').mean()

    fvj_weekly.columns = ['FVJ']
    fmg_weekly.columns = ['FMG']

    result = pd.concat([fvj_weekly,fmg_weekly,pds_weekly],axis=1)
    target_notna_indices = result.iloc[:,2].notna()
    result = result.loc[target_notna_indices,:]

    result.to_csv('data/train_data_weekly.csv')

if __name__=="__main__":
    main()