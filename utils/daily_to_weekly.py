# -*- coding: utf-8 -*-
"""
일별 데이터를 불러와서 train_data_weekly.csv 생성

Created on Wed Aug  1 20:38:12 2018

@author: kkalla
"""

import pandas as pd

def main():
    print("Reading datasets")
    fvj = pd.read_csv('../raw_data_daily/FVJ.SG.csv',
            parse_dates=['Date'])
    fmg = pd.read_csv('../raw_data_daily/FMG.AX.csv',
            parse_dates=['Date'])
    pds = pd.read_csv('../raw_data_daily/pds변수.csv',
            engine='python',skiprows=6,parse_dates=['Date'])
    bhp = pd.read_csv('../raw_data_daily/BHP.csv',
            parse_dates=['Date'])
    rio = pd.read_csv('../raw_data_daily/RIO.csv',
            parse_dates=['Date'])

    fvj_close = fvj[['Date','Close']].set_index('Date')
    fmg_close = fmg[['Date','Close']].set_index('Date')
    bhp_close = bhp[['Date','Close']].set_index('Date')
    rio_close = rio[['Date','Close']].set_index('Date')
    pds = pds.set_index('Date')

    fvj_weekly = fvj_close.resample('W-FRI').mean()
    fmg_weekly = fmg_close.resample('W-FRI').mean()
    pds_weekly = pds.resample('W-FRI').mean()
    rio_weekly = rio_close.resample('W-FRI').mean()
    bhp_weekly = bhp_close.resample('W-FRI').mean()

    result = pd.concat([fvj_weekly,fmg_weekly,
        rio_weekly,bhp_weekly,pds_weekly],axis=1)
    target_notna_indices = result.iloc[:,2].notna()
    result = result.loc[target_notna_indices,:]

    result.columns = ['FVJ','FMG','RIO','BHP',
                      'target',
                      'BCI','BCTR','RICI_Metals','전기동_현물',
                      '니켈_현물','알루미늄_현물','납_현물']
    print("Saving result to csv file...")
    result.to_csv('../data/train_data_weekly.csv')

if __name__=="__main__":
    main()
