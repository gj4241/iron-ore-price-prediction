# -*- coding: utf-8 -*-
"""
train_data_weekly.csv 파일로 t1_data ~ t5_data 생성

Created on Tue Aug  7 13:11:07 2018

@author: kkalla
"""

import os, subprocess
import pandas as pd

data_path = '../data/train_data_weekly.csv'

def main():
    if not os.path.exists(data_path):
        subprocess.call(["python","daily_to_weekly.py"])

    # 현재 -> 과거 순으로 변환
    raw_weekly = pd.read_csv(data_path)[::-1]

    # 맨위에 7개의 빈칸 생성
    empty_df = pd.DataFrame(columns=raw_weekly.columns)
    empty_df = empty_df.append([1,1,1,1,1,1,1])
    empty_df = empty_df.iloc[:,:13]

    result = pd.concat([empty_df,raw_weekly])
    result = result.reset_index(drop=True)

    # 시점별로 1주~7주 당기기
    feature1 = result[['BCI','BCTR','FMG']][1:].reset_index(drop=True)
    feature2 = result[['전기동_현물','니켈_현물',
                       'RICI_Metals','FVJ']][2:].reset_index(drop=True)
    feature5 = result[['알루미늄_현물']][5:].reset_index(drop=True)
    feature7 = result[['BHP','RIO']][7:].reset_index(drop=True)

    result_table = pd.concat(
            [result[['Date','target']],feature1,feature2,feature5,feature7],
            axis=1)

    # 1주~5주 당겨진 target_1 ~ target_5 columns 생성
    for i in range(5):
        a=result.target[i+1:]
        a.reset_index(drop=True,inplace=True)
        result_table['target_'+str(i+1)]=a

    # 과거 -> 현재 순으로 변경 & moving average with window 5
    result_table = result_table[::-1].reset_index(drop=True)
    result_table2 = result_table.drop(
            ['Date','target'],axis=1).rolling(5,min_periods=1).mean()
    result_table = pd.concat(
            [result_table[['Date','target']],result_table2],axis=1)

    # 각 시점별 필요한 데이터만 따로 저장.
    t1_data = result_table.drop(
            ['target_2','target_3','target_4','target_6'],axis=1)
    t2_data = result_table.drop(
            ['target_1','target_3','target_4','target_5','BCI','BCTR','FMG'],
            axis=1)
    t3_data = result_table.drop(
            ['target_1','target_2','target_4','target_5',
             'BCI','BCTR','FMG','전기동_현물','니켈_현물','RICI_Metals','FVJ'],
             axis=1)
    t4_data = result_table.drop(
            ['target_1','target_2','target_3','target_5',
             'BCI','BCTR','FMG','전기동_현물','니켈_현물','RICI_Metals','FVJ'],
             axis=1)
    t5_data = result_table.drop(
            ['target_1','target_2','target_3','target_4',
             'BCI','BCTR','FMG','전기동_현물','니켈_현물','RICI_Metals','FVJ'],
             axis=1)

    # save as csv
    t1_data.to_csv('../data/t1_data.csv',index=False)
    t2_data.to_csv('../data/t2_data.csv',index=False)
    t3_data.to_csv('../data/t3_data.csv',index=False)
    t4_data.to_csv('../data/t4_data.csv',index=False)
    t5_data.to_csv('../data/t5_data.csv',index=False)

if __name__=="__main__":
    main()