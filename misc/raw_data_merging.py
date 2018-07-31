# -*- coding: utf-8 -*-
"""
raw_data에 있는 데이터들을 merge합니다.

Created on Fri Jul 20 22:04:22 2018

@author: user
"""

import os

import pandas as pd

raw_data_dir = '../raw_data_daily'
def merge_data():
    print("merging datasets...")
    dfs = pd.DataFrame(columns=['날짜'])
    for file_name in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir,file_name)
        if file_path[-3:] == 'csv':
            df = pd.read_csv(file_path,engine='python',skiprows=6)
            df.columns = ['날짜',df.columns[1]]
            if df.shape[0]>2000:
                dfs = pd.merge(dfs,df,'outer')

    df = dfs.dropna()
    print('saving dataset...')
    df.to_csv('../data/raw_data_merged.csv',index=False,encoding='utf-8')
    print("DONE!!")

if __name__=='__main__':
    merge_data()