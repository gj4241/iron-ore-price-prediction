# -*- coding: utf-8 -*-
"""
철광석 현물가격과 각종 지수들을 join해서
하나의 DataFrame으로 만듭니다.

Created on Thu Jul  5 19:49:29 2018

@author: user
"""

import os

import pandas as pd

data_dir = '../data/lag'
save_dir = '../data'


def main():
    lag_dir_list = os.listdir(data_dir)
    # 철광석 vs 환율지수
    exchange_index = pd.read_csv(os.path.join(data_dir,lag_dir_list[0]),
                            encoding='euc-kr',skiprows=6)
    # 철광석 vs 주요지수1
    main_index_1 = pd.read_csv(os.path.join(data_dir,lag_dir_list[1]),
                            encoding='euc-kr',skiprows=None)
    # 철광석 vs 주요지수2
    main_index_2 = pd.read_csv(os.path.join(data_dir,lag_dir_list[2]),
                            encoding='euc-kr',skiprows=None)
    # 철광석 vs 해상운임
    ship_index = pd.read_csv(os.path.join(data_dir,lag_dir_list[3]),
                            encoding='euc-kr',skiprows=6)
    
    exchange_index = _change_colnames_set_index(exchange_index)
    main_index_1 = _change_colnames_set_index(main_index_1)
    main_index_2 = _change_colnames_set_index(main_index_2)
    ship_index = _change_colnames_set_index(ship_index)
    
    result = pd.concat([main_index_1,main_index_2,ship_index,exchange_index],
                   axis=1)
    # 철광석 가격이 nan아닌것만 저장.
    result = result.loc[~result.iloc[:,1].isnull(),:]
    # save dataFrame
    print("saving data...")
    result.to_csv(os.path.join(save_dir,'ironOre_vs_allIndex.csv'),
                  encoding='euc-kr')
    
    
def _change_colnames_set_index(dataFrame):
    col_names = dataFrame.columns.values
    col_names[0] = 'date'
    dataFrame.columns = col_names
    
    # change dtype of 'date' column from int to datetime64[ns]
    dataFrame.date = pd.to_datetime(dataFrame.date,format='%Y%m%d')
    
    # set 'date' column as index and delete 'date' column
    dataFrame.set_index('date',inplace=True)
    
    # cut some head and tail rows
    dataFrame = dataFrame.iloc[(dataFrame.index>=pd.Timestamp('2013-07-05')) & 
                               (dataFrame.index<=pd.Timestamp('2018-06-29')),:]
    
    
    return dataFrame
    
if __name__=='__main__':
    main()