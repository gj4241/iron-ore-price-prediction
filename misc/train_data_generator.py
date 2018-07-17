# -*- coding: utf-8 -*-
"""
철광석데이터만 뽑아서 model train을 위한 데이터셋을 만듭니다.

Created on Tue Jul 17 14:27:39 2018

@author: kkalla
"""

import os

import pandas as pd

save_dir = '../data/baseline_model'
file_name = 'ironOre.csv'

def main():
    raw_data = pd.read_csv('../주요철강가격 통합.csv',usecols=[0,1],
                           encoding='euc_kr',engine='python')
    raw_data.sort_values('날짜',inplace=True)
    raw_data.set_index('날짜',inplace=True)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    raw_data.to_csv(os.path.join(save_dir,file_name))
    print("DONE!!!")
    
if __name__ == "__main__":
    main()