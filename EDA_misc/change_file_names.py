# -*- coding: utf-8 -*-
"""
파일이름 변경
컬럼명으로 파일이름은 변경함.

Created on Thu Jul 19 16:35:19 2018

@author: user
"""

import os,re

import pandas as pd

data_dir = '../raw_data_daily'

def main():
    file_name_list = os.listdir(data_dir)
    colnames_list = []
    
    # reading column names
    print('reading column names...')
    for file_name in file_name_list:
        if file_name.split('.')[1] == 'csv':
            file_path = os.path.join(data_dir,file_name)
            colname = pd.read_csv(file_path, engine='python',
                              skiprows=6).columns.values
            if len(colname) < 3:
                colnames_list.append(colname[1])
                new_file_name = re.sub('[/]',',',colname[1]) + '.csv' #'/'제
                os.rename(file_path,os.path.join(data_dir,new_file_name))
    
    print('DONE!!')
    print(colnames_list)
    
if __name__=='__main__':
    main()
        
    
