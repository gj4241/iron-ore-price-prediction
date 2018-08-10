# -*- coding: utf-8 -*-
"""
주간데이터 병

Created on Mon Jul 30 19:27:44 2018

@author: kkalla,
"""
import datetime

import pandas as pd


def main():
    df_1 = pd.read_csv('../raw_data_weekly/철광석비교광물1_주간.csv',
                    encoding='euc-kr',skiprows=6,
                    parse_dates=['Unnamed: 0'],engine='python')
    df_2 = pd.read_csv('../raw_data_weekly/1.csv',
                    encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'])
    df_3 = pd.read_csv('../raw_data_weekly/2.csv',
                    encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'])
    df_4 = pd.read_csv('../raw_data_weekly/환율_주간.csv',
                    encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'],
                    engine='python')
    df_5 = pd.read_csv('../raw_data_weekly/600019.SS.csv',
                    encoding='euc-kr',parse_dates=['Date'])
    df_6 = pd.read_csv('../raw_data_weekly/000709.SZ.csv',
                    encoding='euc-kr',parse_dates=['Date'])

    df_5=df_5[['Date','Close']]
    df_6=df_6['Close']
    df_stock_china= pd.concat([df_5,df_6],axis=1)
    df_stock_china.columns=['Date','Baoshan','Hbis']

    df_stock_1=pd.read_csv('../raw_data_weekly/FMG.AX.csv',
                    encoding='euc-kr',parse_dates=['Date'])
    df_stock_2=pd.read_csv('../raw_data_weekly/VALE.csv',
                    encoding='euc-kr',parse_dates=['Date'])
    df_stock_3=pd.read_csv('../raw_data_weekly/BHP.csv',
                    encoding='euc-kr',parse_dates=['Date'])
    df_stock_4=pd.read_csv('../raw_data_weekly/RIO.csv',
                    encoding='euc-kr',parse_dates=['Date'])

    df_stock_5=pd.read_csv('../raw_data_weekly/FVJ.SG.csv',
                    encoding='euc-kr',parse_dates=['Date'])

    df_stock_1=df_stock_1[['Date','Close']]
    df_stock_2=df_stock_2['Close']
    df_stock_3=df_stock_3['Close']
    df_stock_4=df_stock_4['Close']
    df_stock_5=df_stock_5['Close']
    df_stock_world= pd.concat([df_stock_1,df_stock_2,df_stock_3,df_stock_4,df_stock_5],axis=1)
    df_stock_world.columns=['Date','FMG','VALE','BHP','RIO','FVJ']
    df_stock_china=df_stock_china.sort_values(by = "Date", ascending=False)
    df_stock_world=df_stock_world.sort_values(by = "Date", ascending=False)
    df_stock_world=df_stock_world[:-1]


    df1 = pd.merge(df_1,df_2,on=['Unnamed: 0', '철광석 Fines (Daily) [중국(수입가 CFR)] 현물USD/ton'])
    df2 = pd.merge(df1,df_3,on=['Unnamed: 0', '철광석 Fines (Daily) [중국(수입가 CFR)] 현물USD/ton'])
    df3 = pd.merge(df2,df_4,on=['Unnamed: 0', '철광석 Fines (Daily) [중국(수입가 CFR)] 현물USD/ton'])
    df=df3
    df_stock_china.Date= df_stock_china.Date + datetime.timedelta(days=1)
    df_stock_world.Date= df_stock_world.Date + datetime.timedelta(days=1)
    df.columns=['Date', 'target',
           '전기동_현물', '알루미늄_현물', '니켈_현물','아연_현물', '납_현물', 
           '주석_현물','NASAAC_현물', '몰리브덴_현물','코발트_현물', 
           '철광석_DCE_현물','철광석_FE_CFR_현물',
           'TSI_CFR_China_62_Index','Platts_CFR_China_Index',
           'MB_CFR_China_58_Index','TSI_CFR_China_58_Index',
           '철광석_65_Pellet_현물','철광석_Fine_62', 
           '중국_철광석_항구재고량mmt_x',
           '중국_철광석_항구재고량mmt_y', 
           'BCI','BCTR','TR/CC_CRB_Index',
           'GSCI', 'GSCI_Energy', 'GSCI_Petroleum','GSCI_Non-Energy',
           'GSCI_Industrial_Metals','GSCI_Precious_Metals',
           'RICP',
           'PDS_원자재_지수', 'PDS_철강_하위지수',
           'TOCOM_Index',
           'RICI','RICI_Energy','RICI_Metals','RICI_Agriculture',
           'BDI','BCI','BPI','BCTI','BDTI','BSI','BHSI',
           'SSY/PCI',
           'SSY/ACI',
           'HRCI',
           'CCFI','SCFI','CBFI','CNPI','CNDPI','CNTPI','CNCPI',
           'USD/CNY', 'AUD/USD','USD/KRW', 'AUD/KRW','CNY/KRW', 'USD/RMB']
    df=pd.merge(df_stock_world,df,on='Date',how='right')
    df=pd.merge(df_stock_china,df,on='Date',how='right')
    df.to_csv(('../data/raw_data_weekly_merged.csv'),encoding='euc-kr',index=False)

if __name__ =="__main__":
    main()
