
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import datetime 


# In[8]:


df_1 = pd.read_csv('철광석비교광물1_주간.csv',
                encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'])
df_2 = pd.read_csv('1.csv',
                encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'])
df_3 = pd.read_csv('2.csv',
                encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'])
df_5 = pd.read_csv('환율_주간.csv',
                encoding='euc-kr',skiprows=6,parse_dates=['Unnamed: 0'])
df_7 = pd.read_csv('600019.SS (2).csv',
                encoding='euc-kr',parse_dates=['Date'])
df_8 = pd.read_csv('000709.SZ (1).csv',
                encoding='euc-kr',parse_dates=['Date'])

df_7=df_7[['Date','Close']]
df_8=df_8['Close']
df_stock_china= pd.concat([df_7,df_8],axis=1)
df_stock_china.columns=['Date','Baoshan','Hbis']

df_stock_1=pd.read_csv('FMG.AX.csv',
                encoding='euc-kr',parse_dates=['Date'])
df_stock_2=pd.read_csv('VALE (1).csv',
                encoding='euc-kr',parse_dates=['Date'])
df_stock_3=pd.read_csv('BHP (1).csv',
                encoding='euc-kr',parse_dates=['Date'])
df_stock_4=pd.read_csv('RIO.csv',
                encoding='euc-kr',parse_dates=['Date'])

df_stock_5=pd.read_csv('FVJ.SG.csv',
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
df3 = pd.merge(df2,df_5,on=['Unnamed: 0', '철광석 Fines (Daily) [중국(수입가 CFR)] 현물USD/ton'])
df=df3
df_stock_china.Date= df_stock_china.Date + datetime.timedelta(days=1)
df_stock_world.Date= df_stock_world.Date + datetime.timedelta(days=1)
df.columns=['Date', 'target',
       '전기동 [LME] 현물USD/ton', '알루미늄 [LME] 현물USD/ton', '니켈 [LME] 현물USD/ton',
       '아연 [LME] 현물USD/ton', '납 [LME] 현물USD/ton', '주석 [LME] 현물USD/ton',
       'NASAAC [LME] 현물USD/ton', '몰리브덴 60% [LME] 현물USD/ton',
       '코발트 99.8% [LME] 현물USD/ton', '철광석 62% [DCE] 2018.09RMB/ton',
       '철광석 62% FE [북중국 수입가(CFR)] 현물USD/dt',
       '철광석 TSI CFR China (62% Fe Fines) Index [SGX] 2018.08USD/ton',
       '철광석 Platts CFR China (Lump Premium) Index [SGX] 2018.08USD/100dmtu',
       '철광석 MB CFR China (58% FE Fines) Index [SGX] 2018.08USD/ton',
       '철광석 TSI CFR China (58% FE Fines) Index [SGX] 2018.08USD/ton',
       '철광석 65% Pellet [중국(수입가)] 현물USD/ton',
       '철광석 Fine 62% [중국(CFR)] 현물USD/dmt', '주요 중국 철광석 항구재고량mmt_x',
       '주요 중국 철광석 항구재고량mmt_y', 'Bloomberg(DJ-UBS) Commodity Index1991=100',
       'Bloomberg Commodity Total Return1991=100', 'TR/CC CRB Index1967=100',
       'S&P GSCI1970=100', 'GSCI Energy1970=100', 'GSCI Petroleum1970=100',
       'GSCI Non-Energy1970=100', 'GSCI Industrial Metals1970=100',
       'GSCI Precious Metals1970=100',
       'Reuters Index of Commodity Prices1931.09.18=100',
       'PDS 원자재 지수2015.01.02=100', 'PDS 철강 하위지수2015.01.02=100',
       'TOCOM Index2002.05.31=100',
       'Rogers International Commodities Index1998.07.31=1000',
       'Rogers International Commodities Index Energy1998.07.31=1000',
       'Rogers International Commodities Index Metals1998.07.31=1000',
       'Rogers International Commodities Index Agriculture1998.07.31=1000',
       'Baltic Dry Index (BDI)1985.01.04',
       'Baltic Capesize Index (BCI)1999.03.01',
       'Baltic Panamax Index (BPI)1998.05.06',
       'Baltic Clean Tanker Index (BCTI)1998.08.03',
       'Baltic Dirty Tanker Index (BDTI)1998.08.03',
       'Baltic Supramax index (BSI)2005.01.01',
       'Baltic Handysize Index (BHSI)2006.05.23',
       'SSY/Pacific Capesize Index1997.1.6=4114',
       'SSY/Atlantic Capesize Index1997.1.6=5000',
       'Howe Robinson Container Index1997.1.15=1000',
       'China Containerized Freight Index (CCFI)1998.1.1=1000',
       'Shanghai Containerized Freight Index (SCFI)2009.10.16=1000',
       'China Coastal Bulk Freight Index (CBFI)2000.1.1=1000',
       'China Newbuilding Price Index (CNPI)2011.07.01=1000',
       'China Bulker Newbuilding Price Index (CNDPI)2011.07.01=1000',
       'China Tanker Newbuilding Price Index (CNTPI)2011.07.01=1000',
       'China Container Newbuilding Price Index (CNCPI)2011.07.01=1000',
       'USD/CNY (미국 달러 / 중국 인민폐)', 'AUD/USD (호주 달러 / 미국 달러)',
       'USD/KRW (미국 달러 / 한국 원)', 'AUD/KRW (호주 달러 / 한국 원)',
       'CNY/KRW (중국 인민폐 / 한국 원)', 'USD/RMB (미국 달러 / 중국 인민폐)']
df=pd.merge(df_stock_world,df,on='Date',how='right')
df=pd.merge(df_stock_china,df,on='Date',how='right')
df.to_csv(('total.csv'),
                  encoding='euc-kr')

