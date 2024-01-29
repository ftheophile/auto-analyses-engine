import pandas as pd
import numpy as np
import requests
import yfinance as yf
import quantlib.data_utils as du
import quantlib.general_utils as gu
from bs4 import BeautifulSoup
import random


def get_sp500_instruments():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content, 'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))
    return list(df[0]["Symbol"])


# tickers = get_sp500_instruments()
# print(tickers)

def get_sp500_df(cnt):
    symbols = get_sp500_instruments()
    symbols1 = random.sample(symbols, cnt) #symbols[:cnt]
    print(len(symbols), symbols1)

# get_sp500_df(30)
def create_local_data(cnt):
    df1, instruments = du.get_sp500_df(cnt)
    df = du.extend_dataframe(traded=instruments, df=df1)
    return df1

# df1 = create_local_data(30)
# print(df1.tail())

# for i in range(1900):
#     print('insert into "SYSTEM"."T_PERSONS" values(\''+str(i)+'14551ztc-f022-4bd7-5163-a72f3e1rs\',\'Loana\',\'Lohnson\');')

#     Could not execute 'insert into "SYSTEM"."T_PERSONX" select "ID", "FIRSTNAME", "LASTNAME" from "SYSTEM"."T_PERSONY"' in 1:01.954 minutes . 
# [129]: transaction rolled back by an internal error: Allocation failed ; $failure_type$=GLOBAL_ALLOCATION_LIMIT; $failure_flag$=; $size$=197419520; $name$=0xfe00004bb3; $type$=pool; $inuse_count$=52110885; $allocated_size$=11089727164 


# squ_b2153a3c2b670ffee74ed24acc4b7951d187b7a2

df, instruments = df, instruments = gu.load_file("./Data/data.obj")
window_size = 20
num_of_std = 2
portfolio = pd.DataFrame()
#portfolio = pd.DataFrame(index=df["2023-10-01":].index).reset_index()
portfolio['AAPL close'] = df[40:]['AAPL close']

def bollinger_bands(df, window_size, num_of_std):
    rolling_mean = df.rolling(window=window_size).mean()
    rolling_std = df.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band

portfolio['20DayMean'], portfolio['UpperBand'], portfolio['LowerBand'] = bollinger_bands(portfolio['AAPL close'], window_size, num_of_std)
portfolio['Signal'] = np.where(portfolio['AAPL close'] > portfolio['UpperBand'], 1, np.where(portfolio['AAPL close'] < portfolio['LowerBand'], -1, np.nan))

print(portfolio.tail())

