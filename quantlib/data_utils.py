import pandas as pd
import numpy as np
import requests
import yfinance as yf
import datetime
import random
from bs4 import BeautifulSoup



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
    symbols = random.sample(symbols, cnt) #symbols[:cnt]
    # symbols = ['SHW', 'TRGP', 'CCL', 'VICI', 'CHTR', 'EW', 'UAL', 'PODD', 'HUM', 'MDLZ', 'LLY', 'O', 'SBAC', 'MOH', 'CTLT', 'ZBRA', 'QCOM', 'NUE', 'ACN', 'BEN', 'ANSS', 'AAL', 'NOW', 'ADBE', 'LEN', 'RSG', 'ADSK', 'HCA', 'SJM', 'ATO', 'GOOGL', 'AMZN']
    symbols = ['FSLY', 'QLYS', 'VRNS', 'S', 'CYBR', 'BX', 'CSL', 'MASI', 'FDX', 'SPX', 'AMT', 'UDR', 'EQR', 'AVB', 'PSA', 'NSA', 'CUBE', 'PLD', 'EXR', 'LUV', 'AAL', 'UAL', 'DAL', 'ALK', 'NVO', 'NET', 'PYPL', 'CHKP', 'FTNT', 'ZS', 'LEN', 'MDLZ', 'NUE', 'RSG', 'SHW', 'CRWD', 'MSFT', 'GOOGL', 'AAPL', 'TSLA', 'AMZN', 'SAP', 'META', 'EOG', 'NVDA', 'PANW', 'JPM', 'AOS', 'ABT', 'AFL', 'ORCL', 'CAT', 'LMT', 'AMD', 'APD', 'LDOS', 'BSX', 'SYK', 'HLT', 'CHD', 'TRGP', 'LLY', 'GE', 'PTC', 'MPWR', 'GSK', 'VRTX','VRCA',  'NIO', 'AMC' ] # the good stock
    if "GOOGL" not in symbols:
        symbols.append("GOOGL")
    if "AMZN" not in symbols:
        symbols.append("AMZN")
    symbols.sort()
    ohlcvs = {}
    for symbol in symbols:
        symbol_df = yf.Ticker(symbol).history(period="10y")
        
        ohlcvs[symbol] = symbol_df[["Open", "High", "Low", "Close", "Volume"]].rename(
            columns= {
                "Open":     "open",
                "High":     "high",
                "Low":      "low",
                "Close":    "close",
                "Volume":   "volume"
            }
        )
        # print(ohlcvs[symbol])
        # break;

    df1 = pd.DataFrame(index=ohlcvs["GOOGL"].index)
    df = df1.tz_localize(None)
    # df.index.name = "date"
    # # print(df.index)
    instruments = list(ohlcvs.keys())
    count = 0

    for inst in instruments:
        inst_df1 = ohlcvs[inst]
        inst_df = inst_df1.tz_localize(None)
        # columns = list(map(lambda x: "{} {}".format(inst, x), inst_df.columns)) # this transforms the open, high ... to AAPL open, AAPL high ..
        # df[columns] = inst_df
        df = pd.concat([df, inst_df], axis=1)
        df = df.rename(
            columns={
                "open":     inst+" open",
                "high":     inst+" high",
                "low":      inst+" low",
                "close":    inst+" close",
                "volume":   inst+" volume"
            }
        )
        # break
        # count = count + 1
        # if count > 1:
        #     break;
    df.index.name = "date"
    return df, instruments

# df, instruments = get_sp500_df()
# # df.head()
# # print(df)
# # print(instruments)
# df.to_excel("./Data/sp500_data.xlsx")

# take an ohlcv df and add some other statistics
def format_date(dat):
    yymmdd = list(map(lambda x: int(x), str(dat).split(" ")[0].split("-")))
    return datetime.date(yymmdd[0], yymmdd[1], yymmdd[2])

def extend_dataframe(traded, df, window_size, num_of_std):
    df.index = pd.Series(df.index).apply(lambda x: format_date(x))
    open_cols = list(map(lambda x: str(x) + " open", traded))
    high_cols = list(map(lambda x: str(x) + " high", traded))
    low_cols = list(map(lambda x: str(x) + " low", traded))
    close_cols = list(map(lambda x: str(x) + " close", traded))
    volume_cols = list(map(lambda x: str(x) + " volume", traded))
    historical_data = df.copy()
    historical_data = historical_data[open_cols + high_cols + low_cols + close_cols + volume_cols]
    historical_data.fillna(method="ffill", inplace=True) # fill missing data by first forward filling data, such that [] [] [] a b c [] [] [] becomes [] [] [] a b c c c c
    historical_data.fillna(method="bfill", inplace=True) # fill missing data by backward filling data, such that [] [] [] a b c c c c becomes a a a a b c c c c
    for inst in traded:
        historical_data["{} % ret".format(inst)] = historical_data["{} close".format(inst)] / historical_data["{} close".format(inst)].shift(1) - 1 # close to close return statistic
        historical_data["{} % ret vol".format(inst)] = historical_data["{} % ret".format(inst)].rolling(25).std() # historical rolling standard deviation of returns as realized volatility proxy
        # test if stock is actively trading by using rough measure of non-zero price change from previous time step
        historical_data["{} % active".format(inst)] = historical_data["{} close".format(inst)] != historical_data["{} close".format(inst)].shift(1)
        # add bollinger bands and O/U bought signal
        historical_data["{} 20day_mean".format(inst)], historical_data["{} upperband".format(inst)], historical_data["{} lowerband".format(inst)] = bollinger_bands(historical_data["{} close".format(inst)], window_size, num_of_std)
        historical_data["{} ou_signal".format(inst)] = np.where(historical_data["{} close".format(inst)] > historical_data["{} upperband".format(inst)], 1, np.where(historical_data["{} close".format(inst)] < historical_data["{} lowerband".format(inst)], -1, np.nan))
    return historical_data 

"""
There are multiple ways to fill missing data, depending on your requirements and purpose
Some options
1. Ffill -> bfill
2. Brownian motion/bridge
3. GARCH/GARCH Copula et cetera
4. Synthetic Data, such as GAN and Stochastic Volatility Neural Networks

For choices differ for your requirements. For instance, in backtesting you might favou (1), while in training neural models you might favor (4)
The data cycle can be very complicated, with entire research teams dedicated to obtaining, processing and extracting signals from structured/unstructured data.
We are dealing with well behaved data that is structured and already cleaned for by Yahoo Finance API
"""


# df = extend_dataframe(traded=instruments, df=df)
# print(df)
# df.to_excel("./Data/hist.xlsx")

def df_to_excel(df):
    df.to_excel("./Data/hist.xlsx")

def df_from_excel():
    df = pd.read_excel("./Data/hist.xlsx")
    df = df.rename(columns={'Unnamed: 0': "date"}).set_index("date")
    df.index = pd.Series(df.index).apply(lambda x: format_date(x))
    return df


# df = pd.read_excel("./Data/hist.xlsx")
# df = df.rename(columns={'Unnamed: 0': "date"}).set_index("date")
# df.index = pd.Series(df.index).apply(lambda x: format_date(x))
# print(df)
# print(type(df.index[0]))

# Use bollinger bands to check for overbought or underbought stocks
def bollinger_bands(df, window_size, num_of_std):
    rolling_mean = df.rolling(window=window_size).mean()
    rolling_std = df.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band

def add_over_under_signal(traded, portfolio, window_size, num_of_std):
    for inst in traded:
        portfolio["{} 20DayMean".format(inst)], portfolio["{} upperband".format(inst)], portfolio["{} lowerband".format(inst)] = bollinger_bands(portfolio["{} close".format(inst)], window_size, num_of_std)
        portfolio["{} ou_signal".format(inst)] = np.where(portfolio["{} close".format(inst)] > portfolio["{} upperband".format(inst)], 1, np.where(portfolio["{} close".format(inst)] < portfolio["{} lowerband".format(inst)], -1, np.nan))

def extract_columns_and_date(df, dvalue):
    cols_df = df.filter(like=' '+dvalue).join(df["date"])
    cols = cols_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cols_df = cols_df[cols]
    return cols_df




