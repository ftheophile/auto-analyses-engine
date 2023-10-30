import pandas as pd
import requests
import yfinance as yf
import datetime
from bs4 import BeautifulSoup



def get_sp500_instruments():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content, 'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))
    return list(df[0]["Symbol"])


# tickers = get_sp500_instruments()
# print(tickers)

def get_sp500_df():
    symbols = get_sp500_instruments()
    symbols = symbols[:30]
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

def extend_dataframe(traded, df):
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

