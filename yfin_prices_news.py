from yahoo_fin import stock_info, news as stock_news
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import enum
import os
import pathlib

cwd = os.getcwd()

DEBUG = True
TODAY = date.today().strftime('%Y%m%d')
DATA_STORAGE_DIR = cwd + '/data/' + TODAY + '/'
NEWS_DATA_STORAGE_DIR = cwd + '/data/' + TODAY + '/news/'

# make directory for today's data
pathlib.Path(NEWS_DATA_STORAGE_DIR).mkdir(parents = True, exist_ok = True)

# Window size of data set
class WindowType(enum.Enum):
    LONGLONG = '3y'  # 3 year
    LONG = '1y'  # 1 year
    MID = '20d'   # 20days
    SHORT = '5d' # 5 days (1 week)
    DAY = '1d' # day trading

# Number of data points by period, 756 points for 3yrs, 252 for 1year, 255 for 20days, 125 points for 10days (12.5 points per day x 10, 30min interval)
class DataPointType(enum.Enum):
    LONGLONG = 756
    LONG = 252
    MID = 255
    SHORT = 125

class DataIntervalType(enum.Enum):
    LONG = '1d' # daily points
    SHORT = '30m' # every 30 minutes
    SHORTSHORT = '15m' # every 15 minutes

# Get the members of major indices, DJI, S&P 500, NASDAQ, RUSSELL 2000
def get_market_index_symbols():
    dow = pd.DataFrame(stock_info.tickers_dow()) # Actually parsed from Wikipedia
    # store data in JSON format
    if not dow.empty:
        if DEBUG:
            print('Writing DOW index symbols information to JSON file ...')
        dow.to_json(DATA_STORAGE_DIR + '/dow_index_symbols.json', orient = 'split', compression = 'infer')
    else:
        print('No DOW tickers information ...')
        exit(1)

    print("* DOW members *", dow)

    sp = pd.DataFrame(stock_info.tickers_sp500()) # Actually parsed from Wikipedia
    # store data in JSON format
    if not sp.empty:
        if DEBUG:
            print('Writing S&P500 index symbols information to JSON file ...')
        sp.to_json(DATA_STORAGE_DIR + '/sp_index_symbols.json', orient = 'split', compression = 'infer')
    else:
        print('No S&P500 tickers information ...')
        exit(1)

    print("* S&P500 members *", sp)

    nasdaq = pd.DataFrame(stock_info.tickers_nasdaq()) # Actually parsed from NASDAQ
    # store data in JSON format
    if not nasdaq.empty:
        if DEBUG:
            print('Writing NASDAQ index symbols information to JSON file ...')
        nasdaq.to_json(DATA_STORAGE_DIR + '/nasdaq_index_symbols.json', orient = 'split', compression = 'infer')
    else:
        print('No NASDAQ tickers information ...')
        exit(1)

    print("* NASDAQ members *", nasdaq)

    # russell = {"RUSSELL2000": []} # todo: not yet available, 1000 is available check later

# Get news from Yahoo finance
def get_news(symbol):
    """
    :param symbol:
    :return: [{'summary':, .., 'link': url, 'published': 'Wed, 24 Nov 2021 20:48:02 +0000', ..}, {}, ...]
    """
    symbol = symbol.upper()
    return stock_news.get_yf_rss(symbol)

#### testing functions here

"""
news_data = get_news('AAPL')
print(news_data)
"""
#exit()

"""
# getting one news item at a time
for item in news_data:
    print("-----news---------------")
    print(item['title'])
    print(item['link'])
    print(item['published'])
"""

# Bulk download from Yahoo Finance using yfinance package
#TODO: you can download the bulk of data by a period instead of one day at a time
#TODO: you can use multi-threading to shorten the download time instad of one symbol at a time
def get_bulk_data(symbols):
    print("********Downloading data using yfinance package******")
    tickers = pd.DataFrame(yf.download(tickers=symbols))

    # store data in JSON format
    if not tickers.empty:
        if DEBUG:
            print('Writing tickers information to JSON file ...')
        tickers.to_json(DATA_STORAGE_DIR + 'tickers_hist.json', orient = 'split', compression = 'infer')
    else:
        print('No tickers information ...')
        exit(1)

    print(tickers)
    # return

    # TODO: The section below is for multi-threading but incomplete
    def download(symbol, ticker, securities, invalid_symbols):
        fdata = get_financial_data_from_Yahoo(ticker) # sending ticker object
        if fdata is None:
            invalid_symbols.append(symbol)
        else:
            securities[symbol] = fdata
            print(symbol)

    tickers = yf.Tickers(symbols).tickers # Getting fundamental data for multiple companies and save them to json file
    print("yf.Tickers: ",tickers)
    """
    tasks = dict()

    from threading import Thread
    for symbol in tickers:
        tasks[symbol] = Thread(target=download, args=(symbol, tickers[symbol],))

    for symbol in tasks:
        tasks[symbol].start()

    for symbol in tasks:
        tasks[symbol].join()
    """

# run functions to get and store data
get_market_index_symbols()
get_bulk_data('AAPL, GOOG, META, AMZN, NFLX, NVDA, MSFT, CRWD, AVGO, NOW')
