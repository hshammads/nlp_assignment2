from yahoo_fin import stock_info, news as stock_news
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import date
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import enum
import os
import pathlib
import json

cwd = os.getcwd()

# startDate, we can adjust as needed
startDate = date(2013, 1, 1)
# endDate, we can adjust as needed
endDate = date.today()

DEBUG = True
TODAY = endDate.strftime('%Y%m%d')
DATA_STORAGE_DIR = cwd + '/data/' + TODAY + '/'
TICKERS = ['META','AAPL','AMZN','NFLX','NVDA','GOOG','MSFT','CRWD','AVGO','NOW']
TICKERS_STRING = ", ".join(TICKERS)

# make directory for today's data
pathlib.Path(DATA_STORAGE_DIR).mkdir(parents = True, exist_ok = True)

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

# Bulk download from Yahoo Finance using yfinance package
#TODO: you can use multi-threading to shorten the download time instead of one symbol at a time
def get_bulk_data(symbols):
    print("********Downloading data using yfinance package******")
    tickers = pd.DataFrame(yf.download(tickers=symbols, start = startDate, end = endDate))

    # store data in JSON format
    if not tickers.empty:
        if DEBUG:
            print('Writing tickers information to JSON file ...')
        tickers.to_json(DATA_STORAGE_DIR + 'tickers_hist.json', orient = 'split', compression = 'infer')
    else:
        print('No tickers information ...')
        exit(1)

    if DEBUG:
        print('Bulk data tickers info: ',tickers)

# run functions to get and store price data
get_market_index_symbols()
get_bulk_data(TICKERS_STRING)

# Get news from Yahoo finance
def get_news(symbol):
    """
    #:param symbol:
    #:return: [{'summary':, .., 'link': url, 'published': 'Wed, 24 Nov 2021 20:48:02 +0000', ..}, {}, ...]
    """
    symbol = symbol.upper()
    return stock_news.get_yf_rss(symbol)

# getting one news item at a time, by Ryan
session = HTMLSession()
def extract_a_tags(element, base_url):
    urls = []

    if element is not None:
        if hasattr(element, 'name') and element.name == 'a' and 'href' in element.attrs:
            href = element['href']
            absolute_url = urljoin(base_url, href)
            urls.append(absolute_url)

        if hasattr(element, 'contents'):
            for child in element.contents:
                urls.extend(extract_a_tags(child, base_url))

    return urls

def get_info(link):
    r = session.get(link)
    #print(r)

    # div - 'body yf-5ef8bf'
    div_body = r.html.find('div.body.yf-5ef8bf', first=True)

    # byline-attr-time-style
    by_line = r.html.find('div.byline.yf-1k5w6kz', first=True)

    # p - yf-1pe5jgt
    date = ""
    text = []

    if div_body:
        paragraphs = div_body.find('p.yf-1pe5jgt')

        # ensure there are enough paragraphs to extract first and (potentially) last
        if len(paragraphs) >= 5:
            first_paragraph = paragraphs[0].text

            # dynamically find a valid last paragraph if external links may appear
            last_paragraph = paragraphs[-4].text if len(paragraphs) >= 5 else paragraphs[-1].text

            # return text in a list
            text = [first_paragraph, last_paragraph]
            # return text
        else:
            return None

    if by_line:
        time_element = by_line.find('time', first=True)
        if time_element:
            # extract the datetime attribute
            datetime_str = time_element.attrs.get('datetime')
            if datetime_str:
                # split the string to get only the date (YYYY-MM-DD)
                date = datetime_str.split('T')[0]
                # print(date)
                # return date

    if div_body:
        return [date,link,text,div_body.text]

news_data = [get_news(ticker_name) for ticker_name in TICKERS]

info_dict = dict()
c = 0
for idx, item in enumerate(news_data):
    if item:
        for article in item:
            if DEBUG:
                print('Starting news data grab for link {} ...'.format(article['link']))
            data = get_info(link=article['link'])
            if data:
                info = {
                    "ticker": TICKERS[idx],  # Get the ticker symbol from the list
                    "date_published" : data[0],
                    "title": item[idx]["title"],
                    "summary": item[idx]["summary"],
                    "link": data[1],
                    "first_p": data[2][0],
                    "last_p": data[2][1],
                    "whole_article": data[3]
                }
                info_dict[str(c)] = info
                c += 1

# print(info_dict)
with open(DATA_STORAGE_DIR + '/news_data.json', 'a') as json_file:
    if info_dict:
        json.dump(info_dict, json_file, indent = 5)