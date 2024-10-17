import os
import yfinance as yf
import pandas as pd
import datetime

# set DEBUG param
DEBUG = True

# startDate, we can adjust as needed
startDate = datetime.datetime(2013, 1, 1)

# endDate, we can adjust as needed
endDate = datetime.datetime.now().strftime('%Y-%m-%d')

# define ticker
tsla = yf.Ticker('TSLA')

# pass the parameters as the taken dates for start and end
hist = pd.DataFrame(tsla.history(start = startDate, end = endDate))
news = pd.DataFrame(tsla.get_news()) # TO DO: Grab other news articles not directly limited to TSLA!
recs = pd.DataFrame(tsla.get_recommendations())

if DEBUG:
    print('Hist:\n {}\n'.format(hist))
    print('News:\n {}\n'.format(news))
    print('Recommendations:\n {}\n'.format(recs))

# store data in JSON format
if not hist.empty:
    if DEBUG:
        print('Writing history information to JSON file ...')
    hist.to_json('tsla_hist.json', orient = 'split', compression = 'infer')
else:
    print('No history information ...')
    exit(1)

if not news.empty:
    if DEBUG:
        print('Writing news information to JSON file ...')
    news.to_json('tsla_news.json', orient = 'split', compression = 'infer')
else:
    print('No news information ...')
    exit(1)

if not recs.empty:
    if DEBUG:
        print('Writing recommendations information to JSON file ...')
    recs.to_json('tsla_recs.json', orient = 'split', compression = 'infer')
else:
    print('No recommendations information ...')
    exit(1)

# read the JSON files
hist = pd.read_json('tsla_hist.json', orient ='split', compression = 'infer')
news = pd.read_json('tsla_news.json', orient ='split', compression = 'infer')
recs = pd.read_json('tsla_recs.json', orient ='split', compression = 'infer')

if DEBUG:
    print('Hist data from JSON file:\n {}\n'.format(hist))
    print('News data from JSON file:\n {}\n'.format(news))
    print('Recommendations data from JSON file:\n {}\n'.format(recs))
