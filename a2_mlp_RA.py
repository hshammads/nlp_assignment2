import yfinance as yf
import datetime
import requests
import pandas as pd
import requests
import os
import nltk
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
import random

'''
**************************************************************************

Helper class has 3 functions:

    getNormalizedOpens()

    loadNewsData(train%, val%, test%)

    find_next_available_date(date_str, available_dates)

This class helps with the loading of data (news articles, tickers, dates, and opens)
before we pass it into our MLP.

**************************************************************************
'''

# function to fix the JSON file
def fix_json_file(json_path, date):
    with open(json_path, 'r') as file:
        data = file.read()

    # Fix the invalid object separation by replacing '},{"' with ',{'
    fixed_data = data.replace('},{', ',')

    try:
        # Attempt to load the fixed data
        json_data = json.loads(fixed_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} for {date}")
        return None  

    # Save the fixed JSON data back to the file
    with open(json_path, 'w') as file:
        json.dump(json_data, file, indent=4)

    print(f"JSON file '{json_path}' has been fixed.")
    return json_data


class Helper:

    ''' 
    Function: getNormalizedOpens()

    Input: yfinance opens for FANG+ companies

    Output: FANG_plus_normalized dictionary of normalized values

    The current date's value will be (tomorrow's open - today's open) normalized.
    To access a specific value, access the dictionary as such:

        FANG_plus_normalized[company ticker][date] = (tomorrow's open - today's open)
    '''

    def getNormalizedOpens():

        start_date = datetime.datetime(2024, 10, 23)
        end_date = datetime.datetime.now()

        FANG_plus = ["META", "AAPL", "AMZN", "NFLX", "NVDA", "GOOGL", "MSFT", "CRWD", "AVGO", "NOW"]

        FANG_plus_dict = {}
        FANG_plus_normalized = {}

        for company in FANG_plus:

            ticker = yf.download(company, start=start_date, end=end_date)
            dates = ticker.index
            date_list = dates.tolist()
            date_list = [date.strftime('%Y%m%d') for date in dates] # Used to reframe to 'YYYYMMDD'
            open_list = ticker.iloc[:, 4].tolist() # 4 is 'Open' values
            date_diff_dict = {}

            # This sets the current date's value as (tomorrow's open) - (today's open)

            for i in range(len(ticker) - 1):
                date_diff_dict[date_list[i]] = open_list[i+1] - open_list[i]

            FANG_plus_dict[company] = date_diff_dict

        '''
        Here, we normalize values PER TICKER in a range of [-1,1].

        We find all the negative values and normalize them between [-1,0]
        and we find all the positive values and normalize then between [0,1].
        '''    

        for company, date_diff_dict in FANG_plus_dict.items():
            
            diffs = np.array(list(date_diff_dict.values()))
            pos_diffs = diffs[diffs > 0]
            neg_diffs = diffs[diffs < 0]

            if len(pos_diffs) > 0:
                pos_min = pos_diffs.min()
                pos_max = pos_diffs.max()
                normalized_pos = (pos_diffs - pos_min) / (pos_max - pos_min)
            else:
                normalized_pos = np.array([])

            if len(neg_diffs) > 0:
                neg_min = neg_diffs.min()
                neg_max = neg_diffs.max()
                normalized_neg = (neg_diffs - neg_min) / (neg_max - neg_min) - 1
            else:
                normalized_neg = np.array([])

            normalized_vals = []

            for diff in diffs:
                if diff > 0:
                    normalized_vals.append(normalized_pos[pos_diffs == diff][0])
                elif diff < 0:
                    normalized_vals.append(normalized_neg[neg_diffs == diff][0])
                else:
                    normalized_vals.append(0)  
           
            FANG_plus_normalized[company] = dict(zip(date_diff_dict.keys(), normalized_vals))

        return FANG_plus_normalized
    
    '''
    Function: loadNewsData(train%, val%, test%)

    Input: the % you want to split the data at

    Output: shuffled training_data, val_data, test_data

    data is found in the data directory (different for diff local
    machines & Nautilus), and takes the form

        data = [tuple1, tuple2, ...]

        where tuple = (date, ticker, whole article)
    
    '''

    def loadNewsData(train_split=0.85, val_split=0.15, test_split=0):

        # Below only works locally for Matt.
        # data_folder = "c:\\Users\\Matt\\Desktop\\nlp_assignment2\\data"

        # Works locally for Ryan.
        data_folder = ".\data"

        all_data = defaultdict(lambda: defaultdict(list))

        for date_folder in os.listdir(data_folder):
            date_path = os.path.join(data_folder, date_folder)

            if os.path.isdir(date_path):
                json_path = os.path.join(date_path, "news_data.json")

                data = fix_json_file(json_path, json_path)

                for idx, article in data.items():
                    ticker = article.get('ticker')
                    whole_article = article.get('whole_article')
                    all_data[date_folder][ticker].append(whole_article)

        all_articles = []

        for date, tickers in all_data.items():
            for ticker, articles in tickers.items():
                for article in articles:
                    
                    # Somehow, someway, some articles are still 'GOOG'
                    # so we fix that here

                    if ticker == 'GOOG':
                        ticker = 'GOOGL'

                    all_articles.append((date, ticker, article))
        
        random.shuffle(all_articles)

        # Splits are based off of training & val split.
        # There will be a test_data, but the param is not used.

        n_total = len(all_articles)
        n_train = int(train_split * n_total)
        n_val = int(val_split * n_total)
        
        train_data = all_articles[:n_train]
        val_data = all_articles[n_train:n_train + n_val]
        test_data = all_articles[n_train + n_val:]
        
        return train_data, val_data, test_data

    '''
    Function: find_next_available_date(date, all dates)

    Input: current date & all dates

    Output: the next available date (including current)

    We need this because there are no opens on weekends, but 
    there are news articles on weekends. So e.g. for articles on
    Friday, we use the differences in opens between Monday - Friday.
    '''
    
    def find_next_available_date(date_str, available_dates):
        target_date = datetime.datetime.strptime(date_str, "%Y%m%d")
    
        while date_str not in available_dates:
            target_date += datetime.timedelta(days=1)
            date_str = target_date.strftime("%Y%m%d")
    
        return date_str

'''
Transformer class uses a BERT tokenizer and model to convert our whole 
articles into embeddings. These embeddings will be what we pass into our 
MLP to train.
'''

class Transformer:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.model = AutoModel.from_pretrained("bert-large-uncased").to(self.device)

        '''
        Other BERT models I have tried. FinBERT is for finance data, and other is for 
        a bigger model

        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = AutoModel.from_pretrained("allenai/longformer-base-4096").to(self.device)
        '''
    
    '''
    Function: cleanArticle(self, article)

    Input: an article

    Output: removal of stopwords, newlines, and other various symbols
    
    '''

    def cleanArticle(self, article):
        date = article[0]  # Extract date
        ticker = article[1]  # Extract ticker
        text = article[2]  # Extract the article content
        
        # TEXT-CLEANING
        text = text.lower()
        text = text.strip()  # leading/trailing spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        
        # Remove non-alphabetic characters, URLs, or any other custom cleaning
        text = re.sub(r'http\S+', '', text)  # URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # non-alphabetic characters

        # Remove stopwords
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)

        print(text)  # Print cleaned article (or return it)
        return text


    '''
    Function: getArticleEmbedding(self, article)

    Input: the whole article from a data point (date, ticker, article)

    Output: a 784-dimensional embedding of the article

    We will perform embeddings on 512-token chunks of the input (whole article),
    since we want to use all the data, but the tokenizer can only work on 512 tokens
    at a time.
    '''

    def getArticleEmbedding(self, input):
        cleaned_article = self.cleanArticle(input)

        # Tokenize the input
        # inputs = self.tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # input_ids = inputs['input_ids']     # tokens
        # print(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

        # chunk_size = 512
        # chunks = [input_ids[:, i:i + chunk_size] for i in range(0, input_ids.size(1), chunk_size)]

        # embeddings = []
        # for chunk in chunks:
        #     chunk = chunk.to(self.device)
        #     outputs = self.model(chunk)
        #     article_embedding = outputs.last_hidden_state[:, 0, :] # this goes off of [CLS] token, which identifies the relevant info we want
        #     embeddings.append(article_embedding) 

        # # print("Articles are embedded.")
        # # print(embeddings)

        # return torch.mean(torch.stack(embeddings), dim=0)


if __name__ == "__main__":

    articleToEmbedding = Transformer()
    train_data, val_data, test_data = Helper.loadNewsData()
    opens_data = Helper.getNormalizedOpens()

    # train data - list of tuples
    first_entry = train_data[0]
    embedding = articleToEmbedding.getArticleEmbedding(first_entry)

    # batches = 
    
    # for batch in batches:
    #     embeddings = articleToEmbedding.getArticleEmbedding(batch)

    # mlp = MLP(articleToEmbedding=articleToEmbedding, input_size=1024, hidden_size=512, output_size=1)
    # mlp.apply(init_weights)
    # mlp.train_model(train_data, opens_data, epochs=100, batch_size=5, learning_rate=0.001, accuracy_threshold=0.30)
