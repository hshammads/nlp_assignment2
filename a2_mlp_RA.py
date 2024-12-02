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
from sklearn.preprocessing import MinMaxScaler

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

        FANG_plus = ["META", "AAPL", "AMZN", "NFLX", "NVDA", "GOOG", "MSFT", "CRWD", "AVGO", "NOW"]

        FANG_plus_dict = {}
        FANG_plus_normalized = {}

        FANG_plus_dict = defaultdict(lambda: defaultdict(list))  # {company: {date: [open values]}}

        for company in FANG_plus:
            ticker = yf.download(company, start=start_date, end=end_date)
            dates = ticker.index
            date_list = [date.strftime('%Y%m%d') for date in dates]
            open_list = ticker['Open'].tolist()
            
            for date, open_value in zip(date_list, open_list):
                FANG_plus_dict[company][date].append(open_value)

        average_open_per_company = {
            company: {date: np.mean(open_values) for date, open_values in dates.items()}
            for company, dates in FANG_plus_dict.items()
        }

        # debug
        # for company, date_averages in average_open_per_company.items():
        #     print(f"Company: {company}")
        #     for date, avg_open in date_averages.items():
        #         print(f"  Date: {date}, Average Open: {avg_open}")
        #     print("-" * 40)

        date_diff_list = []

        for company, date_averages in average_open_per_company.items():
            # sort dates to ensure chronological order
            sorted_dates = sorted(date_averages.keys())
            
            # differences for consecutive dates
            for i in range(1, len(sorted_dates)):
                current_date = sorted_dates[i]
                previous_date = sorted_dates[i - 1]
                diff = date_averages[current_date] - date_averages[previous_date]
                
                date_diff_list.append({
                    "Company": company,
                    "Date": current_date,
                    "Diff": diff
                })

        date_diff_df = pd.DataFrame(date_diff_list)

        '''

        CHANGED BY RYAN 

        Here, we normalize values PER TICKER in a range of [0,1].

        Using MinMaxScaler from Scikit-Learn.

        '''    
        scaler = MinMaxScaler()

        date_diff_df['Normalized'] = date_diff_df.groupby("Company")["Diff"].transform(
            lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()   # reshape to 2d, then 1d
        )

        return date_diff_df
    
    '''
    Function: loadNewsData(train%, val%, test%)

    Input: the % you want to split the data at

    Output: shuffled training_data, val_data, test_data

    data is found in the data directory (different for diff local
    machines & Nautilus), and takes the form

        data = [tuple1, tuple2, ...]

        where tuple = (date, ticker, whole article)
    
    '''

    def loadNewsData():

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
        
        # URGENT: ADD THIS BACK AFTER
        # random.shuffle(all_articles)

        # Splits are based off of training & val split.
        # There will be a test_data, but the param is not used.

        # n_total = len(all_articles)
        # n_train = int(train_split * n_total)
        # n_val = int(val_split * n_total)
        
        # train_data = all_articles[:n_train]
        # val_data = all_articles[n_train:n_train + n_val]
        # test_data = all_articles[n_train + n_val:]

        all_articles_df = pd.DataFrame(all_articles, columns=['Date', 'Company','Article'])
        return all_articles_df

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

    Input: List of tuples (articles)

    Output: removal of stopwords, newlines, and other various symbols
    
    '''

    def cleanArticle(self, article):
        # date = article[0]  
        # ticker = article[1] 
        text = article
        
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

        # print(text)  # Print cleaned article (or return it)
        return text


    '''
    Function: getArticleEmbedding(self, article)

    Input: the whole article from a data point (date, ticker, article) - tuple

    Output: a 784-dimensional embedding of the article

    We will perform embeddings on 512-token chunks of the input (whole article),
    since we want to use all the data, but the tokenizer can only work on 512 tokens
    at a time.
    '''

    def getArticleEmbedding(self, input):
        text = input
        cleaned_text = self.cleanArticle(text)

        # Tokenize the input
        inputs = self.tokenizer(cleaned_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs['input_ids']

        chunk_size = 512
        chunks = [input_ids[:, i:i + chunk_size] for i in range(0, input_ids.size(1), chunk_size)]

        embeddings = []
        for chunk in chunks:
            chunk = chunk.to(self.device)
            outputs = self.model(chunk)
            article_embedding = outputs.last_hidden_state[:, 0, :]  # Based on [CLS] token
            embeddings.append(article_embedding)

        # Average over chunks
        article_embedding = torch.mean(torch.stack(embeddings), dim=0)

        # Convert to numpy for DataFrame compatibility
        return article_embedding.detach().cpu().numpy()


class MLP(nn.Module):

    def __init__(self, articleToEmbedding, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.transformer = articleToEmbedding
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_batches(self, train_data, opens, batch_size=5):

        batch = torch.empty(batch_size, 1024)  # 768 is the embedding dimension
        true_labels = torch.empty(batch_size)

        for batch_idx in range(batch_size):
            while True:
                random_sample = random.choice(train_data)
                date = random_sample[0]
                company = random_sample[1]
                article = random_sample[2]
                
                print(date,company,article)
                break
                # embedding = self.transformer.getArticleEmbedding(article)
                # batch[batch_idx] = embedding
                # break
        #         adjusted_date = Helper.find_next_available_date(date, opens[company].keys())
        #         true_labels[batch_idx] = opens[company][adjusted_date]

        #         break  # Move to the next item in the batch

        # return batch, true_labels
    
    def train_model(self, articles_data, opens_data, epochs=20, batch_size=32, learning_rate=0.001, accuracy_threshold=0.10):
        optimizer = optim.Adam(list(self.parameters()) + list(self.transformer.model.parameters()), lr=learning_rate)
        criterion = nn.MSELoss() 
        
        for epoch in range(epochs):
            self.train()
            
            total_train_loss = 0.0
            total_train_accuracy = 0.0
            num_train_batches = 0
            
            batch_data, true_labels = self.get_batches(articles_data, opens_data, batch_size=batch_size)
            
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":

    articleToEmbedding = Transformer()
    opens_data = Helper.getNormalizedOpens()
    news_data = Helper.loadNewsData()

    # joining opens w/ all data
    news_data_w_norm = news_data.merge(
        opens_data[['Company', 'Date', 'Normalized']], 
        on=['Date', 'Company'], 
        how='left' 
    )

    news_data_w_norm.dropna(inplace=True)

    # OPTIONAL : save as .csv
    # news_data_w_norm.to_csv('news_data_w_norm.csv')

    # OPTIONAL : save as .json 
    # news_data_w_norm.to_json('news_data_w_norm.json')

    """
    Cleaned the data and made a new DataFrame.
    
    news_data_w_norm - {"Date", "Company", "Article", "Normalized"}  

    """

    from tqdm import tqdm
    tqdm.pandas()  # Progress bar for pandas
    news_data_w_norm['Embeddings'] = news_data_w_norm['Article'].progress_apply(
        lambda article: articleToEmbedding.getArticleEmbedding(article)
    )

    news_data_w_norm.to_csv('news_data_w_norm_with_embeddings.csv', index=False)

    # train_df = pd.DataFrame(train_data, columns=['Date', 'Company', 'Article'])
    # val_df = pd.DataFrame(val_data, columns=['Date', 'Company', 'Article'])
    # test_df = pd.DataFrame(test_data, columns=['Date', 'Company', 'Article'])


    # mlp = MLP(articleToEmbedding=articleToEmbedding, input_size=1024, hidden_size=512, output_size=1)
    # mlp.apply(init_weights)
    # mlp.train_model(train_data, opens_data, epochs=100, batch_size=5, learning_rate=0.001, accuracy_threshold=0.30)
