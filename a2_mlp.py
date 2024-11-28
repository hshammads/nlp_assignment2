import yfinance as yf
import datetime
import requests
import pandas as pd
import requests
import os
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

        # for ryan lol
        data_folder = "./data"

        all_data = defaultdict(lambda: defaultdict(list))

        for date_folder in os.listdir(data_folder):
            date_path = os.path.join(data_folder, date_folder)

            if os.path.isdir(date_path):
                json_path = os.path.join(date_path, "news_data.json")
                
                with open(json_path, 'r') as f:
                    data = json.load(f)

                for idx, article in enumerate(data):
                    article = article.get(f"{idx}")
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
    Function: getArticleEmbedding(self, article)

    Input: the whole article from a data point (date, ticker, article)

    Output: a 784-dimensional embedding of the article

    We will perform embeddings on 512-token chunks of the input (whole article),
    since we want to use all the data, but the tokenizer can only work on 512 tokens
    at a time.
    '''

    def getArticleEmbedding(self, input):
        # Tokenize the input
        inputs = self.tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        input_ids = inputs['input_ids']
        
        chunk_size = 512
        chunks = [input_ids[:, i:i + chunk_size] for i in range(0, input_ids.size(1), chunk_size)]

        embeddings = []
        for chunk in chunks:
            chunk = chunk.to(self.device)
            outputs = self.model(chunk)
            article_embedding = outputs.last_hidden_state[:, 0, :] # this goes off of [CLS] token, which identifies the relevant info we want
            embeddings.append(article_embedding) 

        return torch.mean(torch.stack(embeddings), dim=0)



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
    
    '''
    
    FIND A WAY TO HOOK UP TRANSFORMER WITH MLP SO YOU CAN PASS THE LOSS BACK TO TRANSFORMER
    

    '''
    
    
    def get_batches(self, train_data, opens, batch_size=5):

        batch = torch.empty(batch_size, 1024)  # 768 is the embedding dimension
        true_labels = torch.empty(batch_size)

        for batch_idx in range(batch_size):
            while True:
                random_sample = random.choice(train_data)
                date = random_sample[0]
                company = random_sample[1]
                article = random_sample[2]
                
                embedding = self.transformer.getArticleEmbedding(article)
                batch[batch_idx] = embedding

                adjusted_date = Helper.find_next_available_date(date, opens[company].keys())
                true_labels[batch_idx] = opens[company][adjusted_date]

                break  # Move to the next item in the batch

        return batch, true_labels

    def train_model(self, articles_data, opens_data, epochs=20, batch_size=32, learning_rate=0.001, accuracy_threshold=0.10):
        optimizer = optim.Adam(list(self.parameters()) + list(self.transformer.model.parameters()), lr=learning_rate)
        criterion = nn.MSELoss() 
        
        for epoch in range(epochs):
            self.train()
            
            total_train_loss = 0.0
            total_train_accuracy = 0.0
            num_train_batches = 0
            
            batch_data, true_labels = self.get_batches(articles_data, opens_data, batch_size=batch_size)
            
            batch_data = batch_data.to(self.fc1.weight.device)
            true_labels = true_labels.to(self.fc1.weight.device)
            
            predictions = self(batch_data).squeeze()
            true_labels = true_labels.squeeze()

            predictions = predictions.unsqueeze(0) if predictions.dim() == 0 else predictions
            true_labels = true_labels.unsqueeze(0) if true_labels.dim() == 0 else true_labels

            print(predictions)
            print(true_labels)
            
            loss = criterion(predictions, true_labels)

            optimizer.zero_grad()
            loss.backward()

            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient norm for {name}: {param.grad.norm().item()}")

            optimizer.step()

            total_train_loss += loss.item()

            # Calculate accuracy for this batch
            abs_error = torch.abs(predictions - true_labels)
            batch_accuracy = (abs_error <= accuracy_threshold * true_labels).float().mean().item() * 100
            total_train_accuracy += batch_accuracy
            num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches
            avg_train_accuracy = total_train_accuracy / num_train_batches
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}%")

            '''
            This is for the validation step, which we will get to once we figure out training
            
            self.eval()
            with torch.no_grad():  # Disables gradient computation for validation
                val_batch_data, val_true_labels = self.get_batches(val_data, opens_data, batch_size=batch_size)
                val_batch_data = val_batch_data.to(self.fc1.weight.device)
                val_true_labels = val_true_labels.to(self.fc1.weight.device)
                val_predictions = self(val_batch_data).squeeze()
                val_true_labels = val_true_labels.squeeze()
                val_predictions = val_predictions.unsqueeze(0) if val_predictions.dim() == 0 else val_predictions
                val_true_labels = val_true_labels.unsqueeze(0) if val_true_labels.dim() == 0 else val_true_labels
                # print(val_predictions)
                # print(val_true_labels)
                val_loss = criterion(val_predictions, val_true_labels)

                abs_error = torch.abs(val_predictions - val_true_labels) 
                accuracy = (abs_error <= accuracy_threshold * val_true_labels).float().mean().item() * 100

                rmse = torch.sqrt(torch.mean((val_predictions - val_true_labels) ** 2))

                print(f"Validation Loss: {val_loss.item():.4f}, Accuracy (within {accuracy_threshold*100}%): {accuracy:.2f}%, RMSE: {rmse}")
                print("")
                    
                # Free up memory
                del val_batch_data, val_true_labels, val_predictions
                torch.cuda.empty_cache()  
            '''

            del batch_data, true_labels, predictions  # Delete variables to free memory
            torch.cuda.empty_cache() # Free cache for more memory

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":

    articleToEmbedding = Transformer()
    train_data, val_data, test_data = Helper.loadNewsData()
    opens_data = Helper.getNormalizedOpens()
    mlp = MLP(articleToEmbedding=articleToEmbedding, input_size=1024, hidden_size=512, output_size=1)
    mlp.apply(init_weights)
    mlp.train_model(train_data, opens_data, epochs=100, batch_size=5, learning_rate=0.001, accuracy_threshold=0.30)
