import yfinance as yf
import datetime
import json
import os
import random
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

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

    def getNormalizedOpens(self):

        start_date = datetime.datetime(2024, 10, 23)
        end_date = datetime.datetime.now()

        FANG_plus = ["META", "AAPL", "AMZN", "NFLX", "NVDA", "GOOGL", "MSFT", "CRWD", "AVGO", "NOW"]

        FANG_plus_dict = {}

        for company in FANG_plus:

            ticker = yf.download(company, start=start_date, end=end_date)
            dates = ticker.index
            date_list = dates.tolist()
            date_list = [date.strftime('%Y%m%d') for date in dates] # Used to reframe to 'YYYYMMDD'
            open_list = ticker.iloc[:, 4].tolist() # 4 is 'Open' values
            date_diff_dict = {}

            # This sets the current date's value as (tomorrow's open - today's open) / today's open.
            # Signifies the % of increase from the previous day, and NOT the absolute value of increase itself  

            for i in range(len(ticker) - 1):
                date_diff_dict[date_list[i]] = open_list[i+1] - open_list[i]

            FANG_plus_dict[company] = date_diff_dict


        return FANG_plus_dict
    
    '''
    Function: loadNewsData(train%, val%, test%)

    Input: the % you want to split the data at

    Output: shuffled training_data, val_data, test_data

    data is found in the data directory (different for diff local
    machines & Nautilus), and takes the form

        data = [tuple1, tuple2, ...]

        where tuple = (date, ticker, whole article)
    
    '''

    def loadNewsData(self, train_split=0.85, val_split=0.15, test_split=0):
        data_folder = "./data"
        all_data = defaultdict(lambda: defaultdict(list))

        for date_folder in os.listdir(data_folder):
            date_path = os.path.join(data_folder, date_folder)

            if os.path.isdir(date_path):
                json_path = os.path.join(date_path, "news_data.json")
                
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print(f"Error reading JSON file {json_path}: {e}")
                    continue  # Skip this file if there's an error

                for idx, article in data.items():
                    if article is None:
                        print(f"Skipping None article at index {idx}")
                        continue  # Skip if article is None

                    ticker = article.get('ticker')
                    whole_article = article.get('whole_article')

                    if ticker and whole_article:
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
    
    def find_next_available_date(self, date_str, available_dates):
        target_date = datetime.datetime.strptime(date_str, "%Y%m%d")
    
        while date_str not in available_dates:
            target_date += datetime.timedelta(days=1)
            date_str = target_date.strftime("%Y%m%d")
    
        return date_str
    
class CustomDataset(Dataset):
    def __init__(self, data, normalized_opens, tokenizer, max_len=512):
        self.data = data
        self.normalized_opens = normalized_opens
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        date, ticker, article = self.data[idx]
        numeric_feature = self.normalized_opens.get(ticker, {}).get(date, 0)
        encoding = self.tokenizer(
            article,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "numeric_feature": torch.tensor(numeric_feature, dtype=torch.float),
        }
    
class BertMLPModel(nn.Module):
    def __init__(self, bert_model_name="bert-large-uncased", numeric_input_size=1):
        super(BertMLPModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze all BERT layers
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + numeric_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Single output for regression
        )

    def forward(self, input_ids, attention_mask, numeric_feature):
        with torch.no_grad():  # Disable gradient computation for BERT
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_output.pooler_output  # [CLS] token embeddings
        combined_input = torch.cat((pooled_output, numeric_feature.unsqueeze(1)), dim=1)
        return self.mlp(combined_input)
    
def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_feature = batch["numeric_feature"].to(device)

            outputs = model(input_ids, attention_mask, numeric_feature)
            loss = loss_fn(outputs.squeeze(), numeric_feature)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train_model_with_early_stopping(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    loss_fn, 
    device, 
    num_epochs=20, 
    patience=3
):
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_feature = batch["numeric_feature"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numeric_feature)
            loss = loss_fn(outputs.squeeze(), numeric_feature)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        val_loss = validate_model(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pt")
            print("Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load("best_model.pt"))
    print("Training complete. Best model restored.")


if __name__ == "__main__":
    helper = Helper()
    normalized_opens = helper.getNormalizedOpens()
    train_data, val_data, _ = helper.loadNewsData()

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    train_dataset = CustomDataset(train_data, normalized_opens, tokenizer)
    val_dataset = CustomDataset(val_data, normalized_opens, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertMLPModel().to(device)
    optimizer = optim.AdamW(model.mlp.parameters(), lr=1e-5)  # Only update MLP parameters
    loss_fn = nn.MSELoss()

    train_model_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=2000, 
        patience=3      
    )

    print("Training complete!")