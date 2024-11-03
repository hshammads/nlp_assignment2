# NLP Class Assignment 2
This is assignment 2 for CPSC 488: Natural Language Processing, taught by Dr. Christopher Ryu, at California State University, Fullerton (CSUF).  
  Collaborators: Hammad Sheikh, Matthew Do, Ryan Avancena.  

# Assignment details, as provided by Dr. Ryu;

## Objectives
In this exercise, you will learn how to develop a sentiment analysis model and an an automated trading system relying on the analysis results.
$\color{blue}\text{Only Python programs written using Python 3.0 or higher will be accepted}$. $\color{red}\text{NO Jupyter notebook or any other Python variants}$. will be accepted for efficient grading.  
**Change: get news articles for at least 4 weeks for all the members of ETF FNGU**

## About the dataset and problem to solve
A stock, in investment terms, represents ownership in a company. When you buy stocks, you essentially buy a small piece of that company. The stock price for a company is mainly determined by the current value of the company that can be computed by the company’s past financial data (e.g., earnings, revenues, profits, etc.) and future value determined by the company’s future performance, which is unknown. The intrinsic value of a company refers to the actual and inherent worth of the company based on both current and future value. Therefore, a stock price is determined by the current value and estimated (perceived) future value of the company. Investors buy stocks based on the company’s estimated intrinsic value. That’s why the stock prices inherently fluctuate depending on the market situation and the company’s future prospectus, which are typically published in news articles. The more information investors have, the better they can estimate the intrinsic value.  

An Exchange-Traded Fund (**ETF**) is a type of investment fund that holds a collection of assets such as stocks, bonds, commodities, or a mix of these. There are many ETFs covering specific sections of the market currently being traded in the market. Some ETFs are conservative (slow price change) and some are highly aggressive or volatile in price change. **FNGU** is one of the ETF funds that concentrates on the top 10 popular technology companies, also known as **FANG+**, which include META, AAPL, AMZN, NFLX, NVDA, GOOGL, MSFT, CRWD, AVGO, and [NOW](https://finance.yahoo.com/quote/FNGU/holdings/). The fund manager occasionally changes the holdings. FNGU is highly volatile due to high leverage (3x). The price of FNGU change when the price of any of these holdings change. As a pair of FNGU, **FNGD** follows the inverse of FNGU for those who want to bet against FNGU. If most of the holdings are up, FNGU will go up 3x, but FNGD will go down 3x, and vice versa. In other words, the price movement of FNGU and FNGD is inverse, so you can buy and sell either one for profit in either direction as long as you can correctly predict the direction without relying on a sell-short strategy. Due to the popularity, the size, and their technological innovations of all those companies in FNGU, they most likely generate lots of news articles that impact the price of FNGU as well as the market indices.  
In this assignment, you will analyze the FNGU price data and related news articles to develop a simple trading system. Your trading system will be equipped with a sentiment analysis model that analyzes all the (directly or indirectly) FNGU-related news articles for $\color{red}\text{at least four weeks}$, computes its market impact, and trades only FNGU, FNGD, or both FNGU and FNGD based on the impact. For example, your trading system will buy a certain number of shares if the news article is considered positive and the balance is enough to cover the trade. Otherwise, the system will sell a certain number of shares if the news article is negative and a sufficient number of shares are already owned. The $\color{green}\text{ultimate goal}$ is maximizing the $\color{green}\text{return}$ from the initial investment, utilizing the sentiment analysis model.

## Required activities
**$\color{red}\text{Utilize}$** $\color{red}\text{the}$ **$\color{red}\text{GPU}$** $\color{red}\text{resources available on the}$ **$\color{red}\text{Nautilus}$** $\color{red}\text{through the}$ **$\color{red}\text{Kubernetes}$** for modeling (**NOT** by the JupyterHub) and **$\color{red}\text{write}$** $\color{red}\text{an analysis report}$ about your system's modeling results and trading performance by answering all the questions below with your **$\color{blue}\text{justification}$ supported $\color{blue}\text{by the data}$**.

1. **Download** the historical price data for FNGU (or FNGD or both) and its (directly or indirectly) related news articles for your selected period of four weeks from any free data sources (e.g., Yahoo Finance) and save the data in a JSON file format on your local machine. JSON is an open standard file format used for data storage or exchange. The price data should include market date, open price, high price, low price, closing price, and volume. The news articles may be related to the entire market, not necessarily only to FANG+. The direct FANG+-related news can be articles about the companies that you can download from the data source. Examples of indirect FANG+-related news can be the news about the market (Dow Jones \scalebox{.8}{\textsuperscript{$\wedge$}}DJI, S\&P \scalebox{.8}{\textsuperscript{$\wedge$}}GSPC, Nasdaq \scalebox{.8}{\textsuperscript{$\wedge$}}IXIC, or the industry FANG+ belongs to). Briefly describe the types of news articles you downloaded for your system, explaining why and how you downloaded the data.

2. **Develop** a sentiment analysis model using Multilayer Perceptron (MLP) that can quantitatively estimate the impact of each news article on the TSLA price. Briefly describe
		\subitem (a) The methods used to (pre)process the news and price data for your MLP algorithm with justification and
		\subitem (b) the method(s) used to analyze the news and quantify its impact that will be used for trade.

3. **Backtest** your trading system based on the sentiment analysis model developed in (2) and measure its performance. Backtesting means testing the effectiveness of trading systems that utilize specific trading algorithms. In this case, your sentiment analysis model is a trading algorithm. For the backtesting, assume your account has an initial investment balance of \$100,000, and buy/sell orders will always be filled without trading fees.

  To evaluate the performance of your trading algorithm, you need to **develop a simple trading system** that will buy a certain number of FNGU shares only if the balance is sufficient to cover the purchase and the positive market impact computed by the model or sell a certain number of shares only if it already holds enough number of shares ($\color{red}\text{no short sell allowed}$) and the negative market impact.  
  **Evaluate your system’s trading performance** *so far* by calculating the following simple metrics  
  - (a) \$gain or \$loss for each trade,
  - (b) the total $gain or $loss for all trades,
  - (c) \% return compared to the initial balance.  

  **Log every trade**, including the key transaction data such as  
  - (a) the transaction date,
  - (b) trading type buy/sell,
  - (c) \# of shares traded,
  - (d) \$amount used for the trade, and
  - (e) the current balance after the trade to a log file $\color{purple}\text{"trade}_\text{log.json"}$ for future analysis, verification, or accounting purpose.  

  **Display the trading summary**, including the  
  - (a) total \$gain or \$loss for all trades and
  - (b) \% return compared to the initial balance (\$100,000.00).

4. **Briefly describe** at least two methods or techniques (based on the relevant topics discussed in class) to improve the model performance and evaluation results on whether or not those methods improved the trading performance.

5. **Create word embeddings** based on Word2vec, other embedding method, or pre-trained embedding for your model and compare the trading performance with the best model without relying on word embeddings.

**$\color{red}\text{Warning:}$** Although you can reuse any source codes available on the Internet, you are not allowed to share your codes with any other team or students in this class. Any student or team violating this policy will receive a **ZERO** score for this assignment, potentially for all the remaining assignments.

## What to submit
1.  $\color{blue}\text{One analysis report}$ includes $\color{blue}\text{all your member names}$, \% contribution made by each member, and all the answers to the questions based on your analysis in **PDF** or **Word format**. If every member contributed equally, simply state $\color{purple}\text{"equal contribution"}$. If your team does not agree on individual contributions, briefly write a task description for each member. Different grades may be assigned based on individual contributions, even if a group completed the work.

2. **Upload one analysis report file** and **Python program file(s)**, individually. Please $\color{red}\text{DO NOT upload any zip file}$ since Canvas cannot open it.

3. When you show some example data in your analysis report (when necessary), select only a few examples, not including the entire dataset.

4. Submit only one for each team.

## Grading criteria
1. The overall quality of work shown in the report about modeling results, supporting data, analysis process, methods used, and correctly implemented programs
2. The level of understanding as reflected in the report
3. Effort (10\%)
