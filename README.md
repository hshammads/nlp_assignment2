# NLP Class Assignment 2
This is assignment 2 for CPSC 488: Natural Language Processing, taught by Dr. Christopher Ryu, at California State University, Fullerton (CSUF).  
  Collaborators: Hammad Sheikh, Matthew Do, Ryan Avancena.  

# Assignment details, as provided by Dr. Ryu.

## Objectives
In this exercise, you will learn how to develop a sentiment analysis model and an an automated trading system relying on the analysis results.
$\color{blue}\text{Only Python programs written using Python 3.0 or higher will be accepted}$. $\color{red}\text{NO Jupyter notebook or any other Python variants}$. will be accepted for efficient grading.


## About the dataset and problem to solve
A stock, in investment terms, represents ownership in a company. When you buy stocks, you essentially buy a small piece of that company. The stock price for a company is mainly determined by the current value of the company that can be computed by the company’s past financial data (e.g., earnings, revenues, profits, etc.) and future value determined by the company’s future performance, which is unknown. The intrinsic value of a company refers to the actual and inherent worth of the company based on both current and future value. Therefore, a stock price is determined by the current value and estimated (perceived) future value of the company. Investors buy stocks based on the company’s estimated intrinsic value. That’s why the stock prices inherently fluctuate depending on the market situation and the company’s future prospectus, which are typically published in news articles. The more information investors have, the better they can estimate the intrinsic value.
**Tesla** may be one of the companies that generate the most news articles due to their market impact and technological innovations. In this assignment, you will analyze the Tesla stock (the stock symbol, **TSLA**) price data and related news articles to develop a simple trading system. Your trading system will be equipped with a sentiment analysis model that analyzes all the (directly or indirectly) Tesla-related news articles for about 10 years, computes its market impact, and trades only TSLA based on the impact. For example, your trading system will buy a certain number of shares if the news article is considered positive and the balance is enough to cover the trade. Otherwise, the system will sell a certain number of shares if the news article is negative and a sufficient number of shares are already owned. The \textcolor{green}{ultimate goal} is maximizing the \textcolor{green}{return} from the initial investment, utilizing the sentiment analysis model.

## Required activities
\textcolor{red}{**Utilize** the **GPU** resources available on the **Nautilus** through the **Kubernetes**} for modeling (**NOT** by the JupyterHub) and \textcolor{red}{**write** an analysis report} about your system's modeling results and trading performance by answering all the questions below with your **\textcolor{red}{justification} supported \textcolor{red}{by the data}**.

1. **Download** the historical price data for TSLA and its (directly or indirectly) related news articles between 1/1/2013 to the last market closing date (today or yesterday) from any free data sources (e.g., Yahoo Finance) and save the data in a JSON file format on your local machine. JSON is an open standard file format used for data storage or exchange. The price data should include market date, open price, high price, low price, closing price, and volume. The news articles may be related to the entire market, not necessarily only to TSLA. The direct TSLA-related news can be articles about the company that you can download from the data source. Examples of indirect TSLA-related news can be the news about the market (Dow Jones \^DJI, S\&P \^GSPC, Nasdaq \^IXIC, or the industry Tesla belongs to). Briefly describe the types of news articles you downloaded for your system, explaining why and how you downloaded the data.

2. **Develop** a sentiment analysis model using Multilayer Perceptron (MLP) that can quantitatively estimate the impact of each news article on the TSLA price. Briefly describe
		\subitem (a) The methods used to (pre)process the news and price data for your MLP algorithm with justification and
		\subitem (b) the method(s) used to analyze the news and quantify its impact that will be used for trade.

3. **Backtest** your trading system based on the sentiment analysis model developed in (2) and measure its performance. Backtesting means testing the effectiveness of trading systems that utilize specific trading algorithms. In this case, your sentiment analysis model is a trading algorithm. For the backtesting, assume your account has an initial investment balance of \$100,000, and buy/sell orders will always be filled without trading fees.

	To evaluate the performance of your trading algorithm, you need to **develop a simple trading system** that will buy a certain number of TSLA shares only if the balance is sufficient to cover the purchase and the positive market impact computed by the model or sell a certain number of shares only if it already holds enough number of shares (\textcolor{red}{no short sell allowed}) and the negative market impact.
	**Evaluate your system’s trading performance** \underline{so far} by calculating the following simple metrics\\
		\subitem (a) \$gain or \$loss for each trade,
		\subitem (b) the total $gain or $loss for all trades,
		\subitem (c) \% return compared to the initial balance.

	**Log every trade**, including the key transaction data such as\\
		\subitem (a) the transaction date,
		\subitem (b) trading type buy/sell,
		\subitem (c) \# of shares traded,
		\subitem (d) \$amount used for the trade, and
		\subitem (e) the current balance after the trade to a log file \enquote{\textcolor{purple}{trade\_log.json}} for future analysis, verification, or accounting purpose.

	**Display the trading summary**, including the
		\subitem (a) total \$gain or \$loss for all trades and
		\subitem (b) \% return compared to the initial balance (\$100,000.00).

4. **Briefly describe** at least two methods or techniques (based on the relevant topics discussed in class) to improve the model performance and evaluation results on whether or not those methods improved the trading performance.

5. **Create word embeddings** based on Word2vec, other embedding method, or pre-trained embedding for your model and compare the trading performance with the best model without relying on word embeddings.

\textcolor{red}{**Warning:**} Although you can reuse any source codes available on the Internet, you are not allowed to share your codes with any other team or students in this class. Any student or team violating this policy will receive a **ZERO** score for this assignment, potentially for all the remaining assignments.

## What to submit
1.  \textcolor{blue}{One analysis report} includes \textcolor{blue}{all your member names}, \% contribution made by each member, and all the answers to the questions based on your analysis in **PDF** or **Word format**. If every member contributed equally, simply state \enquote{\textcolor{purple}{equal contribution}.} If your team does not agree on individual contributions, briefly write a task description for each member. Different grades may be assigned based on individual contributions, even if a group completed the work.

2. **Upload one analysis report file** and **Python program file(s)**, individually. Please \textcolor{red}{DO NOT upload any zip file} since Canvas cannot open it.

3. When you show some example data in your analysis report (when necessary), select only a few examples, not including the entire dataset.

4. Submit only one for each team.

## Grading criteria
1. The overall quality of work shown in the report about modeling results, supporting data, analysis process, methods used, and correctly implemented programs
2. The level of understanding as reflected in the report
3. Effort (10\%)
