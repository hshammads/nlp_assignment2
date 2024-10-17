import os
import yfinance as yf

tsla = yf.Ticker("TSLA")

# get all stock info
tsla.info

# get historical market data
hist = tsla.history(period="1mo")

# show meta information about the history (requires history() to be called first)
tsla.history_metadata

# show actions (dividends, splits, capital gains)
tsla.actions
tsla.dividends
tsla.splits
tsla.capital_gains  # only for mutual funds & etfs

# show share count
tsla.get_shares_full(start="2022-01-01", end=None)

# show financials:
tsla.calendar
tsla.sec_filings
# - income statement
tsla.income_stmt
tsla.quarterly_income_stmt
# - balance sheet
tsla.balance_sheet
tsla.quarterly_balance_sheet
# - cash flow statement
tsla.cashflow
tsla.quarterly_cashflow
# see `Ticker.get_income_stmt()` for more options

# show holders
tsla.major_holders
tsla.institutional_holders
tsla.mutualfund_holders
tsla.insider_transactions
tsla.insider_purchases
tsla.insider_roster_holders

tsla.sustainability

# show recommendations
tsla.recommendations
tsla.recommendations_summary
tsla.upgrades_downgrades

# show analysts data
tsla.analyst_price_targets
tsla.earnings_estimate
tsla.revenue_estimate
tsla.earnings_history
tsla.eps_trend
tsla.eps_revisions
tsla.growth_estimates

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# Note: If more are needed use tsla.get_earnings_dates(limit=XX) with increased limit argument.
tsla.earnings_dates

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
tsla.isin

# show options expirations
tsla.options

# show news
tsla.news

# get option chain for specific expiration
opt = tsla.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts
