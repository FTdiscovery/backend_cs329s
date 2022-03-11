import json
import re
import random
import difflib
import requests
import datetime
from pytz import timezone

API_SECRET = '0qbJ1w4qI2nq3HKpDk7uVpjqLq6qgOLTdANH84kE'
API_KEY = 'AKONDXTFSFXNTOZ15B7D'
EXTRA_HOLD_IF_UP = 0  # we hold stock for 30 less minutes if we want to buy a stock (i.e. we think it's going up)

tz = timezone('EST')

# Find where an already identified ticker is in the Stocks Affected ticker
def get_ticker_index(ticker, words):
    for i in range(len(words)):
        if ticker in words[i]:
            return i
    return -1


# get company names preceeding tickers in Stocks Affected
def get_company_names(tickers, s):
    company_names = []
    all_words = s.split()

    for ticker in tickers:
        # Find the capitalized words that come before ticker in the string s
        ticker_index = get_ticker_index(ticker, all_words)

        words = all_words[:ticker_index]

        company_name = []
        for i in range(len(words) - 1, max(-1, len(words) - 3), -1):
            if any([t in words[i] for t in tickers]):
                break
            if words[i][0].isupper():
                company_name.append(words[i])
            else:
                break

        company_names.append(' '.join(reversed(company_name)))
    return company_names


# get tickers of Stocks Affected:
def get_tickers(s):
    # using regex, find stock tickers with patterns such as (XXX)
    tickers = re.findall(r'\(([A-Z]{1,5})\)', s)

    # use regex to find stock tickers with patterns such as (NYSE: VEPCO)
    tickers += re.findall(r'\(NYSE:\s*([A-Z]{1,5})\)', s)

    tickers += re.findall(r'\(NASDAQ:\s*([A-Z]{1,5})\)', s)

    return tickers


# helper function for apply_stock_move
def apply_move(info, mode, move):
    if mode == 'all':
        for ticker in info.keys():
            info[ticker]['move'] = move
    else:
        info[mode]['move'] = move


# adds the stock move to the info dictionary based on the prediction
def apply_stock_move(info, s):

    up_words = set(['up', 'increase', 'increases', 'raise', 'raises', 'rise', 'rises', 'grow', 'grows', 'positive', 'positively', 'benefit', 'higher', 'outperform'])
    down_words = set(['down', 'decrease', 'decreases', 'fall', 'falls', 'decline', 'declines', 'shrink', 'shrinks', 'negatively', 'lower', 'lower', 'underperform'])
    words = s.split()

    mode = 'all'
    for word in words:
        # first set mode to a company ticker if that company is mentioned in the word
        for c_info in info.values():
            if difflib.SequenceMatcher(None, c_info['ticker'], word).ratio() > 0.5 or difflib.SequenceMatcher(None,
                                                                                                              c_info[
                                                                                                                  'company'].lower(),
                                                                                                              word.lower()).ratio() > 0.46:
                mode = c_info['ticker']

        if word.lower() in up_words:
            apply_move(info, mode, 'up')
        elif word.lower() in down_words:
            apply_move(info, mode, 'down')


# converts date to format of alpaca api

def convert_date(date_obj):
    date_obj = tz.localize(date_obj)
    # make the date_obj timezone aware in ET time zone

    # use isoformat to make this date RFC-3339 compliant
    date_str = date_obj.isoformat()
    # url encode the date_str

    return date_str


def process_article(article):
    tickers = get_tickers(article['Prediction'])

    companies = get_company_names(tickers, article['Prediction'])

    info = {ticker: {'company': company, 'ticker': ticker, 'move': None} for ticker, company in zip(tickers, companies) }

    apply_stock_move(info, article['Prediction'])

    tickers = []
    sentiments = []
    for inf in info:
        tickers.append(inf)
        if info[inf]['move'] == 'up':
            sentiments.append(1)
        elif info[inf]['move'] == 'down':
            sentiments.append(-1)
        else:
            sentiments.append(0)

    article['tickers'] = tickers
    article['sentiments'] = sentiments
    article.pop('blurb', None)
    return article

if __name__ == '__main__':

    filename = 'data/alpaca_predictions/2022-03-04.json'
    with open(filename) as f:
        data = json.load(f)

    random.shuffle(data)
    output_info = []
    for article in data:
        output_info.append(process_article(article, 60))

    # save output_info to a json file
    with open('data/alpaca_predictions/2022-03-04_with_price.json', 'w') as f:
        json.dump(output_info, f)
