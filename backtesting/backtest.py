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

filename = 'data/alpaca_predictions/all.json'
with open(filename) as f:
    data = json.load(f)


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
    down_words = set(['down', 'decrease', 'decreases', 'fall', 'falls', 'decline', 'declines', 'shrink', 'shrinks', 'negatively', 'lower', 'underperform'])
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


def apply_prices_alpaca(info, date, hold_time):

    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M')  # + datetime.timedelta(minutes=1)
    tomorrow = date_obj + datetime.timedelta(minutes=hold_time)

    if date_obj.hour >= 16: #after 4 pm et
        date_obj = date_obj.replace(day=date_obj.day+1, hour=9, minute=30, second=0, microsecond=0)
        tomorrow = date_obj + datetime.timedelta(minutes=hold_time)
        # print("Trade next day")
    if date_obj.hour <= 9 and date_obj.minute <= 30:  # before 9:30 et
        date_obj = date_obj.replace(day=date_obj.day, hour=9, minute=30, second=0, microsecond=0)
        tomorrow = date_obj + datetime.timedelta(minutes=hold_time)
        # print("Trade as soon as nyse opens")

    if tomorrow.hour >= 16: # do not sell after 4 pm et
        tomorrow = tomorrow.replace(hour=15, minute=59)

    for ticker in info.keys():
        # make a request to the alpaca api for the price of the ticker on the date at GET/v2/stocks/{symbol}/trades
        # authenticate with API_KEY and API_SECRET

        print(ticker)
        url = f'https://data.alpaca.markets/v2/stocks/{ticker}/trades'

        # FIRST ACTION (EITHER SHORT OR BUY)
        params = {'start': convert_date(date_obj), 'end': convert_date(date_obj + datetime.timedelta(minutes=30)), 'limit': 10}
        print(params)
        headers = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': API_SECRET}

        response = requests.get(url, params=params, headers=headers)
        response_json = response.json()

        # SECOND ACTION (FINISH SHORT, OR FINISH BUY)
        params2 = {'start': convert_date(tomorrow), 'end': convert_date(tomorrow + datetime.timedelta(minutes=30)), 'limit': 10}
        if info[ticker]['move'] == 'up':
            params2 = {'start': convert_date(tomorrow + datetime.timedelta(minutes=EXTRA_HOLD_IF_UP)),
                       'end': convert_date(tomorrow + datetime.timedelta(minutes=EXTRA_HOLD_IF_UP) + datetime.timedelta(minutes=30)),
                       'limit': 10}
        print(params2)
        response2 = requests.get(url, params=params2, headers=headers)
        response2_json = response2.json()

        if response2_json and response2_json['trades'] and len(response2_json['trades']) > 0 and response_json and response_json['trades'] and len(response_json['trades']) > 0:
            info[ticker]['prices'] = (response_json['trades'][0]['p'], response2.json()['trades'][0]['p'])



def process_article(article, hold_time):
    tickers = get_tickers(article['Prediction'])

    companies = get_company_names(tickers, article['Prediction'])

    info = {ticker: {'company': company, 'ticker': ticker, 'move': None} for ticker, company in zip(tickers, companies) }

    apply_stock_move(info, article['Prediction'])

    apply_prices_alpaca(info, article['date'], hold_time)

    article['price_info'] = info

    return article

if __name__ == '__main__':
    random.shuffle(data)
    output_info = []
    for article in data:
        output_info.append(process_article(article, 60))

    # save output_info to a json file
    with open('data/alpaca_predictions/all_with_price.json', 'w') as f:
        json.dump(output_info, f)
