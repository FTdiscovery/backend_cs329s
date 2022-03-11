from newsapi import NewsApiClient
from datetime import date, timedelta, datetime
from alpaca_trade_api import REST, Stream
import pandas as pd

def scrape_articles_today_newsapi():
    entire_json = []
    newsapi = NewsApiClient(api_key='b9eeb5b9a3ce423ab927527d1904fe14')
    a = newsapi.get_top_headlines(category='business', page_size=100, country='us')
    for elem in a['articles']:
        if elem['description'] != None and elem['content'] != None:
            json = {}
            json['source'] = elem['source']['name']
            json['link'] = elem['url']
            json['title'] = elem['title']
            a = (datetime.strptime(elem['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=5)).strftime(
                "%Y-%m-%d %H:%M")
            json['date'] = a
            json['summary'] = elem['description']
            if elem['content'] != None:
                if elem['content'].partition('.')[0] != elem['content']:  # no full stop
                    json['summary'] = elem['description'] + ' ' + elem['content'].partition('.')[0] + '.'
            if datetime.strptime(elem['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=5) >= datetime.today().replace(hour=0, minute=0, second=0, microsecond=0):
                entire_json.append(json)

    a = newsapi.get_top_headlines(category='technology', page_size=100, country='us')
    for elem in a['articles']:
        if elem['description'] != None and elem['content'] != None:
            json = {}
            json['source'] = elem['source']['name']
            json['link'] = elem['url']
            json['title'] = elem['title']
            a = (datetime.strptime(elem['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=5)).strftime(
                "%Y-%m-%d %H:%M")
            json['date'] = a
            json['summary'] = elem['description']
            if elem['content'] != None:
                if elem['content'].partition('.')[0] != elem['content']:  # no full stop
                    json['summary'] = elem['description'] + ' ' + elem['content'].partition('.')[0] + '.'

            if datetime.strptime(elem['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') - timedelta(
                    hours=5) >= datetime.today().replace(hour=0, minute=0, second=0, microsecond=0):
                entire_json.append(json)

    a = newsapi.get_top_headlines(category='science', page_size=100, country='us')
    for elem in a['articles']:
        if elem['description'] != None and elem['content'] != None:
            json = {}
            json['source'] = elem['source']['name']
            json['link'] = elem['url']
            json['title'] = elem['title']
            a = (datetime.strptime(elem['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=5)).strftime(
                "%Y-%m-%d %H:%M")
            json['date'] = a
            json['summary'] = elem['description']
            if elem['content'] != None:
                if elem['content'].partition('.')[0] != elem['content']:  # no full stop
                    json['summary'] = elem['description'] + ' ' + elem['content'].partition('.')[0] + '.'

            if datetime.strptime(elem['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') - timedelta(
                    hours=5) >= datetime.today().replace(hour=0, minute=0, second=0, microsecond=0):
                # Science is full of not very good articles so add filters:
                if json['source'] != 'ScienceAlert' and json['source'] != 'Phys.Org' and json['source'] != 'Space.com':
                    entire_json.append(json)

    return entire_json

# import http.client, urllib.parse
#
# conn = http.client.HTTPSConnection('api.marketaux.com')
#
# params = urllib.parse.urlencode({
#     'api_token': 'XhZ3a3v5mXLCeluKm8dHA4lzlyyUvk0h0IdGLSG6',
#     'symbols': 'AMZN',
#     'limit': 50,
#     })
#
# print(params)
#
# conn.request('GET', '/v1/news/all?{}'.format(params))
#
# res = conn.getresponse()
# data = res.read()
# dict = json.loads(data.decode('utf-8'))['data']
# for elem in dict:
#     json = {}
#     json['source'] = elem['source']
#     json['link'] = elem['url']
#     json['title'] = elem['title']
#
#     dscr = ""
#     for h in elem['entities']:
#         for highlight in h['highlights']:
#             if highlight['highlighted_in'] == 'main_text' and highlight['sentiment'] > 0.25:
#                 dscr[highlight['highlight']] = highlight['sentiment']
#     print(dscr)
#
#     a = (datetime.strptime(elem['published_at'], '%Y-%m-%dT%H:%M:%S.000000Z') - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
#     json['date'] = a
#     print(json)
#
#     #print(elem)

def scrape_articles_alpaca(date):
    # df = pd.read_csv('data/tickers/NYSE.csv')
    # df2 = pd.read_csv('data/tickers/NASDAQ.csv')
    #
    # tickers = dict(zip(df.Symbol, df.Name)) | dict(zip(df2.Symbol, df2.Name))
    # relevant_tickers = list(tickers.keys())  # this has all stocks... may be a bit slow

    API_SECRET = '0qbJ1w4qI2nq3HKpDk7uVpjqLq6qgOLTdANH84kE'
    API_KEY = 'AKONDXTFSFXNTOZ15B7D'

    entire_json = []

    relevant_tickers = ['AAPL', 'FB', 'GOOG', 'GOOGL', 'JPM', 'MSFT',
                        'SPY', 'TSLA', 'DELL', 'HPE', 'NVDA', 'CSCO', 'CRM',
                        'ORCL', 'ADBE', 'NFLX', 'PFE', 'LU', 'GE', 'TWX',
                        'XOM', 'AMD', 'GRAB', 'F', 'C', 'BAC', 'OXY', 'UBER', 'T', 'INTC', 'PLTR',
                        'X', 'VZ', 'AMC', 'WFC', 'SNAP', 'BABA', 'SPCE', 'GM', 'TWTR', 'TME', 'KO',
                        'DAL', 'HOOD', 'LUMN', 'MS', 'CS', 'AI', 'BILI', 'PINS', 'NET', 'TSM', 'V',
                        'KEY', 'IQ', 'KHC', 'WMT', 'RTX', 'G', 'UAA', 'MCFE', 'DB', 'NKE', 'GSK',
                        'USB', 'SHEL', 'UBS', 'MRNA', 'JNJ', 'TDOC', 'ETSY', 'MELI', 'SE', 'ISRG',
                        'BRK-A', 'AMZN', 'COF', 'EBAY', 'AIG', 'MWD'
                        ]

    news_urls = set()
    rest_client = REST(API_KEY, API_SECRET)
    for ticker in relevant_tickers:
        try:
            news = rest_client.get_news(ticker, date, date)
            for elem in news:
                #print(elem)
                json = {}
                json['source'] = elem.source
                json['link'] = elem.url
                json['title'] = elem.headline
                a = (datetime.strptime(str(elem.updated_at), '%Y-%m-%d %H:%M:%S+00:00') - timedelta(hours=5)).strftime(
                    "%Y-%m-%d %H:%M")
                json['date'] = a
                json['summary'] = elem.summary
                json['tickers'] = elem.symbols
                json['sentiments'] = [0 for i in range(len(json['tickers']))]

                if json['link'] not in news_urls and json['summary'] != "" and len(json['summary']) > 10 and len(json['tickers']) < 7:
                    entire_json.append(json)
                    news_urls.add(json['link'])

            print(ticker)
            print(len(entire_json))
        except:
            continue

    return entire_json


def keep_wsj_article(article):

    category_keywords = ["Business", "WSJ News Exclusive", "Markets",
                         "Heard on the Street"]  # ['Business', 'Economy', 'Heard on the Street', 'Markets', 'Tech', 'Stocks', 'WSJ News Exclusive', 'CFO Journal', 'Pro Bankruptcy', 'CIO Journal']
    # not included: opinion, life & work, World, US, Asia
    # Old keywords: keywords = ['Business', 'Economy', 'Heard on the Street', 'Markets', 'Tech', 'Stocks', 'Opinion', 'Life & Work']

    banned_categories = ['Journal Reports: Small Business']
    category_urls = ['https://www.wsj.com/news/types/deals-deal-makers?mod=breadcrumb',
                     'https://www.wsj.com/news/heard-on-the-street?mod=breadcrumb',
                     'https://www.wsj.com/news/types/today-s-markets?mod=breadcrumb',
                     'https://www.wsj.com/news/markets?mod=breadcrumb',
                     'https://www.wsj.com/news/business?mod=breadcrumb',
                     'https://www.wsj.com/news/types/technology?mod=bigtop-breadcrumb',
                     'https://www.wsj.com/news/types/cfo-journal?mod=breadcrumb'
                     ]
    # category_urls = ['https://www.wsj.com/news/types/deals-deal-makers?mod=breadcrumb',
    #                  'https://www.wsj.com/news/heard-on-the-street?mod=breadcrumb',
    #                  'https://www.wsj.com/news/types/today-s-markets?mod=breadcrumb',
    #                  'https://www.wsj.com/news/markets?mod=breadcrumb',
    #                  'https://www.wsj.com/news/business?mod=breadcrumb',
    #                  'https://www.wsj.com/news/life-work/automotive?mod=breadcrumb',
    #                  'https://www.wsj.com/news/types/rumble-seat?mod=breadcrumb',
    #                  'https://www.wsj.com/news/types/technology?mod=bigtop-breadcrumb',
    #                  'https://www.wsj.com/news/types/cfo-journal?mod=breadcrumb']

    if 'categories' in article:
        if any([ck in article['categories'][0] for ck in banned_categories]):
            return False
        if any([ck in article['categories'][0] for ck in category_keywords]):
            return True
    if 'category_urls' in article:
        if any([cu in article['category_urls'] for cu in category_urls]):
            return True
    return False

import json

if __name__ == "__main__":
    date_str = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')

    articles = scrape_articles_alpaca(date_str)

    file_name = f"data/alpaca/{date_str}.json"
    with open(file_name, "w") as f:
        json.dump(articles, f)

