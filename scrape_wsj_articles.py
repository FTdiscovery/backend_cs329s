"""
Scrapes articles.
"""

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta, datetime
import random
import re
from multiprocessing import Pool
import json
import os
import pyuser_agent
from scrape_utils import keep_wsj_article

def get_headers():
    return {'User-Agent': pyuser_agent.UA().random}

def remove_unicode(string):
    return  (string.replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u2015", "-")
                .replace("\u2017", "_")
                .replace("\u2018", "\'")
                .replace("\u2019", "\'")
                .replace("\u201a", ",")
                .replace("\u201b", "\'")
                .replace("\u201c", "\"")
                .replace("\u201d", "\"")
                .replace("\u201e", "\"")
                .replace("\u2026", "...")
                .replace("\u2032", "\'")
                .replace("\u2033", "\""))

# Remove all extra spaces
def remove_spaces(string):
    return remove_unicode(" ".join(string.split()))

# Gets list of date urls to process
def get_dates():

    start_date = date(2022,2,20)
    end_date = date.today()
    delta = end_date - start_date 

    dates = [start_date + timedelta(days=tdelta) for tdelta in range(delta.days + 1)]
    #Create list of URLS of form https://www.wsj.com/news/archive/2021/01/09 based on dates
    return dates

def scrape_article(url):
    soup = BeautifulSoup(requests.get(url, headers=get_headers(), timeout=10 ).content.decode('utf-8', 'ignore'), 'html.parser')
    
    #select the first article in soup
    article = soup.select_one('article')
    #iterate through all elements with class "article-breadCrumb"
    if article is None:
        return {}

    return_obj = {}
    if article.h2 is not None and article.p is not None:
        return_obj["summary"] = remove_spaces(article.h2.text) + ". " + remove_spaces(article.p.text)

    breadcrumb_soup = article.select('.article-breadCrumb')
    if breadcrumb_soup is not None:
        return_obj["categories"] = [remove_spaces(breadcrumb.text) for breadcrumb in breadcrumb_soup],
        return_obj["category_links"] = [breadcrumb.find('a')['href'] for breadcrumb in breadcrumb_soup if breadcrumb.find('a') if not None]
    return return_obj

def scrape_day(date):
    #convert date to string
    date_str = date.strftime('%Y-%m-%d')
    file_name = f"data/wsj/{date_str}.json"
    #if the file exists, skip
    if os.path.isfile(file_name):
        with open(file_name) as f:
            print(date_str, "already scraped with", len(json.load(f)), "articles.")
        return

    url = f"https://www.wsj.com/news/archive/{date.year}/{date.month}/{date.day}"

    articles = []
    
    #create regex object for strings that start with ".WSJTHeme--timestamp--"
    time_regex = re.compile(r'^\.WSJTHeme\-\-timestamp\-\-') 

    relevant_headlines = ["Business", "Finance"] #"Markets", "Finance", "Financial", "Earnings"]
    for i in range(1, 6): # pages 1-5, will not return error if not seen
        soup = BeautifulSoup(requests.get(url+f"?page={i}", headers=get_headers(), timeout=10).content.decode('utf-8', 'ignore'), 'html.parser')

        for article in soup.select('article'):
            try:
                raw_time = remove_spaces(article.select_one("div[class*=timestamp]").text)
                raw_date = f"{date.year}-{date.month}-{date.day} {raw_time}"
                article_obj = {
                    "source": 'wsj',
                    "link": article.select_one('a').get('href'),
                    "title": remove_spaces(article.h2.text),
                    #create date from format 01/01/2020 12:00 AM ET
                    'date': datetime.strptime(raw_date, '%Y-%m-%d %I:%M %p ET').strftime("%Y-%m-%d %H:%M")
                }
                article_obj.update(scrape_article(article_obj["link"]))
                if keep_wsj_article(article_obj):
                    del article_obj['categories']
                    del article_obj['category_links']
                    articles.append(article_obj)

            except Exception as e:
                print(e)
                print(f"Failed to scrape article from {url} page {i}")

    print("Articles for",date_str,":", len(articles))


    with open(file_name, "w") as f:
        json.dump(articles, f)

def main():
    p = Pool(10)

    dates = get_dates()

    print(dates)
    p.map(scrape_day, dates)
    p.terminate()
    p.join()


if __name__ == '__main__':
    main()