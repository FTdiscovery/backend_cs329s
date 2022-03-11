import requests
import json
import datetime
from multiprocessing import Pool
from bs4 import BeautifulSoup


"""
Makes the same request as this curl request using the requests library

curl 'https://singularity.morningbrew.com/graphql' \
  -H 'authority: singularity.morningbrew.com' \
  -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"' \
  -H 'accept: */*' \
  -H 'content-type: application/json' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'origin: https://www.morningbrew.com' \
  -H 'sec-fetch-site: same-site' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-dest: empty' \
  -H 'referer: https://www.morningbrew.com/' \
  -H 'accept-language: en-US,en;q=0.9' \
  --data-raw $'{"operationName":"IssuesSearch","variables":{"query":"","limit":10,"offset":10,"newsletter":"daily"},"query":"query IssuesSearch($query: String\u0021, $newsletter: String\u0021, $limit: Int\u0021, $offset: Int\u0021) {\\n  allArchiveIssues(\\n    query: $query\\n    newsletter: $newsletter\\n    limit: $limit\\n    offset: $offset\\n  ) {\\n    title\\n    subjectLine\\n    slug\\n    date\\n    __typename\\n  }\\n}\\n"}' \
  --compressed
"""
def get_all_articles_with_request():
    url = 'https://singularity.morningbrew.com/graphql'
    headers = {
        'authority': 'singularity.morningbrew.com',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"',
        'accept': '*/*',
        'content-type': 'application/json',
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
        'sec-ch-ua-platform': '"macOS"',
        'origin': 'https://www.morningbrew.com',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.morningbrew.com/',
        'accept-language': 'en-US,en;q=0.9'
    }
    data = '{"operationName":"IssuesSearch","variables":{"query":"","limit":10000,"offset":0,"newsletter":"daily"},"query":"query IssuesSearch($query: String\u0021, $newsletter: String\u0021, $limit: Int\u0021, $offset: Int\u0021) {\\n  allArchiveIssues(\\n    query: $query\\n    newsletter: $newsletter\\n    limit: $limit\\n    offset: $offset\\n  ) {\\n    title\\n    subjectLine\\n    slug\\n    date\\n    __typename\\n  }\\n}\\n"}'
    response = requests.post(url, headers=headers, data=data)
    return response.json()



#check if data/all_articles.json exists, if so then read it. Otherwise, call get_list_of_articles
def get_all_articles():
    try:
        with open('data/all_articles.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        data_j =  get_all_articles_with_request()
        # save data to file
        with open('data/all_articles.json', 'w') as f:
            f.write(json.dumps(data_j))


def get_article_with_request(article):
    url = f"https://www.morningbrew.com/daily/issues/{article['slug']}"
    response = requests.get(url)
    date_str = article['date']
    #write the repsonse to a file
    #date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f%z')

    output_json = {
        'title': article['subjectLine'],
        'date': date_str,
        'url': url,
        'html': response.text
    }
    with open(f'data/brew_articles_raw/{date_str}.json', 'w') as f:
        f.write(json.dumps(output_json))
    


def main():
    
    data = get_all_articles()
    
    p = Pool(10)
    p.map(get_article_with_request, data['data']['allArchiveIssues'])
    p.terminate()
    p.join()
    

if __name__ == '__main__':
    main()