import glob
import random
from bs4 import BeautifulSoup
import json

def extract_blurbs(html):
    #extract all paragraphs using beautiful soup
    soup = BeautifulSoup(html, 'html.parser')


    #Find all divs with class 'c<<int>>' in soup
    divs = soup.find_all('div', class_=lambda x: x in ['c6', 'c7'])
    
    #For each div, concatenate all text that is inside of a p
    blurbs = []
    for div in divs:
        paragraphs = soup.find_all('p')
        blurbs.append(' '.join([p.text for p in paragraphs]))


    #return the text of each paragraph in a list
    return [p.text for p in paragraphs]

def process_file(filename):
    #read the json file
    with open(filename, 'r') as f:
        data = json.load(f)
        html = data['html']

        blurbs = extract_blurbs(html)
        #print the article title 
        print(data['title'])
        print(data['date'])
        print(data['url'])
        print("*********************************************************")
        for blurb in blurbs:
            print(blurb)
            print("*********************************************************")
        

        

def main():
    #get all files in data/brew_articles_raw
    files = glob.glob('data/brew_articles_raw/*.json')
    #randomly select one file
    filename = random.choice(files)
    process_file(filename)



if __name__ == '__main__':
    main()