import pymongo  # package for working with MongoDB
import pdb
import sys, os

# To import relative folder
sys.path.append(os.path.join(sys.path[0], '..', 'stock_db'))

from mongodb import *
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
total_request_page_number = 100

class news_crawler(object):

    def __init__(self, company="台積電"):

    	# Specified company's name
        self.url = 'https://tw.finance.yahoo.com/news_search.html?ei=Big5&q=' + quote(company.encode('big5'))

        # Store articles' hyperlink
        self.href = []

        # Store articles' content
        self.content = []
        self.date    = []

    def _get_news_title_links(self, url):

        # Get request from website
        list_req = requests.get(url)
        soup = BeautifulSoup(list_req.content, "html.parser")
        allnews = soup.find('table', {'id': 'newListContainer'})
        alllinks = allnews.find_all('a')
        
        href = []
        for i in range(1, len(alllinks), 2):
            link = allnews.find_all('a')[i]['href']
            if link == '#':
                continue
            href.append(link)

        return href

    def _get_total_news_links(self):

        # while loop its page number
        pg = 1
        while True:
            print('page: {}'.format(pg))
            url = self.url + '&pg=' + str(pg)
            pg_href = self._get_news_title_links(url)
            self.href += pg_href
            
            if pg == total_request_page_number:
                break
            pg += 1 

    def _get_news_content(self, url):
        req = requests.get(url)
        soup = BeautifulSoup(req.content, 'html.parser')
        pp = soup.find_all('p')
        text = [p.text for p in pp]
        return ''.join(text), soup.find('time').text

    def get_total_news_content(self):
        self._get_total_news_links()
        for h in self.href:
            if not h.startswith('https'):
                continue
            else:
                print('request articles from: {}'.format(h))
                # content and date of the article
                c, d = self._get_news_content(h)
                self.content.append(c)
                self.date.append(d)
                time.sleep(0.5)

    def _store_to_db(self):
        store_data_list = []
        
        for c, d in zip(self.content, self.date):
            
            if NEWS.find({"date": d.split()[0]}).count() == 0: # no repeat date
                data = {"date": d.split()[0], "content": c}
                store_data_list.append(data)
            
            else:
                print("This date has stored in collection: {}".format(d.split()[0]))
                continue
        if store_data_list:
            x = NEWS.insert_many(store_data_list)



if __name__ == '__main__':
    cc = news_crawler()
    cc.get_total_news_content()
    cc._store_to_db()

