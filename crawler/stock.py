from datetime import datetime
import numpy as np
import pandas as pd
import requests
import json
import pdb

class stock_crawler(object):

	'''
	please initialize the class with
	1. stock id: ****.TW (reference from yahoo stock)
	2. start date with format (yyyy-mm-dd)
	3. end date (yyyy-mm-dd)
	'''
	def __init__(self, stock_id="2330.TW", start_time='2020-01-02', end_time='2020-01-20'):
		
		# search time using second after 1970's
		# time is datetime format
		self.start_time = start_time + ' 01:00:00'
		self.end_time = end_time + ' 01:00:01'

		self.start_time = int((datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds())
		self.end_time   = int((datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds())

		# id is four digits and .TW string
		self.stock_id = stock_id
		
		# Use yahoo stock api
		self.api_url = "https://query1.finance.yahoo.com/v8/finance/chart/" + str(self.stock_id) + "?" + "period1=" + \
						str(self.start_time) + "&period2=" + str(self.end_time) + "&interval=1d&events=history&=hP2rOschxO0"

	def _get_response(self):
		return requests.get(self.api_url).text

	def get_value(self) -> pd.core.frame.DataFrame:
		value = self._get_response()
		value = json.loads(value)
		
		stock_df = pd.DataFrame(value['chart']['result'][0]['indicators']['quote'][0], \
							index=pd.to_datetime(np.array(value['chart']['result'][0]['timestamp'])*1000*1000*1000))

		return stock_df


if __name__ == '__main__':
	crawler = stock_crawler()
	df = crawler.get_value()
	print('result:\n', df)
