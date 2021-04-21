import pymongo
import pandas as pd
from pdb import set_trace as st

client = pymongo.MongoClient("mongodb://localhost:27017/")
DB = client["newsdb"]

# Collections
NEWS = DB["tsmc"]

class NewsDatabase(object):
	def __init__(self, company=None):
		self.data = []
		if company:
			self.db_col = DB[company]
		else:
			print('PLEASE SPECIFY COMPANY NAME')
		self.df = None
	
	def get_data_by_sorted_date(self):
		news = self.db_col.find().sort('date')
		self.df = pd.DataFrame(list(news))
		return self.df

if __name__ == '__main__':

	import sys, os
	sys.path.append(os.path.join(sys.path[0], '..'))
	from config import COMPANY
	cc = NewsDatabase(COMPANY)
	cc.get_data_by_sorted_date()