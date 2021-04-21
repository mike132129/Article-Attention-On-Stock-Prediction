from stock_db.mongodb import NewsDatabase
from torch.utils.data import Dataset
import torch
from pdb import set_trace as st

class news_dataset(Dataset):
	def __init__(self, tokenizer, company='tsmc'):

		self.tokenizer = tokenizer

		db = NewsDatabase(company)
		data = db.get_data_by_sorted_date()
		date, text = data.date.tolist(), data.content.tolist()
		input_ids = self.__tokenization(text)


	def __tokenization(self, text):
		# TODO:
		pass

if __name__ == '__main__':
	from transformers import BertTokenizer
	model_version = 'bert-base-chinese'
	tokenizer = BertTokenizer.from_pretrained(model_version)
	ds = news_dataset(tokenizer)
