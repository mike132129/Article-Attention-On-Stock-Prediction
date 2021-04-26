from stock_db.mongodb import NewsDatabase
from crawler.stock import stock_crawler
from torch.utils.data import Dataset
import torch
from pdb import set_trace as st
import pdb
import numpy as np


def chi_to_eng(s):
    return s.replace('年', '-').replace('月', '-').replace('日', '')


class news_dataset(Dataset):
    def __init__(self, tokenizer, company='tsmc'):

        self.tokenizer = tokenizer

        db = NewsDatabase(company)
        data = db.get_data_by_sorted_date()
        self.date, self.text = data.date.tolist(), data.content.tolist()
        self.date = [chi_to_eng(s) for s in self.date]
        self.stock_price = stock_crawler(
            start_time=self.date[0], end_time=self.date[-1]).get_value()

        self.index =
        self.input_ids = self.__tokenization(self.text)

    def __len__(self):
    return len(self.landmarks_frame)

    def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_name = os.path.join(self.root_dir,
                            self.landmarks_frame.iloc[idx, 0])
    image = io.imread(img_name)
    landmarks = self.landmarks_frame.iloc[idx, 1:]
    landmarks = np.array([landmarks])
    landmarks = landmarks.astype('float').reshape(-1, 2)
    sample = {'image': image, 'landmarks': landmarks}

    if self.transform:
        sample = self.transform(sample)

    return sample

    def __tokenization(self, text):
        pt_batch = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return pt_batch


if __name__ == '__main__':
    from transformers import BertTokenizer
    model_version = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_version)
    ds = news_dataset(tokenizer)
    pdb.set_trace()
