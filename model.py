import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from transformers import AutoTokenizer
import pdb

class StockModel(nn.Module):
    def __init__(self):
        super(StockModel, self).__init__()
        model_name = "bert-base-chinese"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedModel = BertModel.from_pretrained(model_name)

        self.LSTMdecoder = nn.LSTM(768, 2, 2, batch_first=True, bidirectional=True)

    def forward(self, words, stock_info):
        '''
        words should be a list of sentences
        '''
        pt_batch = self.tokenizer(
            words,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        pt_outputs = self.embedModel(**pt_batch).last_hidden_state # (N, seq_len, 768)
        pt_outputs = torch.mean(pt_outputs, dim=1) # (N, 768)
        # here I am not sure if you want the last_hidden_state or the pooler_output
        pt_outputs = torch.cat((pt_outputs, stock_info), dim=-1) # (N, 768 + 4)
        # TODO: here comes the problem 
        # if we average them all, how should we still have temporal properties when decoder part
        print(pt_outputs)