import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from transformers import AutoTokenizer
import pdb


class StockModel(nn.Module):
    def __init__(self):
        super(StockModel, self).__init__()
        model_name = "bert-base-chinese"

        article_number = 20
        hidden_dim = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedModel = BertModel.from_pretrained(model_name)
        self.attentionweight = nn.Linear(
            article_number * 768, article_number, bias=False)

        self.LSTMdecoder = nn.LSTM(
            768 + 4, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, 4)

    def forward(self, words, stock_info):
        '''
        words should be a list of sentences
        input: 
            words: a list of articles (batch size, sequence_length, nums of article, words of article)
            stock_info: stock information (batch size, 4)
        output:
        '''
        pt_batch = self.tokenizer(
            words,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        batch_size = words.size(0)
        pt_outputs = self.embedModel(**pt_batch).last_hidden_state
        # (N, nums of article, words of article, 768)
        pt_outputs = torch.mean(pt_outputs, dim=1)  # (N, nums of article, 768)

        # here I am not sure if you want the last_hidden_state or the pooler_output
        # TODO: add a attention to mearge articles -> (batch size, 768)
        attention_weight = F.softmax(
            self.attentionweight(pt_outputs.view(batch_size, -1), dim=1))
        pt_outputs = torch.bmm(pt_outputs.permute(
            0, 2, 1), attention_weight.unsqueeze(2))

        pt_outputs = torch.cat(
            (pt_outputs.squeeze(), stock_info), dim=-1)  # (N, 768 + 4)
        # TODO: here comes the problem
        # decoder -> add LSTM
        # 先只拿前一天預測下一天的
        pt_outputs, _ = self.LSTMdecoder(pt_outputs)

        pt_outputs = self.hidden2tag(pt_outputs.view(batch_size, -1))

        return pt_outputs
