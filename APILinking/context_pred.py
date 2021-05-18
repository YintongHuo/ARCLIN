from transformers import BertTokenizer, BertModel
import torch
import random
import os, re
import json, sys

class BERTContext(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        #self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(768, 300)
        self.last_layer = torch.nn.Linear(300,5)

    def forward(self, sentence):
        sentence_representation = self.bert(sentence.unsqueeze(0))[0]
        sent_rep = sentence_representation[:,0,:]
        prediction = self.last_layer(self.second_last_layer(sent_rep))
        return prediction


class ContextPredictor():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # current_model = BERTContext.from_pretrained('bert-base-uncased')
        self.current_model = torch.load('ctx_model.pkl', map_location=torch.device('cpu'))
        self.current_model.eval()
        self.device = torch.device("cpu")
        self.current_model.to(self.device)
        self.category = ['pytorch', 'np', 'pd', 'tf', 'matplotlib']
        # optimizer = torch.optim.SGD(current_model.parameters(), lr=0.001)
        # loss_func = torch.nn.CrossEntropyLoss()
    def predict(self, sentence):
        input_sentence = '[SEP]' + sentence + '[CLS]'
        tokens = torch.tensor(self.tokenizer.encode(input_sentence)[:512]).to(self.device)
        output = self.current_model(tokens)
        pred = torch.nn.functional.softmax(output)
        category = self.category[torch.argmax(pred, dim=1).item()]
        #print(sentence, pred)
        return category



