import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from tqdm import tqdm
#from itertools import groupby
#import json
#import re

import transformers
from transformers import AutoTokenizer,AutoModel
from transformers import  DistilBertForTokenClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score

class DistilbertNER_lstm(nn.Module):
  """
  Implement NN class based on distilbert pretrained from Hugging face.
  Inputs : 
    tokens_dim : int specifyng the dimension of the classifier
  """
  
  def __init__(self, tokens_dim):
    super(DistilbertNER_lstm,self).__init__()
    
    if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

    if tokens_dim <= 0:
          raise ValueError('Classification layer dimension should be at least 1')

    self.base_model = AutoModel.from_pretrained('distilbert-base-uncased')
    self.dropout = nn.Dropout(0.1, inplace=False)
    self.lstm = nn.LSTM(input_size = 768, hidden_size = tokens_dim, num_layers = 2, bidirectional = True,batch_first=True) # output features from bert is 768 and 2 is ur number of labels
    self.classifier = nn.Linear(2*tokens_dim, tokens_dim)
    self.loss_ = nn.CrossEntropyLoss()
    self.num_labels = tokens_dim
    

  def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss
    """
  Forwad computation of the network
  Input:
    - inputs_ids : from model tokenizer
    - attention :  mask from model tokenizer
    - labels : if given the model is able to return the loss value
  """
    out = self.base_model(input_ids = input_ids, attention_mask = attention_mask)
    out = self.dropout(out[0])
    out, (hidden,cell) = self.lstm(out)
    logits = self.classifier(out)
    #inference time no labels
    outputs = (logits,)
    if labels is not None:
      loss_fct = self.loss_
      # Only keep active parts of the loss
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        loss = loss_fct(active_logits, active_labels)
      else:
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    outputs = (loss,) + outputs

    return outputs