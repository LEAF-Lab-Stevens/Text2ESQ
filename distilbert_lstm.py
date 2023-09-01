import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
#import seaborn as sns
#import matplotlib.pyplot as plt
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

sentences_splited_into_words = [sentence.split(" ") for sentence in df_train["sentence"]]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", add_prefix_space=True)

class NerDataset(torch.utils.data.Dataset):
  """
  Custom dataset implementation to get (text,labels) tuples
  Inputs:
   - df : dataframe with columns [tags, sentence]
  """
  
  def __init__(self, df):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')
    
    if "tags" not in df.columns or "sentence" not in df.columns:
      raise ValueError("Dataframe should contain 'tags' and 'sentence' columns")

     
    
    tags_list = [i.split() for i in df["tags"].values.tolist()]
    #texts = df["sentence"].values.tolist()
    texts = [re.split('\s+',sentence) for sentence in df["sentence"]]

    self.texts = [tokenizer.batch_encode_plus([text], padding = "max_length", truncation = True, return_tensors = "pt",is_split_into_words=True) for text in texts] #tokenization
    self.labels = [match_tokens_labels(text, tags) for text,tags in zip(self.texts, tags_list)] # match addition token with tags

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    batch_labels = self.labels[idx]

    return batch_text, torch.LongTensor(batch_labels)

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
  
class MetricsTracking():
  """
  In order make the train loop lighter I define this class to track all the metrics that we are going to measure for our model.
  """
  def __init__(self):

    self.total_acc = 0
    self.total_f1 = 0
    self.total_precision = 0
    self.total_recall = 0

  def update(self, predictions, labels , ignore_token = -100):
    '''
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100
    '''  
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    predictions = predictions[labels != ignore_token]
    labels = labels[labels != ignore_token]

    predictions = predictions.to("cpu")
    labels = labels.to("cpu")

    acc = accuracy_score(labels,predictions)
    f1 = f1_score(labels, predictions, average = "macro")
    precision = precision_score(labels, predictions, average = "macro")
    recall = recall_score(labels, predictions, average = "macro")

    self.total_acc  += acc
    self.total_f1 += f1
    self.total_precision += precision
    self.total_recall  += recall

  def return_avg_metrics(self,data_loader_size):
    n = data_loader_size
    metrics = {
        "acc": round(self.total_acc / n ,3), 
        "f1": round(self.total_f1 / n, 3), 
        "precision" : round(self.total_precision / n, 3), 
        "recall": round(self.total_recall / n, 3)
          }
    return metrics   

def train_loop(model, train_dataset, dev_dataset, optimizer,  batch_size, epochs):
  
  train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) #DataLoader class in PyTorch that helps us to load and iterate over elements in a dataset.
  dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  for epoch in range(epochs) : 
    
    train_metrics = MetricsTracking()
    total_loss_train = 0

    model.train() #train mode

    for train_data, train_label in tqdm(train_dataloader):

      train_label = train_label.to(device)
      '''
      squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
      '''
      mask = train_data['attention_mask'].squeeze(1).to(device)
      input_id = train_data['input_ids'].squeeze(1).to(device)

      optimizer.zero_grad() # gradient initialization
      
      output = model(input_id, mask, train_label)
      loss, logits = output[:2]
      predictions = logits.argmax(dim= -1) 

      #compute metrics
      train_metrics.update(predictions, train_label)
      total_loss_train += loss.item()

      #grad step
      loss.backward()
      optimizer.step()
    
    '''
    EVALUATION MODE
    '''            
    model.eval()

    dev_metrics = MetricsTracking()
    total_loss_dev = 0
    
    with torch.no_grad(): # the gradient will not be tracked insided this wrapper
      for dev_data, dev_label in dev_dataloader:

        dev_label = dev_label.to(device)

        mask = dev_data['attention_mask'].squeeze(1).to(device)
        input_id = dev_data['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask, dev_label)
        loss, logits = output[:2]

        predictions = logits.argmax(dim= -1)     

        dev_metrics.update(predictions, dev_label)
        total_loss_dev += loss.item()
    
    train_results = train_metrics.return_avg_metrics(len(train_dataloader))
    dev_results = dev_metrics.return_avg_metrics(len(dev_dataloader))

    print(f"TRAIN \nLoss: {total_loss_train / len(train_dataset)} \nMetrics {train_results}\n" ) 
    print(f"VALIDATION \nLoss {total_loss_dev / len(dev_dataset)} \nMetrics{dev_results}\n" )  


def tags_mapping(tags_series : pd.Series):
  """
  tag_series = df column with tags for each sentence.
  Returns:
    - dictionary mapping tags to indexes (label)
    - dictionary mappign inedexes to tags
    - The label corresponding to tag 'O'
    - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
  """

  if not isinstance(tags_series, pd.Series):
      raise TypeError('Input should be a padas Series')

  unique_tags = set()
  
  for tag_list in df_train["tags"]:
    for tag in tag_list.split():
      unique_tags.add(tag)


  tag2idx = {k:v for v,k in enumerate(sorted(unique_tags))}
  idx2tag = {k:v for v,k in tag2idx.items()}

  unseen_label = tag2idx["O"]

  return tag2idx, idx2tag, unseen_label, unique_tags


def tags_2_labels(tags : str, tag2idx : dict):
  '''
  Method that takes a list of tags and a dictionary mapping and returns a list of labels (associated).
  Used to create the "label" column in df from the "tags" column.
  '''
  return [tag2idx[tag] if tag in tag2idx else unseen_label for tag in tags.split()] 


def match_tokens_labels(tokenized_input, tags, ignore_token = -100):
        '''
        Used in the custom dataset.
        -100 will be the label used to match additional tokens like [CLS] [PAD] that we dont care about. 
        Inputs : 
          - tokenized_input : tokenizer over the imput text -> {input_ids, attention_mask}
          - tags : is a single label array -> [O O O O O O O O O O O O O O B-tim O]
        
        Returns a list of labels that match the tokenized text -> [-100, 3,5,6,-100,...]
        '''

        #gives an array [ None , 0 , 1 ,2 ,... None]. Each index tells the word of reference of the token
        word_ids = tokenized_input.word_ids()
        #print(1)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx != previous_word_idx:
              previous_word_idx = word_idx
              
              
              label = ignore_token if word_idx is None else tag2idx[tags[word_idx]]
              label_ids.append(label)
            elif word_idx is None:
                label_ids.append(ignore_token)

            #if its equal to the previous word we can add the same label id of the provious or -100 
            else :
                label = tags[word_idx]
            # If the label is B-XXX we change it to I-XXX
                if tag2idx[label] % 2 ==1:
                  label_ids.append(tag2idx[label]+1)
                else:
                  label_ids.append(tag2idx[label])


        return label_ids


#create tag-label mapping
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping(df_train["tags"])

#rearrange
tag_list=['O']
for i in list(tag2idx.keys())[:34]:
    tag_list.append('B-'+i.split('-')[1])
    tag_list.append('I-'+i.split('-')[1])
tag2idx = {j:i for i,j in enumerate(tag_list)}

#create the label column from tag. Unseen labels will be tagged as "O"
for df in [df_train, df_dev, df_test]:
  df["labels"] = df["tags"].apply(lambda tags : tags_2_labels(tags, tag2idx))


# load saved parameters and predict
model= DistilbertNER_lstm(len(tag_list))
model.load_state_dict(torch.load('DisBERT-NER_lstm_simple.pt'))
model.eval()