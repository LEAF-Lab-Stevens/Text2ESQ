#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 200)


# In[9]:


import csv
import pandas as pd

df = pd.read_json('bt_small.json')

df.to_csv('bt_small.txt', sep='\t', index=False)
print(df)


# In[10]:


# function to read raw text file
def read_text(filename):
        # open the file
        file = open(filename, mode='rt', encoding='utf-8')
        
        # read all text
        text = file.read()
        file.close()
        return text


# In[11]:


# split a text into sentences
def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents


# In[12]:


data = read_text("bt_small.txt")
deu_eng = to_lines(data)
deu_eng = array(deu_eng)
deu_eng


# In[17]:


deu_eng[2,2]


# In[18]:


# convert text to lowercase
for i in range(1,len(deu_eng)):
    deu_eng[i,0] = deu_eng[i,0].lower()
    deu_eng[i,2] = deu_eng[i,1].lower()


# In[19]:


# empty lists
eng_l = []
deu_l = []

# populate the lists with sentence lengths
for i in deu_eng[:,0]:
      eng_l.append(len(i.split()))

for i in deu_eng[:,1]:
      deu_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'deu':deu_l})

length_df.hist(bins = 30)
plt.show()


# In[20]:


# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer(filters='')
    #tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[22]:


# prepare english tokenizer
eng_tokenizer = tokenization(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 24
print('English Vocabulary Size: %d' % eng_vocab_size)
eng_tokenizer


# In[23]:


# prepare Deutch tokenizer
deu_tokenizer = tokenization(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1

deu_length = 24
print('Deutch Vocabulary Size: %d' % deu_vocab_size)


# In[24]:


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


# In[25]:


from sklearn.model_selection import train_test_split

# split data into train and test set
train, test = train_test_split(deu_eng, test_size=0.2, random_state = 12)


# In[26]:


# prepare training data
trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])


# In[27]:


# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model


# In[28]:


# model compilation
model = define_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)


# In[16]:


rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')


# In[251]:


filename = 'model.h1.24_jan_19'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
print(trainY.shape)

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=500, batch_size=512, validation_split = 0.2,callbacks=[checkpoint], 
                    verbose=1)


# In[252]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()


# In[123]:


import numpy as np
model = load_model('model.h1.24_jan_19')
preds = np.argmax(model.predict(testX.reshape((testX.shape[0],testX.shape[1]))),axis = -1)
preds


# In[124]:


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


# In[125]:


# convert predictions into text (English)
preds_text = []
for i in preds:
  temp = []
  for j in range(len(i)):
    t = get_word(i[j], eng_tokenizer)
    if j>0:
      if(t == get_word(i[j-1], eng_tokenizer)) or (t == None):
        temp.append('')
      else:
        temp.append(t)
    else:
      if(t == None):
        temp.append('')
      else:
        temp.append(t)

  preds_text.append(' '.join(temp))


# In[167]:


pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
# print 15 rows randomly
pred_df.head(15)


# In[158]:


data = read_text("logic form1.txt")
eng[10][0].split(" ")



# In[159]:



str1=''.join(preds_text[1])
str2=str1.split(" ")

str2


# In[173]:


a=[]
for i in range(len(eng)): 
    tokens=''.join(eng[i][0]).split(" ")
    if tokens!=' ':  
        if tokens!='':
            a.append(tokens)
b=[]
for i in range(len(preds_text)): 
    tokens=''.join(preds_text[i]).split(" ")
    if tokens!=' ':  
        if tokens!='':
            b.append(tokens)

print(a)


# In[172]:


from nltk.translate.bleu_score import sentence_bleu
references = a
candidates = b
for candidate in candidates:
    print(sentence_bleu(references, candidate))


# In[ ]:


rouge_1 = keras_nlp.metrics.RougeN(order=1)
rouge_2 = keras_nlp.metrics.RougeN(order=2)

for test_pair in test_pairs[:30]:
    input_sentence = test_pair[0]
    reference_sentence = test_pair[1]

    translated_sentence = decode_sequences(tf.constant([input_sentence]))
    translated_sentence = translated_sentence.numpy()[0].decode("utf-8")
    translated_sentence = (
        translated_sentence.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )

    rouge_1(reference_sentence, translated_sentence)
    rouge_2(reference_sentence, translated_sentence)

print("ROUGE-1 Score: ", rouge_1.result())
print("ROUGE-2 Score: ", rouge_2.result())

