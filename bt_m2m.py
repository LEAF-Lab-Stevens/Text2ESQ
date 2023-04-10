import pandas as pd
import json
import bt
import random
import torch
import os
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

f = open('medium_query.json',) 
data = json.load(f)
df = pd.DataFrame.from_dict(data)
df = df.drop(columns=['query', 'token'])
df = df.dropna()

m2m100 = bt.m2m100_bt()
out_zh, out_en = m2m100.back_translate(target_lang='zh', text=df.question)
bt_out = {'en': list(df.question),'zh': list(out_zh.values()), 'zh_en': list(out_en.values())}

f = open("nl_22_medium.json","w")
json.dump(bt_out, f)
f.close()


