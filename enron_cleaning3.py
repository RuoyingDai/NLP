# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:39:42 2021

@author: cosmo
"""
import pandas as pd
import numpy as np
import multiprocessing
#import PIL 
#import matplotlib.pyplot as plt
#import seaborn as sns
import email
#import Image
import datetime
from dateutil import parser
import csv
import pickle
from nltk.tokenize import RegexpTokenizer

#%% Load data
df = pd.read_csv("D:/BDS/5Hackason/data/emails.csv/emails.csv")

def get_field(field, messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))
    return column

# Extract Message Body
def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column

df['body'] = body(df['message'])
df.head(3)
#%%
new_body = [(row.rpartition('Subject:')[2]).partition('\n\n')[2] for row in df['body']]
df['body'] = new_body
#%%
tokenizer = RegexpTokenizer(r'\w+')
token_set1 = [[word.lower() for word in tokenizer.tokenize(row)] for row in new_body[0:250000]]
#''.join(row)
#%%
token_set2 = [[word.lower() for word in tokenizer.tokenize(row)] for row in new_body[250000:]]
#%%
df['token'] = token_set1 + token_set2


#%%
# Replace empty field with np.nan
def replace_empty_with_nan(subject):
    column = []
    for val in subject:
        if (val == "" or val == list()):
            column.append(np.nan) 
        else:
            column.append(val)
    return column
#%%
df['body'] = replace_empty_with_nan(new_body)
df.dropna(axis=0, inplace=True)
#%%
with open("D:/BDS/5Hackason/data/clean_body_token1.pkl", "wb") as f:
    pickle.dump(df[:250000], f)
with open("D:/BDS/5Hackason/data/clean_body_token2.pkl", "wb") as f:
    pickle.dump(df[250000:], f)