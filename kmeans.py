# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:17:49 2021

@author: Ana-Maria, Ruoying
"""

import pickle
import pandas as pd
import re
import pysentiment2 as ps
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Position
#position = pd.read_excel("D:/BDS/5Hackason/data/enron_position.xlsx", header = None)
#position.columns = ['name', 'position']

#%% read the data1
#with open("D:/BDS/5Hackason/data/data1.pkl", "rb") as f:
with open("D:/BDS/5Hackason/data/clean_body1.pkl", "rb") as f:
    df = pickle.load(f)
df = df[df['body'].notna()]
#array = [p[]for name in df.employee]

#%% read the data2
with open("D:/BDS/5Hackason/data/clean_body2.pkl", "rb") as f:
#with open("D:/BDS/5Hackason/data/data2.pkl", "rb") as f:
    df = pickle.load(f)
df = df[df['body'].notna()]
#%%
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
body = list(df.body)
token_set = [[word.lower() for word in tokenizer.tokenize(row)] for row in body]
df['token'] = token_set
#%% Four + Two lists of words
file1 = open("D:/BDS/5Hackason/data/actionVerb.txt","r",encoding="utf-8")
action=[]
for line in file1: 
    action.append(line.replace('\n',"").lower())   
    #negative.append(line)
file1.close()

file2 = open("D:/BDS/5Hackason/data/negative-words.txt","r",encoding="utf-8")
negative=[]
for line in file2: 
    negative.append(line.replace('\n',""))   
    #negative.append(line)
file2.close()


# first-person pronouns
person1 = ['I', "i", 'Me', 'My', "me", "mine", "Mine", "my", "me"]
# exclusive words
exclusive = ['but', 'except', 'without']
# work regulatory word
work = ['electricity', 'gas', 'water']
# spam word
# from 
# https://journeys.autopilotapp.com/blog/email-spam-trigger-words/
spam = ['subscription', '.com','subsribe','urgent','instant',
        '100%', 'bonus', 'free', 'bargain', 'prize','deal','unlimited',
        'access','boss','cancel','cheap','certified','cheap',
        'compare','congratulations', 'cures', 'friend', 'guarantee',
        'guaranteed','hello', 'offer', 'opportunity', 'winner',
        'winning', 'won', 'amazing', 'billion', 'cash', 'earn',
        'extra', 'home', 'lose', 'income', 'vacation', 'addresses',
        'beneficiary', 'billiing', 'casino', 'celebrity','hidden',
        'investment', 'junk', 'legal', 'loan','lottery','medicine',
        'miracle', 'money', 'nigerian', 'offshore', 'passwords',
        'refinance', 'request', 'rolex', 'score', 'spam','unsolicited',
        'valium', 'viagra', 'vivodin', 'warranty','xanax']
# Sentiment
lm = ps.LM()
list_scores = []
tokens = [lm.tokenize(email) for email in body]
for i in range(0,len(df)):
    #tokens = lm.tokenize(body[i])
    score = lm.get_score(tokens[i])
    list_scores.append(score)
# Count
person1_list1 = []
exclusive_list1 = []
negative_list1 = []
action_list1 = []
length_list1 = []
work_list = []
spam_list = []
for i in range(0,len(df)):
    #words_in_phrase = re.findall(r'\w+', df.token.iloc[i])
    words_in_phrase = token_set[i]  
    person1_occurences = 0 + len(set(person1).intersection(words_in_phrase))
    exclusive_occurences = 0 + len(set(exclusive).intersection(words_in_phrase))
    negative_occurences = 0 + len(set(negative).intersection(words_in_phrase))
    action_occurences = 0 + len(set(action).intersection(words_in_phrase))
    # added
    work_occurences = 0 + len(set(work).intersection(words_in_phrase))
    spam_occurences = 0 + len(set(spam).intersection(words_in_phrase))
    person1_list1.append(person1_occurences)
    exclusive_list1.append(exclusive_occurences)
    negative_list1.append(negative_occurences)
    action_list1.append(action_occurences)
    length_list1.append(len(words_in_phrase))
    # added
    work_list.append(work_occurences)
    spam_list.append(spam_occurences)
    

# Combine count and sentiment
df1_sentiment = pd.DataFrame.from_dict(list_scores) 
#df1_sentiment.assign(person1 = person1_list1, exclusive = exclusive_list1,
                   # negative = negative_list1, action = action_list1)
df1_sentiment['Person1'] = person1_list1
df1_sentiment['Exclusive'] = exclusive_list1
df1_sentiment['NegativeEmo'] = negative_list1
df1_sentiment['Action'] = action_list1
df1_sentiment['length'] = length_list1
df1_sentiment['Spam'] = spam_list
df1_sentiment['Work'] = work_list
df1_sentiment['Sender'] = df.employee.values 
df1_sentiment['Date'] = df.date.values
#df1_sentiment['Position'] = df.position.values



with open("D:/BDS/5Hackason/set3/input2b.pkl", "wb") as f:
    pickle.dump(df1_sentiment, f)
    
    
    
#%% K means!
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# read in input
with open("D:/BDS/5Hackason/set3/input2b.pkl", "rb") as f:
    df2 = pickle.load(f) # 208165 rows
with open("D:/BDS/5Hackason/set3/input1b.pkl", "rb") as f:
    df1 = pickle.load(f) #204955 rows
df =pd.concat([df1, df2], ignore_index=True)
del df1
del df2
# df.to_csv('D:/BDS/5Hackason/set3/kmeans_input_b.csv')
# real k means going on

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.iloc[:,0:-3])

#%%
kmeans = KMeans(
    init="random",
    n_clusters=11,
    n_init=10,
    max_iter=300,
    random_state=42  
)

kmeans.fit(scaled_features)
plt.scatter(scaled_features[:,2],scaled_features[:,6], c=kmeans.labels_, cmap='rainbow')

#  save k-means result
df['label'] = kmeans.labels_
df.iloc[:,0:-4] = pd.DataFrame(scaled_features).values
with open("D:/BDS/5Hackason/set12/finalb.pkl", "wb") as f:
    pickle.dump(df, f)
    
#%%
with open("D:/BDS/5Hackason/set3/final.pkl", "rb") as f:
    df = pickle.load(f)
    
#%%
with open("D:/BDS/5Hackason/data/final.pkl", "wb") as f:
    pickle.dump(df, f)
    
#%% Randomly assign
