# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:19:02 2021

@author: Ruoying
"""
#%% Import files and packages
import seaborn as sns
import pickle
import random
import numpy as np
#with open("D:/BDS/5Hackason/setfreq/final.pkl", "rb") as f:
#    df4 = pickle.load(f)
with open("D:/BDS/5Hackason/set4/finalb.pkl", "rb") as f:
    df4 = pickle.load(f)
#temp = df4.label.values
#df4['newlabel'] = [int(l==0) for l in temp]
df4 = df4.rename(columns={'Person1':'1stPersonNoun','Exclusive':'ExclusiveWord','length':'Length','label':'Class'}, errors="raise")
df4.loc[(df4.Class == 1),'Class']='Deceptive'
df4.loc[(df4.Class == 0),'Class']='Normal 1'
df4.loc[(df4.Class == 2),'Class']='Normal 2'
rowlist = list(np.random.choice(len(df4), 10000, replace=False))
sns.set(rc={'figure.figsize':(11.7,9)})
#%%
sns.set(font_scale=2) 
#sns.color_palette("Set2")
#Generate 5 random numbers between 10 and 30
sns.pairplot(df4[df4.index.isin(rowlist)],
             x_vars=['Polarity','Subjectivity','Length','Action'],
             y_vars=['Polarity','Subjectivity','Length','Action'],
             hue="Class", 
             palette=[(0.33999999999999997, 0.6059428571428571, 0.86),(0.86, 0.3712, 0.33999999999999997),(1.0, 0.8509803921568627, 0.1843137254901961),],
             #diag_kind="kde", 
             #markers=["o", "+","x"],
             plot_kws={"s": 80})
# summary of this part: only polarity divides classes well
#%%
sns.set(font_scale=2) 
#sns.color_palette("Set2")
#Generate 5 random numbers between 10 and 30
sns.pairplot(df4[df4.index.isin(rowlist)],
             x_vars=['Work','1stPersonNoun','ExclusiveWord','NegativeEmo'],
             #y_vars=['Polarity'],
             y_vars=['Work','1stPersonNoun','ExclusiveWord','NegativeEmo'],
             hue="Class", 
             palette=[(0.33999999999999997, 0.6059428571428571, 0.86),(0.86, 0.3712, 0.33999999999999997),(1.0, 0.8509803921568627, 0.1843137254901961),],
             #diag_kind="auto", 
             #markers=["^", "8"])
             plot_kws={"s": 80})
# summary of this part: STILL! only polarity divides classes well
#%%
sns.set(font_scale=2) 
#sns.color_palette("Set2")
#Generate 5 random numbers between 10 and 30
sns.pairplot(df4[df4.index.isin(rowlist)],
             x_vars=['Polarity','Positive', 'Negative','Spam'],
             y_vars=['Polarity','Positive', 'Negative','Spam'],
             hue="Class", 
             palette=[(0.33999999999999997, 0.6059428571428571, 0.86),(0.86, 0.3712, 0.33999999999999997),(1.0, 0.8509803921568627, 0.1843137254901961),],
             #diag_kind="hist", 
             #markers=["^", "8"])
             plot_kws={"s": 80})
#%% Sentiment Analysis VS  Deceptive theory
sns.set(font_scale=2) 
#sns.color_palette("Set2")
#Generate 5 random numbers between 10 and 30
sns.pairplot(df4[df4.index.isin(rowlist)],
             x_vars=['Polarity','Subjectivity', '1stPersonNoun','ExclusiveWord'],
             y_vars=['Polarity','Subjectivity', '1stPersonNoun','ExclusiveWord'],
             hue="Class", 
             palette=[(0.33999999999999997, 0.6059428571428571, 0.86),(0.86, 0.3712, 0.33999999999999997),(1.0, 0.8509803921568627, 0.1843137254901961),],
             #diag_kind="hist", 
             #markers=["^", "8"])
             plot_kws={"s": 80})