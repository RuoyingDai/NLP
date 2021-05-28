# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:23:04 2021

@author: Ruoying
"""
#%% import classification result
import pickle
#with open("D:/BDS/5Hackason/data/final.pkl", "rb") as f:
#    df0 = pickle.load(f)
#with open("D:/BDS/5Hackason/set1/final.pkl", "rb") as f:
#    df1 = pickle.load(f)
#with open("D:/BDS/5Hackason/set2/final.pkl", "rb") as f:
#    df2 = pickle.load(f)
with open("D:/BDS/5Hackason/set3/finalb.pkl", "rb") as f:
    df3 = pickle.load(f)
with open("D:/BDS/5Hackason/set4/finalb.pkl", "rb") as f:
    df4 = pickle.load(f)
with open("D:/BDS/5Hackason/set5/finalb.pkl", "rb") as f:
    df5 = pickle.load(f)
with open("D:/BDS/5Hackason/set6/finalb.pkl", "rb") as f:
    df6 = pickle.load(f)
with open("D:/BDS/5Hackason/set7/finalb.pkl", "rb") as f:
    df7 = pickle.load(f)
with open("D:/BDS/5Hackason/set8/finalb.pkl", "rb") as f:
    df8 = pickle.load(f)
with open("D:/BDS/5Hackason/set9/finalb.pkl", "rb") as f:
    df9 = pickle.load(f)
with open("D:/BDS/5Hackason/set10/finalb.pkl", "rb") as f:
    df10 = pickle.load(f)
with open("D:/BDS/5Hackason/set11/finalb.pkl", "rb") as f:
    df11 = pickle.load(f)
with open("D:/BDS/5Hackason/set12/finalb.pkl", "rb") as f:
    df12 = pickle.load(f)
# import human label by us
import pandas as pd
#label = pd.read_excel('D:/BDS/5Hackason/data/180_tokens_labelled.xlsx',
#                    header = None)
label = pd.read_excel('D:/BDS/5Hackason/data/180_tokens_v2.xlsx',
                    header = None)
#%% extract the label by index, for df0-df3
with open("D:/BDS/5Hackason/data/sample.pkl", "rb") as f:
    sample = pickle.load(f)
d_index = [13,14,16,18,50,51,58,67,68,69,72,74,75,79,
                   85,88,90,97,98,106,116,134,173,175]
df180 = df3[df3.index.isin(sample.index)]
kmeanslabel = [df180.iloc[row,-1] for row in d_index]
print(kmeanslabel.count(0))
print(kmeanslabel.count(1))
print(kmeanslabel.count(2))
print(kmeanslabel.count(3))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_true = list(label.iloc[:-1,0])
temp = list(df180.label)
y_pred = temp
#y_pred = [abs(l -1) for l in temp]
#y_pred = list(df180.label)
print(confusion_matrix(y_true, y_pred))
target_names = ['Innocent', 'Deceptive']
print(classification_report(y_true, y_pred, target_names=target_names))
#%% extract the label by index, for df4df5df6
import pandas as pd
label = pd.read_excel('D:/BDS/5Hackason/data/180_tokens_v2.xlsx',
                    header = None)
with open("D:/BDS/5Hackason/data/sample.pkl", "rb") as f:
    sample = pickle.load(f)
#d_index = [13,14,16,18,50,51,58,67,68,69,72,74,75,79,
#                   85,88,90,97,98,106,116,134,173,175]
df180 = df13[df13.index.isin(sample.index)]
#kmeanslabel = [df180.iloc[row,-1] for row in d_index]
#print(kmeanslabel.count(0))
#print(kmeanslabel.count(1))
#print(kmeanslabel.count(2))
#print(kmeanslabel.count(3))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#y_true = list(label.iloc[:-1,0])
y_true = list(label.iloc[:,0])
temp = list(df180.label)
#df4:class 0
y_pred = [int(l==4) for l in temp]
#y_pred = list(df180.label)
print(confusion_matrix(y_true, y_pred))
target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred, target_names=target_names))

#%% label study

print('2 classes: \nclass 0 has {} points;\nclass 1 has {} points.'.format(list(df3.label).count(0),list(df3.label).count(1)))

print('3 classes: \nclass 0 has {} points;\nclass 1 has {} points;\nclass 2 has {} points.'.format(list(df4.label).count(0),list(df4.label).count(1), list(df4.label).count(2)))

print('4 classes: \nclass 0 has {} points;\nclass 1 has {} points;\nclass 2 has {} points; \nclass 3 has {} points.'.format(list(df5.label).count(0),list(df5.label).count(1), list(df5.label).count(2), list(df5.label).count(3)))

print('5 classes: \nclass 0 has {} points;\nclass 1 has {} points;\nclass 2 has {} points; \nclass 3 has {} points;\nclass 4 has {} points.'.format(list(df6.label).count(0),list(df6.label).count(1), list(df6.label).count(2), list(df6.label).count(3), list(df6.label).count(4)))

print('6 classes: \nclass 0 has {} points;\nclass 1 has {} points;\nclass 2 has {} points; \nclass 3 has {} points;\nclass 4 has {} points;\nclass 5 has {} points.'.format(list(df7.label).count(0),list(df7.label).count(1), list(df7.label).count(2), list(df7.label).count(3), list(df7.label).count(4),list(df7.label).count(5)))


#%% overlapp study
base =sum((df3.label.values == 1))
sum((df3.label.values == 1) & (df4.label.values ==0))
# 96158
sum((df3.label.values == 1) & (df4.label.values ==1))
# 167
#%% df5, 4 classes
print(sum((df3.label.values == 1) & (df5.label.values ==0)))
#87028
print(sum((df3.label.values == 1) & (df5.label.values ==1)))
#0
print(sum((df3.label.values == 1) & (df5.label.values ==2)))
#15863
print(sum((df3.label.values == 1) & (df5.label.values ==3)))
#269

#%% df6, 5 classes
print(sum((df3.label.values == 1) & (df6.label.values ==0)))
#85328
print(sum((df3.label.values == 1) & (df6.label.values ==1)))
#0
print(sum((df3.label.values == 1) & (df6.label.values ==2)))
#1106
print(sum((df3.label.values == 1) & (df6.label.values ==3)))
#15887
print(sum((df3.label.values == 1) & (df6.label.values ==4)))
#839
#%% df7, 6 classes
print(sum((df3.label.values == 1) & (df7.label.values ==0)))
#773
print(sum((df3.label.values == 1) & (df7.label.values ==1)))
#0
print(sum((df3.label.values == 1) & (df7.label.values ==2)))
#54
print(sum((df3.label.values == 1) & (df7.label.values ==3)))
#37710
print(sum((df3.label.values == 1) & (df7.label.values ==4)))
#52286
print(sum((df3.label.values == 1) & (df7.label.values ==5)))
#12337

#%% df8, 7 classes
print(sum((df3.label.values == 1) & (df8.label.values ==0)))
#62514
print(sum((df3.label.values == 1) & (df8.label.values ==1)))
#11883
print(sum((df3.label.values == 1) & (df8.label.values ==2)))
#1293
print(sum((df3.label.values == 1) & (df8.label.values ==3)))
#415
print(sum((df3.label.values == 1) & (df8.label.values ==4)))
#27030
print(sum((df3.label.values == 1) & (df8.label.values ==5)))
#25
print(sum((df3.label.values == 1) & (df8.label.values ==6)))
#0

#%% df9, 8 classes
print(sum((df3.label.values == 1) & (df9.label.values ==0)))
#409
print(sum((df3.label.values == 1) & (df9.label.values ==1)))
#0
print(sum((df3.label.values == 1) & (df9.label.values ==2)))
#0
print(sum((df3.label.values == 1) & (df9.label.values ==3)))
#36166
print(sum((df3.label.values == 1) & (df9.label.values ==4)))
#35763
print(sum((df3.label.values == 1) & (df9.label.values ==5)))
#25349
print(sum((df3.label.values == 1) & (df9.label.values ==6)))
#1182
print(sum((df3.label.values == 1) & (df9.label.values ==7)))
#4291
