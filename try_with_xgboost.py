# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:51:32 2017

@author: vushesh
"""
import json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Merge
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.regularizers import l2 as l2_reg
from datetime import datetime

data = pd.read_csv("D:/Studies/Online Competitions/DSG/final.csv")
test = pd.read_csv("D:/Studies/Online Competitions/DSG/test.csv")

len(test["genre_id"].unique())
b = test["media_id"].value_counts()
a = data["media_id"].value_counts()

count = 0

for key in b.keys():
    if key in a:
        count += a[key]

with open('D:/Studies/Online Competitions/DSG/songs_meta_all/songs_meta_all.json','r') as s:
	songs=json.loads(s.read())
   
data["bpm"] = data["media_id"].apply(lambda x: songs[str(x)]['bpm'] if str(x) in songs else 121 )
data["rank"] = data["media_id"].apply(lambda x: songs[str(x)]['rank'] if str(x) in songs else 655757 )
data["gain"] = data["media_id"].apply(lambda x: songs[str(x)]['gain'] if str(x) in songs else -9.2 )

test["bpm"] = test["media_id"].apply(lambda x: songs[str(x)]['bpm'] if str(x) in songs else 121 )
test["rank"] = test["media_id"].apply(lambda x: songs[str(x)]['rank'] if str(x) in songs else 655757 )
test["gain"] = test["media_id"].apply(lambda x: songs[str(x)]['gain'] if str(x) in songs else -9.2 )


#data = pd.get_dummies(data, columns=['platform_name', 'platform_family'], drop_first=True)
 
import json
with open('D:/Studies/Online Competitions/DSG/users_loc.json') as data_file:    
    loc = json.loads(data_file.read())

data['country'] = data['user_id'].apply(lambda x: loc[str(x)])

data['country'] = data['country'].apply(lambda x: 'FR' if x== '' else x)


a = data["country"].value_counts()
b = data["genre_id"].value_counts()
c = data["album_id"].value_counts()
d = data["artist_id"].value_counts()

temp = np.array(data[["country","genre_id","album_id","artist_id","is_listened"]])
x = {}
y = {}
z = {}
t = {}

for i in range(len(temp)):
    if temp[i][4] == 1:
        if temp[i][0] in x:
            x[temp[i][0]] += 1
        else:
            x[temp[i][0]] = 1
             
        if temp[i][1] in y:
            y[temp[i][1]] += 1
        else:
            y[temp[i][1]] = 1
             
        if temp[i][2] in z:
            z[temp[i][2]] += 1
        else:
            z[temp[i][2]] = 1
             
        if temp[i][3] in t:
            t[temp[i][3]] += 1
        else:
            t[temp[i][3]] = 1
    else:
        x[temp[i][0]] = 0
        y[temp[i][1]] = 0
        z[temp[i][2]] = 0
        t[temp[i][3]] = 0
         
country_average = {}   
genre_id_average = {}
album_id_average = {}
artist_id_average = {}     

for key in x.keys():
    country_average[key] = x[key]/a[key]
        
for key in y.keys():
    genre_id_average[key] = y[key]/b[key]

for key in z.keys():
    album_id_average[key] = z[key]/c[key]
    
for key in t.keys():
    artist_id_average[key] = t[key]/d[key]    

data['country_avg'] = data['country'].apply(lambda x: country_average[x])
data['genre_id_avg'] = data['genre_id'].apply(lambda x: genre_id_average[x])
data['album_id_avg'] = data['album_id'].apply(lambda x: album_id_average[x])
data['artist_id_avg'] = data['artist_id'].apply(lambda x: artist_id_average[x])

country_std = {}
for key in x.keys():
    country_std[key] = (x[key]*((1-country_average[key])*(1-country_average[key])) + (a[key]-x[key])*(country_average[key]*country_average[key]) ) /(a[key]-1)
    
data['country_std'] = data['country'].apply(lambda x: country_std[x])
data.drop('country' ,axis=1 ,inplace=True)

artist_id_std = {}
for key in t.keys():
    artist_id_std[key] = (t[key]*((1-artist_id_average[key])*(1-artist_id_average[key])) + (d[key]-t[key])*(artist_id_average[key]*artist_id_average[key]) ) /(d[key]-1)
 

print("location done")


test['country'] = test['user_id'].apply(lambda x: loc[str(x)])

test['country'] = test['country'].apply(lambda x: 'FR' if x== '' else x)

test['country_avg'] = test['country'].apply(lambda x: country_average[x] if x in country_average else 1)
test['genre_id_avg'] = test['genre_id'].apply(lambda x: genre_id_average[x] if x in genre_id_average else 1)
test['album_id_avg'] = test['album_id'].apply(lambda x: album_id_average[x] if x in album_id_average else 1)
test['artist_id_avg'] = test['artist_id'].apply(lambda x: artist_id_average[x] if x in artist_id_average else 1) 

test['country_std'] = test['country'].apply(lambda x: country_std[x])
test.drop('country' ,axis=1 ,inplace=True)
"""
age_norm = preprocessing.StandardScaler().fit(data['user_age'])
data['user_age'] = age_norm.transform(data['user_age'])
"""


data = data.loc[data["listen_type"]==1]
#data = data.loc[data["context_type"].isin([1,5,20,23])]


test['ts_listen1'] = pd.to_datetime(test["ts_listen"], unit='s')

test['ts_listen2'] = test['ts_listen1'].apply(pd.tslib.normalize_date)

test['hour'] = test['ts_listen1'].dt.hour
test['day'] = test['ts_listen2'].apply(lambda x: x.day)


features = data.columns
for i in range(len(features)):
    print(features[i],data[features[i]].corr(data['is_listened']))

y = data[["is_listened"]]
#...................................................#   


trump = test[['platform_name', 'platform_family' ,"user_age","listen_type", "day","user_id" ,"hour","ts_listen","user_gender" ,"context_type" ,"country_code"]] 



dump = data[['platform_name', 'platform_family' ,"user_age","user_id" ,"user_gender",
              "gain","rank" ,"artist_id" ,"ts_listen" ,"hour" ,"day","context_type"]] 


trump = test[['platform_name', 'platform_family' ,"user_age","user_id" ,"user_gender",
              "gain" ,"rank" ,"artist_id" ,"ts_listen" ,"hour","day","context_type"]] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dump, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Use ...............xgboost....................#
import xgboost as xgb

def runXGB(train_X, train_y, seed_val=0):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.3
	param['max_depth'] = 10
	param['silent'] = 1
	param['num_class'] = 2
	param['eval_metric'] = "auc"
	param['min_child_weight'] = 3
	param['subsample'] = 0.8
	param['colsample_bytree'] = 0.8
	param['seed'] = seed_val
	num_rounds = 80

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model
  
model = runXGB(dump, y, seed_val=0)
#model = runXGB(X_train, y_train, seed_val=0)

xgtest = xgb.DMatrix(trump)
#xgtest = xgb.DMatrix(X_test)

preds = model.predict(xgtest)  

for i in range(10):
    print(preds[i][0],preds[i][1])
    
from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)


sample = test[['sample_id']]

target=open('try_with_xgboost.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
    target.write("{},{}\n".format(sample['sample_id'][i],preds[i][1]))
target.close()  


a = pd.read_csv("try_with_xgboost.csv")
b = pd.read_csv("correl_ensemble.csv")

a = np.array(a)
b = np.array(b)

target=open('try_with.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
    target.write("{},{}\n".format(sample['sample_id'][i],max(a[i][1],b[i][1])))
target.close()  

            
    
    















    
    