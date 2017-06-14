# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:34:47 2017

@author: vushesh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("D:/Studies/Online Competitions/DSG/train.csv")

data['ts_listen1'] = pd.to_datetime(data["ts_listen"], unit='s')

data['ts_listen2'] = data['ts_listen1'].apply(pd.tslib.normalize_date)

data['weekday'] = data['ts_listen1'].dt.dayofweek
data['hour'] = data['ts_listen1'].dt.hour

 
def dif(a,b):
    years = a.year 
    months =a.month
    days = a.day
    x = int(b/10000)
    y = b%10000
    return (years - x )*365 + (months - int(y/100))*30 + days - (y%100)
    
data['ageofsong'] = data.apply(lambda x: dif(x['ts_listen2'],x['release_date']),axis=1)
              
data['year'] = data['ts_listen2'].apply(lambda x: x.year)
data['month'] = data['ts_listen2'].apply(lambda x: x.month) 
data['day'] = data['ts_listen2'].apply(lambda x: x.day)
data.drop('ts_listen2',axis =1,inplace = True)   
data.drop('ts_listen1',axis =1,inplace = True)   

data['release_year'] = data['release_date'].apply(lambda x: int(x/10000))
data['release_month'] = data['release_date'].apply(lambda x: int((x%10000)/100))
data['release_day'] = data['release_date'].apply(lambda x:(x%10000)%100)


dump = data[["user_id","genre_id",'year','month','day']]

dump = dump.sort_values(['user_id' ,'year','month','day'],ascending = [True ,True,True ,True])
dump.drop('year',axis=1, inplace=True)
dump.drop('month',axis=1, inplace=True)
dump.drop('day',axis=1, inplace=True)

temp = np.array(dump)

preferred = []
preferred.append({})

def pgenre(boo,a,b):
    if boo == False:
        preferred.append({})
        preferred[a][b] = 1
        return b
    else:
        if b not in preferred[a]:
            preferred[a][b] = 1
        else:
            preferred[a][b] += 1
                     
    max_key = max(preferred[a], key=lambda k: preferred[a][k])
    return max_key

x = 0
 
a = []
 
inde = dump.index
      
for i in range(len(temp)):
    if temp[i][0] == x:
        a.append(pgenre(True,temp[i][0],temp[i][1]))
    else:
        x = temp[i][0]
        a.append(pgenre(False,temp[i][0],temp[i][1]))

len(a)
a = pd.DataFrame(a ,index = inde ,columns = ['preferred_genre'])

data = pd.concat([data, a], axis=1, join='inner')

data.count() 

data.to_csv("D:/Studies/Online Competitions/DSG/final.csv")

y = data[["is_listened"]]
data.drop("is_listened",axis=1,inplace=True)


#..................................Test..........................................#

test = pd.read_csv("D:/Studies/Online Competitions/DSG/test.csv")

test['ts_listen'] = pd.to_datetime(test["ts_listen"], unit='s')

test['ts_listen1'] = test['ts_listen'].apply(pd.tslib.normalize_date)

test['weekday'] = test['ts_listen'].dt.dayofweek
test['hour'] = test['ts_listen'].dt.hour


test['ageofsong'] = test.apply(lambda x: dif(x['ts_listen1'],x['release_date']),axis=1)
               
test['year'] = test['ts_listen1'].apply(lambda x: x.year)
test['month'] = test['ts_listen1'].apply(lambda x: x.month) 
test['day'] = test['ts_listen1'].apply(lambda x: x.day)
test.drop('ts_listen1',axis =1,inplace = True)   
test.drop('ts_listen',axis =1,inplace = True)   

test['release_year'] = test['release_date'].apply(lambda x: int(x/10000))
test['release_month'] = test['release_date'].apply(lambda x: int((x%10000)/100))
test['release_day'] = test['release_date'].apply(lambda x:(x%10000)%100)

sample = test[['sample_id']]
test.drop('sample_id',axis=1,inplace=True)


#--------Feature Selection.................#
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
ch2 = SelectKBest(chi2, k = 13)
X_new = ch2.fit(temp, tr)
np.set_printoptions(precision=5)
print(X_new.scores_)

features = data.columns
for i in range(13):
    print(features[i],": " ,X_new.scores_[i])


#............................................#


features = data.columns
for i in range(len(features)):
    for j in range(i+1,len(features)):
        x = data[features[i]].corr(data[features[j]])
        if x > 0.4:
            print(features[i],features[j],x)

data["platform_family"].unique()

data = pd.get_dummies(data, columns=['platform_name', 'platform_family'], drop_first=True)

dump = data[['platform_name', 'platform_family', "user_age","hour","user_id","user_gender", "day", "artist_id","Context_type"]]

trump = test[["platform_family","context_type","platform_name","user_age","hour","user_id","user_gender", "day", "artist_id"]]


dump = data[["platform_family","context_type","platform_name","user_age","user_gender","user_id","artist_id","day","hour"]]
#............................................#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dump, y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Use ...............xgboost....................#
import xgboost as xgb

def runXGB(train_X, train_y, seed_val=0):
	param = {}
	param['objective'] = 'multi:softmax'
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
  
#model = runXGB(dump, y, seed_val=0)
model = runXGB(X_train, y_train, seed_val=0)

#xgtest = xgb.DMatrix(trump)
xgtest = xgb.DMatrix(X_test)

preds = model.predict(xgtest)  

from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)


#.................Final result..........................#

target=open('D:/Studies/Online Competitions/DSG/preditions.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
    target.write("{},{}\n".format(sample['sample_id'][i],preds[i][1]))
target.close()  












      