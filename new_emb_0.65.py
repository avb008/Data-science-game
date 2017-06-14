import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Merge
from sklearn import preprocessing
from keras.callbacks import EarlyStopping

data = pd.read_csv("D:/Studies/Online Competitions/DSG/train.csv")
test = pd.read_csv("D:/Studies/Online Competitions/DSG/test.csv")

xcols = ['genre_id', 'ts_listen', 'media_id', 'album_id', 'context_type', 'release_date', 'platform_name', 'platform_family', 'media_duration', 'listen_type', 'user_gender', 'user_id', 'artist_id', 'user_age']

context_map = {}
for i in data['context_type'].unique():
	context_map[i] = np.mean(data['is_listened'].loc[data['context_type']==i])
data['context_type'] = data['context_type'].apply(lambda x: context_map[x])

age_norm = preprocessing.StandardScaler().fit(data['user_age'])
data['user_age'] = age_norm.transform(data['user_age'])

meddur_norm = preprocessing.StandardScaler().fit(data['media_duration'])
data['media_duration'] = meddur_norm.transform(data['media_duration'])

data = pd.get_dummies(data, columns=['platform_name', 'platform_family'], drop_first=True)

users = list(set(data['user_id'].unique())|set(test['user_id'].unique()))
songs = list(set(data['media_id'].unique())|set(test['media_id'].unique()))
genres = list(set(data['genre_id'].unique())|set(test['genre_id'].unique()))
artists = list(set(data['artist_id'].unique())|set(test['artist_id'].unique()))

users_map = { v:k for k, v in dict(enumerate(users)).items() }
songs_map = { v:k for k, v in dict(enumerate(songs)).items() }
genres_map = { v:k for k, v in dict(enumerate(genres)).items() }
artists_map = { v:k for k, v in dict(enumerate(artists)).items() }

data['user_id'] = data['user_id'].apply(lambda x: users_map[x])
data['media_id'] = data['media_id'].apply(lambda x: songs_map[x])
data['genre_id'] = data['genre_id'].apply(lambda x: genres_map[x])
data['artist_id'] = data['artist_id'].apply(lambda x: artists_map[x])

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
 
data['artist_id_std'] = data['artist_id'].apply(lambda x: artist_id_std[x])
print("location done")

data_grouped = data.groupby(['user_id', 'media_id']).agg({'ts_listen':'max'})
data_grouped = data_grouped.reset_index()
data_grouped = data_grouped.rename(columns={'ts_listen':'ts_listen_max'})
data = pd.merge(data, data_grouped, how='left', on=['user_id', 'media_id'])


features = data.columns
for i in range(len(features)):
    print(features[i],data[features[i]].corr(data['is_listened']))
    


x11 = np.array(data['user_id']).reshape(data['user_id'].shape[0],1)
x12 = np.array(data['media_id']).reshape(data['media_id'].shape[0],1)
x13 = np.array(data['genre_id']).reshape(data['genre_id'].shape[0],1)
x14 = np.array(data['artist_id']).reshape(data['artist_id'].shape[0],1)
x2 = data[['platform_name_1', 'platform_family_1', 'platform_name_2', 'platform_family_2', 'listen_type', 'user_gender',
'user_age','context_type' ,"genre_id_avg","album_id_avg" ,"artist_id_avg" ]]
n_features = len(x2.columns)
print(x2.columns)
x2 = np.array(x2)

y=np.array(data['is_listened'])

print("dataset prepared")

test['user_age'] = age_norm.transform(test['user_age'])
test['media_duration'] = meddur_norm.transform(test['media_duration'])
"""
test['ageofsong'] = ageofsong_norm.transform(test['ageofsong'])
test['weekday'] = weekday_norm.transform(test['weekday'])
test['hour'] = hour_norm.transform(test['hour'])
"""
test = pd.get_dummies(test, columns=['platform_name', 'platform_family'], drop_first=True)

test['user_id'] = test['user_id'].apply(lambda x: users_map[x])
test['media_id'] = test['media_id'].apply(lambda x: songs_map[x])
test['genre_id'] = test['genre_id'].apply(lambda x: genres_map[x])
test['artist_id'] = test['artist_id'].apply(lambda x: artists_map[x])
test['context_type'] = test['context_type'].apply(lambda x:context_map[x])
test['country'] = test['user_id'].apply(lambda x: loc[str(x)])

test['country'] = test['country'].apply(lambda x: 'FR' if x== '' else x)

test['country_avg'] = test['country'].apply(lambda x: country_average[x] if x in country_average else 1)
test['genre_id_avg'] = test['genre_id'].apply(lambda x: genre_id_average[x] if x in genre_id_average else 1)
test['album_id_avg'] = test['album_id'].apply(lambda x: album_id_average[x] if x in album_id_average else 1)
test['artist_id_avg'] = test['artist_id'].apply(lambda x: artist_id_average[x] if x in artist_id_average else 1) 

test['country_std'] = test['country'].apply(lambda x: country_std[x])
test.drop('country' ,axis=1 ,inplace=True)

test['artist_id_std'] = test['artist_id'].apply(lambda x: artist_id_std[x] if x in artist_id_average else 0)

test11 = np.array(test['user_id']).reshape(test['user_id'].shape[0],1)
test12 = np.array(test['media_id']).reshape(test['media_id'].shape[0],1)
test13 = np.array(test['genre_id']).reshape(test['genre_id'].shape[0],1)
test14 = np.array(test['artist_id']).reshape(test['artist_id'].shape[0],1)
test2 = test[['platform_name_1', 'platform_family_1', 'platform_name_2', 'platform_family_2','listen_type',
                  'user_gender','user_age','context_type' ,"genre_id_avg","album_id_avg" ,"artist_id_avg" ]]
test2 = np.array(test2)

print("test prepared")

users_emb = Sequential()
users_emb.add(Embedding(len(users), 70, input_length=1))
users_emb.add(Flatten())

songs_emb = Sequential()
songs_emb.add(Embedding(len(songs), 30, input_length=1))
songs_emb.add(Flatten())

genre_emb = Sequential()
genre_emb.add(Embedding(len(genres), 60, input_length=1))
genre_emb.add(Flatten())

artist_emb = Sequential()
artist_emb.add(Embedding(len(artists), 20, input_length=1))
artist_emb.add(Flatten())
"""
Emb1 = Sequential()
Emb1.add(Merge([users_emb, songs_emb] ,mode='concat'))
Emb1.add(Dense(75))
Emb1.add(Activation('relu'))
Emb1.add(Dropout(0.4))

Emb2 = Sequential()
Emb2.add(Merge([users_emb, genre_emb],mode='concat'))
Emb2.add(Dense(95))
Emb2.add(Activation('relu'))
Emb2.add(Dropout(0.3))
"""
Emb3 = Sequential()
Emb3.add(Merge([users_emb, artist_emb],mode='concat'))
Emb3.add(Dense(68))
Emb3.add(Activation('relu'))
Emb3.add(Dropout(0.2))

features = Sequential()
features.add(Dense(9, input_dim=n_features))

model = Sequential()
model.add(Merge([users_emb,Emb3,features], mode='concat'))

model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss',patience=3,verbose=0)]
model.fit([x11,x14,x2], y, batch_size=500, epochs=9,shuffle=True,verbose=1,validation_split=0.1,callbacks=callbacks)

#print(model.layers[1].get_weights())

preds = model.predict([test11,test14,test2])


target = open('n.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
	target.write("{},{}\n".format(i,preds[i][0]))

"""
preds1 = pd.read_csv("newemb_0.65.csv")
preds2 = pd.read_csv("Submission7.csv")


temp1 = np.array(preds1)
temp2 = np.array(preds2)

temp3 = []
for i in range(len(preds1)):
    x = (temp1[i][1] +temp2[i][1])/2
    temp3.append(x)   
    
    
target = open('ensemble5.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(temp3)):
	target.write("{},{}\n".format(i,temp3[i]))   
    
target.close()    
"""    

