import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Merge
from sklearn import preprocessing
from keras.callbacks import EarlyStopping

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#--- combine test and train with indicator variable
data['istest'] = 0
test['istest'] = 1
datatest = data.append(test, ignore_index = True)
#---------------------------------------------------

#-----group by user_id and order on timestamp
datatest['ts_listen1'] = pd.to_datetime(datatest["ts_listen"], unit='s')
datatest = datatest.groupby('user_id')
datatest = datatest.apply(lambda x : x.sort_values(by = 'ts_listen1'))
datatest = datatest.reset_index(drop = True)
datatest.drop('ts_listen1', axis=1, inplace=True)
#------------------------------------------------------------

#------ create history variable as string of is_listened for last 8 songs (for first 8 songs, string of all previous songs)
datatest['history'] = datatest['is_listened'].shift().where(datatest['user_id'].shift() == datatest['user_id'], '').astype('str')+datatest['is_listened'].shift(2).where(datatest['user_id'].shift(2) == datatest['user_id'], '').astype('str') + datatest['is_listened'].shift(3).where(datatest['user_id'].shift(3) == datatest['user_id'], '').astype('str')+datatest['is_listened'].shift(4).where(datatest['user_id'].shift(4) == datatest['user_id'], '').astype('str') +datatest['is_listened'].shift(5).where(datatest['user_id'].shift(5) == datatest['user_id'], '').astype('str')+datatest['is_listened'].shift(6).where(datatest['user_id'].shift(6) == datatest['user_id'], '').astype('str') + datatest['is_listened'].shift(7).where(datatest['user_id'].shift(7) == datatest['user_id'], '').astype('str')+datatest['is_listened'].shift(8).where(datatest['user_id'].shift(8) == datatest['user_id'], '').astype('str')
#----------------------------------------------------------

#---------separate train and test and remove indicator variable
data = datatest[datatest['istest']==0]
data.drop('istest', axis=1, inplace=True)
test = datatest[datatest['istest']==1]
test.drop('istest', axis=1, inplace=True)
#---------------------------------------------------------

#----------encode strings as categories
history = list(set(data['history'].unique())|set(test['history'].unique()))
history_map = { v:k for k, v in dict(enumerate(history)).items() }
data['history'] = data['history'].apply(lambda x: history_map[x])
test['history'] = test['history'].apply(lambda x: history_map[x])
#--------------------------------------------------------------

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

x11 = np.array(data['user_id']).reshape(data['user_id'].shape[0],1)
x12 = np.array(data['media_id']).reshape(data['media_id'].shape[0],1)
x13 = np.array(data['genre_id']).reshape(data['genre_id'].shape[0],1)
x14 = np.array(data['artist_id']).reshape(data['artist_id'].shape[0],1)
x15 = np.array(data['history']).reshape(data['history'].shape[0],1)
x2 = data[['platform_name_1', 'platform_family_1', 'platform_name_2', 'platform_family_2', 'listen_type', 'user_gender',
'user_age','context_type']]
n_features = len(x2.columns)
print(x2.columns)
x2 = np.array(x2)

y=np.array(data['is_listened'])

print("dataset prepared")

test['user_age'] = age_norm.transform(test['user_age'])
test['media_duration'] = meddur_norm.transform(test['media_duration'])

test = pd.get_dummies(test, columns=['platform_name', 'platform_family'], drop_first=True)

test['user_id'] = test['user_id'].apply(lambda x: users_map[x])
test['media_id'] = test['media_id'].apply(lambda x: songs_map[x])
test['genre_id'] = test['genre_id'].apply(lambda x: genres_map[x])
test['artist_id'] = test['artist_id'].apply(lambda x: artists_map[x])
test['context_type'] = test['context_type'].apply(lambda x:context_map[x])

test11 = np.array(test['user_id']).reshape(test['user_id'].shape[0],1)
test12 = np.array(test['media_id']).reshape(test['media_id'].shape[0],1)
test13 = np.array(test['genre_id']).reshape(test['genre_id'].shape[0],1)
test14 = np.array(test['artist_id']).reshape(test['artist_id'].shape[0],1)
test15 = np.array(test['history']).reshape(test['history'].shape[0],1)
test2 = test[['platform_name_1', 'platform_family_1', 'platform_name_2', 'platform_family_2','listen_type',
                  'user_gender','user_age','context_type']]
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

history_emb = Sequential()
history_emb.add(Embedding(len(history), 20, input_length=1))
history_emb.add(Flatten())

features = Sequential()
features.add(Dense(9, input_dim=n_features))

model = Sequential()
model.add(Merge([users_emb,songs_emb, genre_emb, artist_emb, history_emb, features], mode='concat'))

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

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss',patience=3,verbose=0)]
model.fit([x11,x12,x13,x14, x15, x2], y, batch_size=500, epochs=15,shuffle=True,verbose=1,validation_split=0.1,callbacks=callbacks)


preds = model.predict([test11, test12, test13, test14, test15, test2])

target = open('newe.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
	target.write("{},{}\n".format(i,preds[i][0]))
