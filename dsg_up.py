import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Merge
from sklearn import preprocessing
from datetime import datetime
from keras.regularizers import l2 as l2_reg

data = pd.read_csv("D:/Studies/Online Competitions/DSG/final.csv")
test = pd.read_csv("D:/Studies/Online Competitions/DSG/test.csv")

day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
data['day'] = data['ts_listen'].apply(lambda x: day_map[datetime.fromtimestamp(x).strftime("%A")]) 

hour_map = {'00': 4, '01': 10, '02': 12, '03': 14, '04': 16, '05': 17, '06': 19, '07': 20, '08': 21, '09': 23, '10': 22, '11': 18, '12': 15, '13': 13, '14': 11, '15': 8, '16': 6, '17': 9, '18': 7, '19': 5, '20': 3, '21': 2, '22': 0, '23': 1}
data['hour'] = data['ts_listen'].apply(lambda x: hour_map[str(datetime.fromtimestamp(x).strftime("%H"))])

age_norm = preprocessing.StandardScaler().fit(data['user_age'])
data['user_age'] = age_norm.transform(data['user_age'])

meddur_norm = preprocessing.StandardScaler().fit(data['media_duration'])
data['media_duration'] = meddur_norm.transform(data['media_duration'])

users = list(set(data['user_id'].unique())|set(test['user_id'].unique()))
albums = list(set(data['album_id'].unique())|set(test['album_id'].unique()))
songs = list(set(data['media_id'].unique())|set(test['media_id'].unique()))
genres = list(set(data['genre_id'].unique())|set(test['genre_id'].unique()))
artists = list(set(data['artist_id'].unique())|set(test['artist_id'].unique()))

users_map = { v:k for k, v in dict(enumerate(users)).items() }
albums_map = { v:k for k, v in dict(enumerate(albums)).items() }
songs_map = { v:k for k, v in dict(enumerate(songs)).items() }
genres_map = { v:k for k, v in dict(enumerate(genres)).items() }
artists_map = { v:k for k, v in dict(enumerate(artists)).items() }

data['user_id'] = data['user_id'].apply(lambda x: users_map[x])
data['album_id'] = data['album_id'].apply(lambda x: albums_map[x])
data['media_id'] = data['media_id'].apply(lambda x: songs_map[x])
data['genre_id'] = data['genre_id'].apply(lambda x: genres_map[x])
data['artist_id'] = data['artist_id'].apply(lambda x: artists_map[x])

rel_map = {1900: 74, 1901: 75, 1902: 73, 1903: 85, 1905: 79, 1912: 65, 1928: 87, 1930: 82, 1933: 77, 1937: 64, 1939: 86, 1940: 71, 1941: 72, 1942: 81, 1944: 69, 1945: 67, 1946: 83, 1947: 84, 1949: 70, 1950: 80, 1951: 58, 1952: 78, 1953: 68, 1954: 66, 1955: 53, 1956: 51, 1957: 56, 1958: 33, 1959: 49, 1960: 25, 1961: 61, 1962: 35, 1963: 62, 1964: 52, 1965: 59, 1966: 46, 1967: 54, 1968: 55, 1969: 60, 1970: 30, 1971: 42, 1972: 36, 1973: 28, 1974: 47, 1975: 39, 1976: 27, 1977: 40, 1978: 44, 1979: 50, 1980: 43, 1981: 41, 1982: 48, 1983: 12, 1984: 57, 1985: 15, 1986: 32, 1987: 19, 1988: 17, 1989: 7, 1990: 45, 1991: 38, 1992: 20, 1993: 21, 1994: 13, 1995: 10, 1996: 18, 1997: 22, 1998: 14, 1999: 16, 2000: 3, 2001: 5, 2002: 11, 2003: 8, 2004: 0, 2005: 9, 2006: 1, 2007: 6, 2008: 4, 2009: 23, 2010: 24, 2011: 26, 2012: 29, 2013: 31, 2014: 2, 2015: 34, 2016: 37, 2017: 63, 3000: 76}
data['song_era'] = data['release_date'].apply(lambda x: rel_map[int(str(x)[:4])])

import json
with open('D:/Studies/Online Competitions/DSG/users_loc.json') as data_file:    
    dt = json.loads(data_file.read())

#data['country'] = data['user_id'].apply(lambda x: loc[str(x)])

user_loc = {}
for user, loc  in dt.items():
	if loc == '': loc = 'FR'
	user_loc[int(user)]=loc

locs = list(set(user_loc.values()))
loc_map = { v:k for k, v in dict(enumerate(locs)).items() }

data['user_loc'] = data['user_id'].apply(lambda x: loc_map[user_loc[x]])

x_user = np.array(data['user_id']).reshape(data['user_id'].shape[0],1)
x_media = np.array(data['media_id']).reshape(data['media_id'].shape[0],1)
x_album = np.array(data['album_id']).reshape(data['album_id'].shape[0],1)
x_genre = np.array(data['genre_id']).reshape(data['genre_id'].shape[0],1)
x_artist = np.array(data['artist_id']).reshape(data['artist_id'].shape[0],1)
x_gender = np.array(data['user_gender'])
x_era = np.array(data['song_era']).reshape(data['song_era'].shape[0],1)
x_hour = np.array(data['hour']).reshape(data['hour'].shape[0],1)
x_day = np.array(data['day']).reshape(data['day'].shape[0],1)
x_userloc = np.array(data['user_loc']).reshape(data['user_loc'].shape[0],1)
x_platname = np.array(data['platform_name']).reshape(data['platform_name'].shape[0],1)
x_platflam = np.array(data['platform_family']).reshape(data['platform_family'].shape[0],1)
x_context = np.array(data['context_type']).reshape(data['context_type'].shape[0],1)
x_listen = np.array(data['listen_type']).reshape(data['listen_type'].shape[0],1)

x_age = np.array(data['user_age'])
x_mediadur = np.array(data['media_duration'])
y = np.array(data['is_listened'])

test['user_age'] = age_norm.transform(test['user_age'])
test['media_duration'] = meddur_norm.transform(test['media_duration'])

test['day'] = test['ts_listen'].apply(lambda x: day_map[datetime.fromtimestamp(x).strftime("%A")]) 
test['hour'] = test['ts_listen'].apply(lambda x: hour_map[str(datetime.fromtimestamp(x).strftime("%H"))])

test['user_id'] = test['user_id'].apply(lambda x: users_map[x])
test['album_id'] = test['album_id'].apply(lambda x: albums_map[x])
test['media_id'] = test['media_id'].apply(lambda x: songs_map[x])
test['genre_id'] = test['genre_id'].apply(lambda x: genres_map[x])
test['artist_id'] = test['artist_id'].apply(lambda x: artists_map[x])

test['song_era'] = test['release_date'].apply(lambda x: rel_map[int(str(x)[:4])])
test['user_loc'] = test['user_id'].apply(lambda x: loc_map[user_loc[x]])

test_user = np.array(test['user_id']).reshape(test['user_id'].shape[0],1)
test_media = np.array(test['media_id']).reshape(test['media_id'].shape[0],1)
test_album = np.array(test['album_id']).reshape(test['album_id'].shape[0],1)
test_genre = np.array(test['genre_id']).reshape(test['genre_id'].shape[0],1)
test_artist = np.array(test['artist_id']).reshape(test['artist_id'].shape[0],1)
test_gender = np.array(test['user_gender'])
test_era = np.array(test['song_era']).reshape(test['song_era'].shape[0],1)
test_hour = np.array(test['hour']).reshape(test['hour'].shape[0],1)
test_day = np.array(test['day']).reshape(test['day'].shape[0],1)
test_userloc = np.array(test['user_loc']).reshape(test['user_loc'].shape[0],1)
test_platname = np.array(test['platform_name']).reshape(test['platform_name'].shape[0],1)
test_platflam = np.array(test['platform_family']).reshape(test['platform_family'].shape[0],1)
test_context = np.array(test['context_type']).reshape(test['context_type'].shape[0],1)
test_listen = np.array(test['listen_type']).reshape(test['listen_type'].shape[0],1)

test_age = np.array(test['user_age'])
test_mediadur = np.array(test['media_duration'])
#----model----#

users_emb = Sequential()
users_emb.add(Embedding(len(users), 70, input_length=1, W_regularizer=l2_reg(0.0)))
users_emb.add(Flatten())

user_loc_emb = Sequential()
user_loc_emb.add(Embedding(len(loc_map), 5, input_length=1, W_regularizer=l2_reg(0.0)))
user_loc_emb.add(Flatten())

songs_emb = Sequential()
songs_emb.add(Embedding(len(songs), 50, input_length=1, W_regularizer=l2_reg(0.0)))
songs_emb.add(Flatten())

genre_emb = Sequential()
genre_emb.add(Embedding(len(genres), 5, input_length=1, W_regularizer=l2_reg(0.0)))
genre_emb.add(Flatten())

artist_emb = Sequential()
artist_emb.add(Embedding(len(artists), 5, input_length=1, W_regularizer=l2_reg(0.0)))
artist_emb.add(Flatten())

era_emb = Sequential()
era_emb.add(Embedding(len(data['song_era'].unique()), 2, input_length=1, W_regularizer=l2_reg(0.0)))
era_emb.add(Flatten())

hour_emb = Sequential()
hour_emb.add(Embedding(len(hour_map), 8, input_length=1, W_regularizer=l2_reg(0.0)))
hour_emb.add(Flatten())

day_emb = Sequential()
day_emb.add(Embedding(len(day_map), 8, input_length=1, W_regularizer=l2_reg(0.0)))
day_emb.add(Flatten())

platname_emb = Sequential()
platname_emb.add(Embedding(3, 10, input_length=1, W_regularizer=l2_reg(0.0)))
platname_emb.add(Flatten())

platfam_emb = Sequential()
platfam_emb.add(Embedding(3, 10, input_length=1, W_regularizer=l2_reg(0.0)))
platfam_emb.add(Flatten())

context_emb = Sequential()
context_emb.add(Embedding(len(data['context_type'].unique()), 10, input_length=1, W_regularizer=l2_reg(0.0)))
context_emb.add(Flatten())

listen_emb = Sequential()
listen_emb.add(Embedding(2, 4, input_length=1, W_regularizer=l2_reg(0.0)))
listen_emb.add(Flatten())

age = Sequential()
age.add(Dense(1, input_dim=1))

gender = Sequential()
gender.add(Dense(1, input_dim=1))

users_model = Sequential()
users_model.add(Merge([users_emb, user_loc_emb, gender, age], mode='concat'))
users_model.add(Dense(64))
users_model.add(Activation('relu'))
users_model.add(Dropout(0.5))
users_model.add(Dense(48))
users_model.add(Activation('relu'))
users_model.add(Dropout(0.5))
users_model.add(Dense(32))
users_model.add(Activation('relu'))
users_model.add(Dropout(0.5))
users_model.add(Dense(16))
users_model.add(Activation('relu'))

songs_model = Sequential()
songs_model.add(Merge([songs_emb, genre_emb, artist_emb, era_emb], mode='concat'))
songs_model.add(Dense(82))
songs_model.add(Activation('relu'))
songs_model.add(Dropout(0.5))
songs_model.add(Dense(64))
songs_model.add(Activation('relu'))
songs_model.add(Dropout(0.5))
songs_model.add(Dense(48))
songs_model.add(Activation('relu'))
songs_model.add(Dropout(0.5))
songs_model.add(Dense(32))
songs_model.add(Activation('relu'))
songs_model.add(Dropout(0.5))
songs_model.add(Dense(16))
songs_model.add(Activation('relu'))

model = Sequential()
model.add(Merge([users_model, songs_model, hour_emb, day_emb, platname_emb, platfam_emb, context_emb, listen_emb], mode='concat'))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([x_user,x_userloc,x_gender,x_age,x_media,x_genre,x_artist,x_era,x_mediadur,x_hour,x_day,x_platname,x_platflam,x_context,x_listen], y, batch_size=500, validation_split=0.1, nb_epoch=10)

preds = model.predict([test_user,test_userloc,test_gender,test_age,test_media,test_genre,test_artist,test_era,test_mediadur,test_hour,test_day,test_platname,test_platflam,test_context,test_listen])

target = open('sub_dsg_up.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
	target.write("{},{}\n".format(i,preds[i][0]))
target.close()








