import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Merge
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
import json
from scipy import stats

data = pd.read_csv("D:/Studies/Online Competitions/DSG/train.csv")
test = pd.read_csv("D:/Studies/Online Competitions/DSG/test.csv")

y = np.array(data['is_listened'])

#------for LSTM-----#
def catmap(data,field):
	dt = data[[field,'is_listened']].groupby([field]).mean()
	dt = dt.to_dict()['is_listened']
	return dt

context_catmap = catmap(data,'context_type')
data['context'] = data['context_type'].apply(lambda x:context_catmap[x])

songs_catmap = catmap(data,'media_id')
data['song'] = data['media_id'].apply(lambda x:songs_catmap[x]) 

genres_catmap = catmap(data,'genre_id')
data['genre'] = data['genre_id'].apply(lambda x:genres_catmap[x]) 

artists_catmap = catmap(data,'artist_id')
data['artist'] = data['artist_id'].apply(lambda x:artists_catmap[x]) 

dt = pd.get_dummies(data[['platform_name', 'platform_family']], columns=['platform_name', 'platform_family'], drop_first=True)
data['platform_name_1'] = dt['platform_name_1']
data['platform_family_1'] = dt['platform_family_1']
data['platform_name_2'] = dt['platform_name_2']
data['platform_family_2'] = dt['platform_family_2']

users = list(set(data['user_id'].unique())|set(test['user_id'].unique()))
songs = list(set(data['media_id'].unique())|set(test['media_id'].unique()))
genres = list(set(data['genre_id'].unique())|set(test['genre_id'].unique()))
artists = list(set(data['artist_id'].unique())|set(test['artist_id'].unique()))


with open('D:/Studies/Online Competitions/DSG/users_loc.json') as data_file:    
    loc = json.loads(data_file.read())
    
data['country'] = data['user_id'].apply(lambda x: loc[str(x)])
data['country'] = data['country'].apply(lambda x: 'FR' if x== '' else x)
loc_map = catmap(data,'country')
data['country'] = data['country'].apply(lambda x: loc_map[x])

test['country'] = test['user_id'].apply(lambda x: loc[str(x)])
test['country'] = test['country'].apply(lambda x: 'FR' if x== '' else x)
test['country'] = test['country'].apply(lambda x: loc_map[x])


#-------for embeddings------#

users_map = { v:k for k, v in dict(enumerate(users)).items() }
songs_map = { v:k for k, v in dict(enumerate(songs)).items() }
genres_map = { v:k for k, v in dict(enumerate(genres)).items() }
artists_map = { v:k for k, v in dict(enumerate(artists)).items() }

data['user_id'] = data['user_id'].apply(lambda x: users_map[x])
data['media_id'] = data['media_id'].apply(lambda x: songs_map[x])
data['genre_id'] = data['genre_id'].apply(lambda x: genres_map[x])
data['artist_id'] = data['artist_id'].apply(lambda x: artists_map[x])

age_norm = preprocessing.StandardScaler().fit(data['user_age'])
data['user_age'] = age_norm.transform(data['user_age'])

#---songs features--

# songs features
songs_feats = ['song','genre','artist','context','listen_type','platform_name_1','platform_family_1','platform_name_2','platform_family_2','is_listened']

feats = ['song','genre','artist','context','listen_type','platform_name_1','platform_family_1','platform_name_2','platform_family_2','country']

#---datapreppp

x_users = np.array(data['user_id']).reshape(data['user_id'].shape[0],1)
x_songs = np.array(data['media_id']).reshape(data['media_id'].shape[0],1)
x_genre = np.array(data['genre_id']).reshape(data['genre_id'].shape[0],1)
x_artist = np.array(data['artist_id']).reshape(data['artist_id'].shape[0],1)

#------history dataprepp---for lstm---#

from scipy import mean
#imputing unseen songs in test
songval = mean(list(songs_catmap.values()))
for i in set(test['media_id'])-set(data['media_id']):
	songs_catmap[i] = songval

#imputing unseen artists in test
artistval = mean(list(artists_catmap.values()))
for i in set(test['artist_id'])-set(data['artist_id']):
	artists_catmap[i] = artistval

test['context'] = test['context_type'].apply(lambda x:context_catmap[x])
test['song'] = test['media_id'].apply(lambda x: songs_catmap[x] if x in songs_catmap else songval) 
test['genre'] = test['genre_id'].apply(lambda x:genres_catmap[x]) 
test['artist'] = test['artist_id'].apply(lambda x:artists_catmap[x] if x in artists_catmap else artistval)
test['user_age'] = age_norm.transform(test['user_age'])
test = pd.get_dummies(test, columns=['platform_name', 'platform_family'], drop_first=True)

test['user_id'] = test['user_id'].apply(lambda x: users_map[x])
test['media_id'] = test['media_id'].apply(lambda x: songs_map[x])
test['genre_id'] = test['genre_id'].apply(lambda x: genres_map[x])
test['artist_id'] = test['artist_id'].apply(lambda x: artists_map[x])

test_user_index = { v:k for k, v in test['user_id'].to_dict().items() }

def userprepp(data,lookback):
	global test_user_index
	# unique users
	users = data['user_id'].unique()
	users_song_history = {} # {current_index:[history_indexes]}
	test_users_song_history = {}
	for ix,user in enumerate(users):
		print('user history prepp - ',ix)
		insts=data.loc[data['user_id']==user]
		#---for train data
		# index tells you how many above
		idx = insts.sort_values(by = ['ts_listen'], ascending=True).index.tolist()
		idx_new = [7558834]*lookback+idx # 123 is a zero-embedding-id
		for i in range(lookback,len(idx_new)):
			users_song_history[idx[i-lookback]]=idx_new[i-lookback:i]
		#---for test data
		#idx_t = insts.sort(['ts_listen'], ascending=True)[:lookback].index.tolist()
       #idx_t = insts[:lookback].index.tolist()
		idx_t = idx[-lookback:]
		test_users_song_history[test_user_index[user]] = [7558834]*(lookback-len(idx_t))+idx_t
	return (users_song_history, test_users_song_history)

lookback=4
users_song_history, test_users_song_history = userprepp(data,lookback)
null = pd.DataFrame([[-1]*len(songs_feats)],columns=songs_feats) #for padding with -1
dtx = data[songs_feats]
dtx = dtx.append(null,ignore_index=True)
dt=[]
for index in range(len(users_song_history)):
	print('train prepp - ', index)
	dt.append(np.array(dtx.iloc[users_song_history[index]]))
lstm_data=np.array(dt)

np.save('lstm.npy', lstm_data)
#a = np.load('lstm.npy')
#---test lstm dataprepp--
ldt = []
for index in range(len(test_users_song_history)):
	print('test prepp - ',index)
	ldt.append(np.array(dtx.iloc[test_users_song_history[index]]))
test_lstm = np.array(ldt)

np.save('lstm_test.npy', test_lstm)



#---test datapreppp

test_users = np.array(test['user_id']).reshape(test['user_id'].shape[0],1)
test_songs = np.array(test['media_id']).reshape(test['media_id'].shape[0],1)
test_genre = np.array(test['genre_id']).reshape(test['genre_id'].shape[0],1)
test_artist = np.array(test['artist_id']).reshape(test['artist_id'].shape[0],1)


#---architecture---#

users_emb = Sequential()
users_emb.add(Embedding(len(users), 70, input_length=1))
users_emb.add(Flatten())

songs_emb = Sequential()
songs_emb.add(Embedding(len(songs), 40, input_length=1))
songs_emb.add(Flatten())

genre_emb = Sequential()
genre_emb.add(Embedding(len(genres), 40, input_length=1))
genre_emb.add(Flatten())

artist_emb = Sequential()
artist_emb.add(Embedding(len(artists), 40, input_length=1))
artist_emb.add(Flatten())


#---interaction user-song
users_emb_1 = Sequential()
users_emb_1.add(Embedding(len(users), 40, input_length=1))
users_emb_1.add(Flatten())

songs_emb_1 = Sequential()
songs_emb_1.add(Embedding(len(songs), 40, input_length=1))
songs_emb_1.add(Flatten())

user_song = Sequential()
user_song.add(Merge([users_emb_1,songs_emb_1],mode='mul'))


#---interaction user-genre
users_emb_2 = Sequential()
users_emb_2.add(Embedding(len(users), 40, input_length=1))
users_emb_2.add(Flatten())

genre_emb_1 = Sequential()
genre_emb_1.add(Embedding(len(genres), 40, input_length=1))
genre_emb_1.add(Flatten())

user_genre = Sequential()
user_genre.add(Merge([users_emb_2,genre_emb_1],mode='mul'))


#---interaction user-artist
users_emb_3 = Sequential()
users_emb_3.add(Embedding(len(users), 40, input_length=1))
users_emb_3.add(Flatten())

artist_emb_1 = Sequential()
artist_emb_1.add(Embedding(len(artists), 40, input_length=1))
artist_emb_1.add(Flatten())

user_artist = Sequential()
user_artist.add(Merge([users_emb_3,artist_emb_1],mode='mul'))

#-----

features = Sequential()
features.add(Dense(len(feats), input_dim=len(feats)))

lstm_model = Sequential()
lstm_model.add(LSTM(128, input_dim=len(songs_feats), input_length=lookback))
lstm_model.add(Dense(50))

#---fused model---#
model = Sequential()
model.add(Merge([users_emb,songs_emb,genre_emb,artist_emb,user_song,user_genre,user_artist,features,lstm_model],mode='concat'))

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
#callbacks = [EarlyStopping(monitor='val_loss',patience=3,verbose=0)]

model.fit([x_users,x_songs,x_genre,x_artist, x_users,x_songs, x_users,x_genre, x_users,x_artist, np.array(data[feats]),lstm_data], y, batch_size=500, nb_epoch=3,validation_split=0.1)

preds = model.predict([test_users,test_songs,test_genre,test_artist, test_users,test_songs, test_users,test_genre, test_users,test_artist, np.array(test[feats]),test_lstm])

target = open('dsg_lstm7.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(preds)):
	target.write("{},{}\n".format(i,preds[i][0]))



"""

a =  pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/correl_ensemble.csv")
b =  pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/sub_hist_inrow.csv")
c = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/lstm+1.csv")
d = pd.read_csv("D:/Studies/Online Competitions/DSG/add.csv")
e = pd.read_csv("D:/Studies/Online Competitions/DSG/dsg_combine4.csv")

temp1 = np.array(a)
temp2 = np.array(b)
temp3 = np.array(c)
temp4 = np.array(d)
temp5 = np.array(e)

temp5 = []

for i in range(len(temp3)):
    temp5.append((temp1[i][1] + 0.5*temp2[i][1] + 1.5*temp3[i][1] + temp4[i][1])/4)

for i in range(10):
    print(temp5[i])
    
    
target = open('dsg_combine_lstm_max3.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(temp5)):
	target.write("{},{}\n".format(i,temp5[i]))

target.close()


"""

a =  pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/correl_ensemble.csv")
b =  pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/sub_hist_inrow.csv")
c = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/lstm+1.csv")
d = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/l2dist_waqar.csv")
e = pd.read_csv("D:/Studies/Online Competitions/DSG/dsg_combine4.csv")

temp1 = np.array(a)
temp2 = np.array(b)
temp3 = np.array(c)
temp4 = np.array(d)
temp5 = np.array(e)

d1=0
d2=0
d3=0
d4=0
d5=0

for i in range(len(temp5)):
    d1 += abs(temp1[i][1] - temp5[i][1])
    d2 += abs(temp2[i][1] - temp5[i][1])
    d3 += abs(temp3[i][1] - temp5[i][1])
    d4 += abs(temp4[i][1] - temp5[i][1])
    d5 += abs(temp5[i][1] - temp5[i][1])

a1 = []
a1.append(a["is_listened"].corr(a["is_listened"]))
a1.append(a["is_listened"].corr(b["is_listened"]))   
a1.append(a["is_listened"].corr(c["is_listened"]))   
a1.append(a["is_listened"].corr(d["is_listened"]))   
a1.append(a["is_listened"].corr(e["is_listened"]))   
t1 = sum(a1)/5
 
a2 = []
a2.append(b["is_listened"].corr(a["is_listened"]))
a2.append(b["is_listened"].corr(b["is_listened"]))   
a2.append(b["is_listened"].corr(c["is_listened"]))   
a2.append(b["is_listened"].corr(d["is_listened"]))   
a2.append(b["is_listened"].corr(e["is_listened"]))   
t2 = sum(a2)/5
        
a3 = []
a3.append(c["is_listened"].corr(a["is_listened"]))
a3.append(c["is_listened"].corr(b["is_listened"]))   
a3.append(c["is_listened"].corr(c["is_listened"]))   
a3.append(c["is_listened"].corr(d["is_listened"]))   
a3.append(c["is_listened"].corr(e["is_listened"]))   
t3 = sum(a3)/5
        
a4 = []
a4.append(d["is_listened"].corr(a["is_listened"]))
a4.append(d["is_listened"].corr(b["is_listened"]))   
a4.append(d["is_listened"].corr(c["is_listened"]))   
a4.append(d["is_listened"].corr(d["is_listened"]))   
a4.append(d["is_listened"].corr(e["is_listened"]))   
t4 = sum(a4)/5
        
a5 = []
a5.append(e["is_listened"].corr(a["is_listened"]))
a5.append(e["is_listened"].corr(b["is_listened"]))   
a5.append(e["is_listened"].corr(c["is_listened"]))   
a5.append(e["is_listened"].corr(d["is_listened"]))   
a5.append(e["is_listened"].corr(e["is_listened"]))   
t5 = sum(a5)/5

print(t1,t2,t3,t4,t5)

"""
w1 = (1+0.66467)/(1+t1)
w2 = (1+0.6626)/(1+t2)
w3 = (1+0.68)/(1+t3)
w4 = (1+0.66472)/(1+t4)
w5 = (1+0.67180)/(1+t5)
"""   

w1 = (1+0.66467)/(0.5+t1)
w2 = (1+0.6626)/(0.5+t2)
w3 = (1+0.68)/(0.5+t3)
w4 = (1+0.66472)/(0.5+t4)
w5 = (1+0.67180)/(0.5+t5)

sum1 = w1+w2+w3+w4+w5

l1 = w1/sum1
l2 = w2/sum1
l3 = w3/sum1
l4 = w4/sum1
l5 = w5/sum1

print(l1+l2+l3+l4+l5)
print(l1,l2,l3,l4,l5)

temp6 = []
for i in range(len(temp5)):
    temp6.append(l1*temp1[i][1]+ l2*temp2[i][1] +l3*temp3[i][1]+l4*temp4[i][1]+l5*temp5[i][1])

for i in range(10):
    print(temp6[i])

target = open('final_try.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(temp6)):
	target.write("{},{}\n".format(i,temp6[i]))

target.close()


x =  pd.read_csv("D:/Studies/Online Competitions/DSG/ds.csv")
y =  pd.read_csv("D:/Studies/Online Competitions/DSG/final_try.csv")

print(x["is_listened"].corr(y["is_listened"]))















		
		

