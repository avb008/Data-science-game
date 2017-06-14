# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:08:35 2017

@author: vushesh
"""

import pandas as pd
import numpy as np
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Merge
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
"""

a1 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/2.csv")
a2 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/abhilash_0.64867.csv")
a3 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/Best_0.65193.csv")
a4 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/dist_ensemble.csv")
a5 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/ensemble_0.66041.csv")
a6 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/ensemble2_0.65992.csv")
a7 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/ensemble3_0.65946.csv")
a8 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/ensemble5_0.66238.csv")
a9 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/ensemble6_0.66104.csv")
a10 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/ensemble7_0.66147.csv")
a11 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/final.csv")
a12 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/l1dist_ensemble.csv")
a13 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/l2dist_ensemble.csv")
a14 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/newemb_0.64321.csv")
a15 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/newemb_0.65176.csv")
a16 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/newemb_without_emb2emb3_0.64190.csv")
a17 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/newidea_with location_0.64805.csv")
a18 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/only 2 with media duration.csv")
a19 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/sub_dsg.csv")
a20 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/sub_dsg_cont.csv")
a21 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/sub_dsg_newemb.csv")
a22 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/Submission7_0.65876.csv")
a23 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/Submission9_0.65995.csv")
a24 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/Submission13_0.6626.csv")
a25 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/Submission11_0.65667.csv")
a26 = pd.read_csv("D:/Studies/Online Competitions/DSG/Ensemble_folder/correl_ensemble.csv")

xr = {}
xr[1] = a1
xr[2] = a2
xr[3] = a3
xr[4] = a4
xr[5] = a5
xr[6] = a6
xr[7] = a7  
xr[8] = a8
xr[9] = a9
xr[10] = a10
xr[11] = a11
xr[12] = a12
xr[13] = a13
xr[14] = a14
xr[15] = a15
xr[16] = a16
xr[17] = a17
xr[18] = a18
xr[19] = a19 
xr[20] = a20
xr[21] = a21
xr[22] = a22
xr[23] = a23
xr[24] = a24
xr[25] = a25
xr[26] = a26

value = a5['is_listened'] + a8['is_listened'] + a9['is_listened']  + a10['is_listened'] + a12['is_listened'] + a13['is_listened'] + a26['is_listened']
value = pd.DataFrame(value ,columns=["sample_id","is_listened"]) 
value["sample_id"] = a1["sample_id"].apply(lambda x: x)
value['is_listened'] = value['is_listened'].apply(lambda x: x/29)
value.head()
temp12 = np.array(value)

for i in range(1,26):
    value = xr[26]["is_listened"].corr(xr[i]['is_listened'])
    if value < 0.90:
        print(i,value)


temp = {}
temp.append(np.array(a1))
temp1 = np.array(a1)
temp2 = np.array(a2)
temp3 = np.array(a11)
temp4 = np.array(a14)
temp5 = np.array(a16)
temp6 = np.array(a17)
temp7 = np.array(a21)
temp8 = np.array(a26)


d = []
l = []
w = []
r = [] 
for i in range(8):
    d.append(0)
    w.append(1)
    l.append(0)
    r.append(1)
    
for i in range(len(temp8)):
    d[0] += (temp1[i][1] - temp8[i][1])*(temp1[i][1] - temp8[i][1])
    d[1] += (temp2[i][1] - temp8[i][1])*(temp2[i][1] - temp8[i][1])
    d[2] += (temp3[i][1] - temp8[i][1])*(temp3[i][1] - temp8[i][1])
    d[3] += (temp4[i][1] - temp8[i][1])*(temp4[i][1] - temp8[i][1])
    d[4] += (temp5[i][1] - temp8[i][1])*(temp5[i][1] - temp8[i][1])
    d[5] += (temp6[i][1] - temp8[i][1])*(temp6[i][1] - temp8[i][1])
    d[6] += (temp7[i][1] - temp8[i][1])*(temp7[i][1] - temp8[i][1])
    d[7] += (temp8[i][1] - temp8[i][1])*(temp8[i][1] - temp8[i][1])
   
   
for i in range(len(temp8)):
    l[0] += abs(temp1[i][1] - temp8[i][1])
    l[1] += abs(temp2[i][1] - temp8[i][1])
    l[2] += abs(temp3[i][1] - temp8[i][1])
    l[3] += abs(temp4[i][1] - temp8[i][1])
    l[4] += abs(temp5[i][1] - temp8[i][1])
    l[5] += abs(temp6[i][1] - temp8[i][1])
    l[6] += abs(temp7[i][1] - temp8[i][1])
    l[7] += abs(temp8[i][1] - temp8[i][1])
   
 
x = []
x.append(0.64175)
x.append(0.64867)
x.append(0.64520)
x.append(0.64321)
x.append(0.64190)
x.append(0.64805)
x.append(0.64338)
x.append(0.66467)

import math 
   
for i in range(8):
    d[i] = math.sqrt(d[i])
    w[i] = 1/(1+d[i])
    r[i] = 1/(1+l[i])
    

sum1 = 0
sum2 = 0

for i in range(8):
    sum1 += w[i]
    sum2 += r[i]

for i in range(8):
    w[i] = w[i]/sum1    
    r[i] = r[i]/sum2


temp11= []
temp12 = []
for i in range(len(temp8)):
    x = w[0]*temp1[i][1] + w[1]*temp2[i][1] +  w[2]*temp3[i][1] + w[3]*temp4[i][1] + w[4]*temp5[i][1] + w[5]*temp6[i][1] +w[6]*temp7[i][1] +w[7]*temp8[i][1]
    y = r[0]*temp1[i][1] + r[1]*temp2[i][1] +  r[2]*temp3[i][1] + r[3]*temp4[i][1] + r[4]*temp5[i][1] + r[5]*temp6[i][1] +r[6]*temp7[i][1] +r[7]*temp8[i][1] 
    temp11.append(x)
    temp12.append(y)
    
    
target = open('l2dist_waqar.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(temp11)):
	target.write("{},{}\n".format(i,temp11[i]))   
    
target.close()   

target = open('add.csv','w')
target.write("sample_id,is_listened\n")
for i in range(len(temp12)):
	target.write("{},{}\n".format(i,temp12[i][1]))   
    
target.close()             
            
            
            
            
            
            
            
            
            