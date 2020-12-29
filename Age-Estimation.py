# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:49:30 2020

@author: Poorvahab
"""
#################################################### STEP 1

import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split

path_20_27='G:/deep learning/dataset face/class_20_50/20_27/'
path_28_34='G:/deep learning/dataset face/class_20_50/28_34/'
path_35_41='G:/deep learning/dataset face/class_20_50/35_41/'
path_42_50='G:/deep learning/dataset face/class_20_50/42_50/'
age_1=glob.glob(path_20_27+'*.jpg')
age_2=glob.glob(path_28_34+'*.jpg')
age_3=glob.glob(path_35_41+'*.jpg')
age_4=glob.glob(path_42_50+'*.jpg')

images_20_27=[]
labels_20_27=[]

for x in age_1:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float32')
    img=img/np.max(img)
    images_20_27.append(img)
    labels_20_27.append(24)
   
images_28_34=[]
labels_28_34=[]

for x in age_2:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float32')
    img=img/np.max(img)
    images_28_34.append(img)
    labels_28_34.append(31)
    
images_35_41=[]
labels_35_41=[]

for x in age_3:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float32')
    img=img/np.max(img)
    images_35_41.append(img)
    labels_35_41.append(38)
       
images_42_50=[]
labels_42_50=[]       
    
for x in age_4:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float32')
    img=img/np.max(img)
    images_42_50.append(img)
    labels_42_50.append(45)
        
images_20_27.extend(images_28_34)
images_20_27.extend(images_35_41)
images_20_27.extend(images_42_50)    
   
labels_20_27.extend(labels_28_34)
labels_20_27.extend(labels_35_41)
labels_20_27.extend(labels_42_50)

images=np.array(images_20_27)
labels=np.array(labels_20_27)

x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=None)

#################################################### STEP 2

from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv1D
from keras.optimizers import Adam
from keras.losses import mae

model=Sequential()
model.add(Conv1D(128,3,activation='relu',padding='same',strides=2,input_shape=(100,100)))
model.add(Conv1D(256,3,activation='relu',strides=2,padding='same'))
model.add(Conv1D(512,3,activation='relu',strides=2,padding='same')) 

model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(LSTM(50,activation='relu'))

model.add(Dense(1))

#################################################### STEP 3

model.compile(optimizer=Adam(),loss=mae)

#################################################### STEP 4

model.fit(x_train,y_train,epochs=30,validation_split=0.2)

loss,mae=model.evaluate(x_test,y_test)
print(f'loss is : {loss} accuracy is: {mae}')










    
    
    
    
    
    
    
    
    
    
    
    
    
    
    