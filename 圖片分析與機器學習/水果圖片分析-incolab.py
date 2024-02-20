import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,Activation,Conv2D,MaxPooling2D
from tensorflow.python.keras.utils import np_utils

#掛接雲端硬碟
from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/photo_py/水果30分類/train'
files = os.listdir(path)
directories = list()

for dir in files:
  if os.path.isdir(os.path.join(path,dir)):
    directories.append(dir)
directories

def resizeFile(dir_path,img_files):
  size = (250,250)
  for item in img_files:

    item = os.path.join(dir_path,item)

    images = cv2.imread(item)
    h,w = images.shape[:2]
    if h > size[0] or w > size[1]:
      ratio = max(h/size[0],w/size[1])
      images = cv2.resize(images,(int(w/ratio),int(h/ratio)))

    h,w = images.shape[:2]
    pad_h = size[0] - h
    pad_w = size[1] - w
    top,bottom = pad_h // 2 , pad_h - (pad_h//2)
    left,right = pad_w // 2 , pad_w - (pad_w//2)

    images = cv2.copyMakeBorder(images,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
    name = item.split('/')[-1]

    cv2.imwrite(f"{dir_path}/resize/{name}",images)

x_train = list()
y_train = list()

def create_data(path,category):
  for p in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,p),0)
    x_train.append(img_array)
    y_train.append(category)

category= 0
for directory in directories:
  dir_path = os.path.join(path,directory)
  dir_file = os.listdir(dir_path) #讀取資料夾內的檔案
  img_files = list()
  for files in dir_file:
    if files.endswith('.jpg') or files.endswith('.png') or files.endswith('.jpeg'):
      img_files.append(files)
  resizeDir = os.path.join(dir_path,'resize')
  if not os.path.exists(resizeDir):
    os.makedirs(resizeDir)
    resizeFile(dir_path,img_files)
  create_data(resizeDir,category)
  category += 1

len(x_train)

x_train = np.array(x_train).reshape(-1,250,250,1)
y_train = np.array(y_train)
x_train = x_train / 255   #色階
y_train = np_utils.to_categorical(y_train)

len(y_train[0])

#建模
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024,activation='relu'))
model.add(Dense(29,activation='softmax'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#打散
from sklearn.model_selection import train_test_split

rX_train,rX_test,ry_train,ry_test = train_test_split(x_train,y_train,test_size=1,random_state=1)

model.fit(rX_train,ry_train,epochs=10,batch_size=50,validation_split=0.25)

#其他訓練方式
#model.fit(rX_train,ry_train,epoch=10,batch_size=100,validation_data=(x_test,y_test))
#

predict1 = model.predict(rX_train)

right = 0
for i in range(len(predict1)):
  ans = np.argmax(predict1[i])
  pre = ry_train[i][ans]
  if pre ==1:
    right += 1

right / len(predict1)





