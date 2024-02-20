import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,Activation,Conv2D,MaxPool2D
from tensorflow.python.keras.utils import np_utils

train_dir = "/content/drive/MyDrive/photo_py/train"
path = os.path.join(train_dir)
path

X = list()
y = list()

convert = lambda category:int(category == "dog")  #適用於只有兩個種類時，0、1是布林直

classes = convert("dog")
classes

classes = convert("cat")
classes

#圖片整理




def create_class(path):
  for p in os.listdir(path):
    category = p.split(".")[0]
    category = convert(category)
    img_arr = cv2.imread(os.path.join(path,p),0)
    new_img = cv2.resize(img_arr,dsize=(100,100))
    X.append(new_img)
    y.append(category)

create_class(path)

len(X)

len(y)

len(X[0])

X[1]

X = np.array(X).reshape(-1,100,100,1)
y = np.array(y)

X = X/255
y_train = np_utils.to_categorical(y)

#建模





model = Sequential()
#第一層
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:],activation="relu"))  #卷基
model.add(MaxPool2D(pool_size=(2,2)))   #池化
#第二層
model.add(Conv2D(128,(3,3),activation="relu"))  #卷基
model.add(MaxPool2D(pool_size=(2,2)))   #池化
#第三層
model.add(Conv2D(256,(3,3),activation="relu"))  #卷基
model.add(MaxPool2D(pool_size=(2,2)))  #池化

#扁平
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024,activation="relu"))
#model.add(Dense(1,activation="sigmoid"))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(X,y_train,epochs=20,batch_size=32)

predict = model.predict(X)

ans = np.argmax(predict[200])
ans

y_train[200]

val = [int(round(p[0])) for p in predict]
val


y_test = list()
for t in y_train:
  if t[0] == 0:
    y_test.append("cat")
  else:
    y_test.append("dog")

df = pd.DataFrame({"id":y_test,"label":val})
df

X_test = list()
path = "/content/testCat.jpg"
img_arr = cv2.imread(os.path.join(path),0)
new_img = cv2.resize(img_arr,dsize=(100,100))
X_test.append(new_img)
X_test = np.array(X_test).reshape(-1,100,100,1)
X_test = X_test/255

pre = model.predict(X_test)

np.argmax(pre[0])

 #上傳圖片




import os
from google.colab import files
#由本地端上傳
def upload():
  uploaded = files.upload()
  for name,data in upload.items():
    with open(name,"wb") as f:
      f.write(data)
      print("上傳成功")





