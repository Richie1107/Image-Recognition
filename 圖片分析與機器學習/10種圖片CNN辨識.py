from keras.datasets import cifar10
from tensorflow.python.keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
#載入

len(x_train)

len(x_test)

x_train.shape

x_train[0]

plt.imshow(x_train[1111],cmap='binary')

y_test[1111]

#收斂
x_train_normal = x_train.astype('float32')/255
x_test_normal = x_test.astype('float32')/255

x_train[0]

#OneHot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

y_train[0]

# 建模
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  #conv2D:捲基層，MaxPooling2D:磁化層，Dropout:除去過多的，Dense:打平用
from keras.layers import Activation #激勵函式

#第一層
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
#filters=特徵點，針對圖片每一個輸出點。kernel_size=卷積核，padding=遇到邊界給予補零。activation=激勵函示，修正用，負數變成零。
model.add(MaxPooling2D(pool_size=(2,2)))
#第二層
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) #丟棄的比例
model.add(Flatten())  #扁平
model.add(Dense(1024,activation='relu'))  #連接層
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_history = model.fit(x_train_normal,y_train,batch_size=128,epochs=30)

def showHistory(train_history,train,validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train History')
  plt.xlabel('Epoch')
  plt.ylabel(train)
  plt.legend(['train','validation'],loc=0)
  plt.show()

showHistory(train_history,'accuracy','loss')

predict = model.predict(x_test_normal)


print(np.argmax(predict[3]))

y_test[3]

anslabel={0:'airplane',1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'boart',9:'trunk'}

def plot_predict(images,labels,predict,idx,num=25):
  fig = plt.gcf()
  fig.set_size_inches(14,16)
  for i in range(num):
    ix = plt.subplot(5,5,i+1)
    ix.imshow(images[idx],cmap='binary')
    lab=np.argmax(labels[i])
    title = str(i)+"-"+anslabel[lab]
    ans = np.argmax(predict[i])
    title += "==>"+anslabel[ans]
    ix.set_title(title,fontsize=8)
    idx += 1
  plt.show()

plot_predict(x_test,y_test,predict,0)

#存檔model
model.save('tenPhoto.h5') #h5=hdf5格式


#載入
import tensorflow
md = tensorflow.keras.models.load_model('/content/tenPhoto.h5')
pred = md.predict(x_test_normal)

plot_predict(x_test,y_test,pred,0)











