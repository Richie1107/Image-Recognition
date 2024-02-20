import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential  #神經網絡
from tensorflow.keras.utils import to_categorical #分類物件
from tensorflow.keras.layers import Dense,Activation #Dense=連階層 Activation=激勵函式
from tensorflow.keras.datasets import mnist #手寫辨識資料集
from tensorflow.keras.optimizers import SGD #優化器 (梯形修正)



(x_train,y_train),(x_test,y_test) = mnist.load_data()
len(x_train)




len(x_test)

y_train[100]

plt.imshow(x_train[100],cmap='binary')
#白底黑字

x_train[0]

x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000,-1)


x_train[0]

x_train.shape

#說明
n = np.arange(6)
print(n)

r=n.reshape((3,2))
r

#當值太大時，請收斂
x_train = x_train/255
x_test = x_test/255

x_train[0]

y_test

#OneHot
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

y_train[0]

#建模
model = Sequential()
model.add(Dense(20,activation='relu',input_dim=784))
model.add(Dense(40,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(10,activation='softmax')) #多元分類
model.summary() #摘要

# 神經元算法

# 20*784+20= 15700 第一層算法

# 20*40+40 = 840  第二層算法

# 40*80+80 = 3280 第三層算法

# 80*10+10 = 810 第四層算法




#編譯
model.compile(optimizer='adam',loss='mse',metrics=['accuracy']) #optimizer=優化器 loss=平均方差 metrics=正確度


#訓練
model.fit(x_train,y_train,validation_batch_size=0.2,batch_size=100,epochs=30) #batch_size=長度,epochs=訓練回數

#預測
predict = model.predict(x_test)


predict[0]

np.argmax(predict[0])

y_test[0]













