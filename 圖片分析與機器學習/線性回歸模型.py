# 線性回歸:Y(X)=W0+W1X

# 多項式回歸:Y(X)=W0+W1X+W2X^2+.....WnX^n

# 多元回歸:在線性中的線->要用最小平方法求出，取兩個類別距離最近者

  # Y=a+b*X+e

  # a=>y軸的截距

  # b=>回歸斜距

  # e=>誤差


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#溫度與冰的走勢圖
temperatuers = np.array([29,28,34,31,25,29,32,21,24,33,25,31,26,30])
ice = np.array([77,62,93,84,59,64,80,75,58,91,51,73,65,84])


lr = LinearRegression()

lr.fit(np.reshape(temperatuers,(len(temperatuers),1)),np.reshape(ice,(len(ice),1)))  #len(XXX)=組數，且必須可以整除後面的常數

newTemp = np.array([30])
sales = lr.predict(np.reshape(newTemp,(len(newTemp),1)))
sales

plt.scatter(temperatuers,ice,color="blue")
plt.plot(temperatuers,lr.predict(np.reshape(temperatuers,(len(temperatuers),1))),color="b",linewidth=3)
plt.plot(newTemp,sales,color="r",marker="o",markersize=12)

# **油價與交通工具的關係**

import pandas as pd

x_values = pd.DataFrame([1,2,3,4])

y_values = pd.DataFrame([0,0.3,0.6,0.9])

x_test = pd.DataFrame([1.5,3,5])



body_reg = LinearRegression()
body_reg.fit(x_values,y_values)
y_test_predict = body_reg.predict(x_test)
y_test_predict

plt.scatter(x_values,y_values)
plt.scatter(x_test,y_test_predict,color="r")
plt.plot(x_test,y_test_predict,color="b")
plt.show()

# x軸=自變數

# y軸=應變數

# 線性分析的線=最佳配適線

# y1=b0+b1x1

# b0=y的截距

# b1=斜率

# # 台積電股票分析(線性回歸)

from google.colab import drive

drive.mount("/content/drive")

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


path = '/content/drive/MyDrive/視覺辨識/TSMC'
os.chdir(path)
file_list = os.listdir()
file_list.sort()
file_list

df = open(path+"/"+file_list[0])
data = df.readline()  #讀掉第一行
data = pd.read_csv(df)
data
data = data.drop(["日期","Unnamed: 9"],axis=1)
data = data.dropna()
data

allData = data
for i in range(1,13):
  df = open(path+"/"+file_list[i])
  data = df.readline()
  data = pd.read_csv(df)
  data = data.drop(["日期","Unnamed: 9"],axis = 1)
  data = data.dropna()
  allData = pd.concat([allData,data])

allData

x = np.linspace(0,1,len(allData))
y = np.array(allData.iloc[:,2]/np.mean(allData.iloc[:,2])-0.6)
plt.rcParams['figure.figsize']=(16,6)
plt.plot(x,y,'b-',linewidth=3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

df = open(path+"/"+file_list[-1])
data = df.readline()  #讀掉第一行
data = pd.read_csv(df)
data = data.drop(["日期","Unnamed: 9"],axis=1)
data = data.dropna()
test_x = np.linspace(len(x)+1,len(x)+len(data),len(data))*(x[1]-x[0])
test_y = np.array(data.iloc[:,2]/np.mean(allData.iloc[:,2])-0.6)


print(len(x))
print(len(data))
t_x = np.linspace(len(x)+1,len(x)+len(data),len(data))
print(t_x)
print(x[1])
print(x[0])


# 線性回歸方程式

# ```
# # h(x) = w0+w1x

# ```
# 線性代數
# ```
# # 權重X Ax=B

# ```
# 線性代數
# ```
# # 權重X Ax=B

# ```
# 將兩邊同乘
# A^TAx = A^TB =>x=(A^TA)^-1 A^TB


def refresh(X,y,w,a):
  dJ=(X.dot(w.T)-y).dot(X)/len(y) #權重公式

  newW = w-a*dJ

  return newW
#多項式線性迴歸
def Polynomial_regression(s,x,y,test_x,test_y):
  #n n次特徵轉換
  #a :學習速度參數
  #T :更新次數
  #w :權重
  #X :資料矩陣
  #.dot: 向量內積
  n = s+1
  a = 1
  T = 1000
  X = np.zeros((len(x),n))
  #產生資料矩陣
  for i  in range(n):
    X[:,i] = x ** i
  #初始化權重
  w=(np.linalg.inv((X.T).dot(X)).dot(X.T)).dot(y) #inv=逆矩陣:整個加出來的是偶數，基數會發生例外。(array內的加總合)
  #紀錄W產生的預估值
  plot_yy = X.dot(w.T)
  for t in range(T):
    w =refresh(X,y,w,a)
  plot_y = X.dot(w.T)
  #計算相差多少
  error = 0
  for k in range(len(y)):
    error += abs(plot_y[k]-y[k])
  error = error/len(y)
  print('訓練錯誤有:',error)
  #產生測試集資料矩陣
  test_X = np.zeros((len(test_x),n))
  #產生資料矩陣
  for i in range(n):
    test_X[:,i]=test_x ** i
  pred_error=0
  pred_y = test_X.dot(w.T)
  #計算測試集的錯誤率
  for k in range(len(pred_y)):
    pred_error += abs(pred_y[k]-test_y[k])
  pred_error = pred_error/len(test_y)
  print('測試錯誤有:',pred_error)
  #繪圖
  plt.rcParams['figure.figsize'] = (10,10)
  plt.plot(x,plot_y,'g-',linewidth=6,label='fitting')
  plt.plot(test_x,pred_y,'g-',linewidth=6)
  plt.plot(x,y,'b-',linewidth=3,label='training data')
  plt.plot(test_x,test_y,'r-',linewidth=3,label='testing data')
  plt.title('TSMC:'+str(n-1)+'feature')
  plt.legend(loc=6,fontsize=20)
  plt.grid()
  plt.show()
  return error,pred_error


errorx = np.zeros(20)
errory_train = np.zeros(20)
errory_test = np.zeros(20)
for i in range(1,21):
  errorx[i-1] = i
  errory_train[i-1] , errory_test[i-1] = Polynomial_regression(i,x,y,test_x,test_y)

plt.rcParams['figure.figsize'] = (10,10)
plt.plot(errorx , errory_train,'r-o', linewidth = 3 , markersize = 10 , label='training error')
plt.plot(errorx , errory_test,'g-o', linewidth = 3 , markersize = 10 , label='testing error')
plt.xlabel('power')
plt.ylabel('error')
plt.legend(loc=6)
plt.grid()
plt.show()




#練習

a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
a1

b1 = np.array([[10,20,30],[40,50,60],[70,80,90]])

a1+b1

a1-b1

a1*b1

a1/b1

#向量內積

a2 = np.array([[2,1],[3,4]])
b2 = np.array([[5,6],[7,8]])


a2

b2

a2.dot(b2)
#內積: c = ab
#C(1,1) = 2*5 + 1*7 =17
#C(1,2) = 2*6 + 1*8 =20
#C(1,3) = 3*5 + 4*7 =43
#C(1,4) = 3*6 + 4*8 =50

b2.dot(a2)



# 干擾因子

import numpy as np
import matplotlib.pyplot as plt


#隨機種子，值會一樣
np.random.seed(0)
#高斯分布:平均為0、變異數為1(標準差)。
np.random.randn(100)


np.random.seed(0)
time = np.arange(1,101)
#股價
stock_price = 50 + 2 * time + np.random.randn(100) * 5

#隨機產生干擾因子
interface = np.random.randn(100) * 5

#實際價格
interface_stock_price = stock_price + interface

#繪圖
plt.figure(figsize = (10,6))
plt.plot(time,stock_price,label='stock price')
plt.plot(time,interface_stock_price,label='interface')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend(loc=6)
plt.title('Stock')
plt.show()




from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(time.reshape(-1,1),interface_stock_price)
info = model.coef_[0]
info  #係數


info = model.coef_[0]
info  #係數

intercept = model.intercept_
intercept #截距

print(f"stock_price={info:.2f}*time+{intercept:.2f}")



# 羅吉斯回歸

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


iris = datasets.load_iris()
x = pd.DataFrame(iris['data'],columns=iris['feature_names'])
print(iris['feature_names'])
y = pd.DataFrame(iris['target'],columns=['target'])
iris_data = pd.concat([x,y],axis=1)


iris_data = iris_data[['sepal length (cm)', 'petal length (cm)','target']]
iris_data = iris_data[iris_data['target'].isin([0,1])]
iris_data.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(iris_data[['sepal length (cm)', 'petal length (cm)']],iris_data[['target']],test_size=3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train) #std=標準差
X_test_std = sc.transform(X_test)
#將所有特徵標準化(高斯分布)，使得數據平均為零、方差唯一。適合使用時機，於當有些特徵方差過大時，使用標準化能夠有效地讓模型快速收斂。
#transform使用該類的好處在於可以保存訓練集中的參數(均質和等差)直接使用其對象轉換測試集數據


X_train_std

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_std,y_train['target'].values)

lr.predict(X_train_std)


y_test['target'].values

lr.predict_proba(X_train_std)



#範例
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sx = [[5.0,3.5],[5.5,3.8],[5.0,1.0],[5.5,1.4]]
sc.fit(sx)
x_std = sc.transform(sx)
x_std

#繪圖
from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
  markers = ('s','x','o','^')
  colors = ('red','blue','gray','cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])
  x1_min,x1_max = X[:,0].min-1,X[:,0].max+1
  x2_min,x2_max = X[:,1].min-1,X[:,1].max+1
  xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
  Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
  plt.xlim(xx1.min(),xx1.max())
  plt.xlim(xx2.min(),xx2.max())
  for idx,cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)
    plot_decision_regions(X_train_std,y_train['target'].values,classifier=lr)
    plt.xlabel('sepal length')
    plt.ylabel('petal lengeh')
    plt.legend(loc=6)
    plt.tight_layout()
    plt.show





