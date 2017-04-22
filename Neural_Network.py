
# coding: utf-8

# In[3]:

#改变数据文件格式
import os
def change_name(self, newname):
    
    path= 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.xls'
    newpath = self.newname
    if path.find('.') < 0:
       path.replace(path, newpath)
    print (newpath,'ok')


# In[389]:

#!/usr/bin/env python3

"""
Created on Sun Apr 16 22:18:32 2017

@author: leslieeeeee
"""
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Neu_Net(object):
    #初始化函数
    def __init__(self, rate = 0.01, n = 50):
        self.rate = rate
        self.n = n
        
    #输入样本和一个期望输出
    #初始化权重向量
    def Input_w(self, x, target):
        #x:shape[n_samples,n_features]
        #初始化为0
        self.w = np.zeros(1 + x.shape[1])
        self.const_ = []
        #重复训练n次
        for i in range(self.n): 
            k = self.Cal(x)
            #求出误差并与期望输出相比较
            new_e= (target - k)              
            self.w[0] += self.rate * new_e.sum()
            #求出权值变化值再更新
            self.w[1:] += self.rate * x.T.dot(new_e)
            cost = (new_e ** 2).sum()/2.0
            self.const_.append(cost)
        return self
    
   
    #计算输出
    def Cal(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]
        #y = self.w[1:].np.T * x + self.w[0]
        
    def activation():
        pass
    
    def sort(self, x):
        return np.where(self.Cal(x) >= 0.0 , -1, 1)


# In[390]:

#如何设置合理的resolution
def plot_decision_regions(x, y, classifer, resolution = 0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'green', 'gray') 
    #从颜色集中选择相应数目的元素
    cmap = ListedColormap(colors[:len(np.unique(colors))])

    #找到输入数据的最大最小值
    x0_min = x[:, 0].min() - 1
    x0_max = x[:, 0].max()
    x1_min = x[:, 1].min() - 1
    x1_max = x[:, 1].max()
    #构建的训练集(分类)
    [x0, x1]= np.meshgrid(np.arange(x0_min, x0_max, resolution),np.arange(x1_min, x1_max, resolution))
    
    z = classifer.sort(np.array([x0.ravel(), x1.ravel()]).T)
    print(z)
    
    #绘图
    z = z.reshape(x0.shape)
    plt.contourf(x0, x1, z, alpha=0.4, cmap = cmap)    
    plt.xlim(x0.min(), x0.max())
    plt.ylim(x1.min(), x1.max())      
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], alpha=0.8, c=cmap(idx), marker=marker[idx], label=cl)


# In[391]:

#构造训练数据集
file ='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(file, header = None, encoding="ISO-8859-1")
#xx = df.loc[7:106, [11, 25]].values
#y = df.loc[7:106, 13].values
#y = np.where(y == 'PG-13',-1, 1)
#x = xx.astype('float')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0,2]].values
x_std = np.copy(x)
x_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

#print(x)
#print(y)
# 使用散点图可视化样本
plt.scatter(x[:50, 0], x[:50,1], color='red', marker='o', label='setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('Total time')
plt.ylabel('IMDB Mark')
plt.legend(loc='upper left')
plt.show()

#训练
work = Neu_Net(rate = 0.001, n = 0)
work.Input_w(x_std, y)
                        
#绘图
plot_decision_regions(x_std, y, classifer=work, resolution=0.02)
plt.title('Correlation of Movie Booking Office and IMDB Mark')
plt.xlabel('IMDB Mark')
plt.ylabel('Movie Booking Office')
plt.legend(loc = 'upper left')
plt.show()  


# In[12]:

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file ="/Users/leslieeeeee/Documents/movie_data.csv"

df = pd.read_csv(file, header = None, encoding="ISO-8859-1")
xx = df.loc[7:4007, [25, 26]].values
yy = df.loc[7:106, 13].values
x = xx.astype('float')
y = yy.astype('float')
#print(x)
#print(x.shape)
#print(x.ndim)
#print(x[:, 0])
#m = np.zeros(1 + x.shape[0])
#print(m)

#marker参数有固定取值
plt.scatter(x[:2000, 0], x[:2000, 1], color='red', marker='o', label='xxx')
plt.scatter(x[2001:4000, 0], x[2001:4000, 1], color='blue', marker='x', label='qqq')
plt.title('Correlation of IMDB Mark and Movie Booking Office')
plt.xlabel('IMDB Mark')
plt.ylabel('Movie Booking Office')
plt.legend(loc = 'upper left')
plt.show()  


# In[ ]:




# In[ ]:



