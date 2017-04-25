
# coding: utf-8

# In[ ]:

#PCA的python实现
import numpy as np
import pandas as pd

Class PLA_(object):
    
    #初始化
    def __init__(self, percentage):
        self.percentage = percentage
    
    #标准化
    def zeresmeans(self, datemat):
        averrage = np.means(datemat, axis = 0)
        new_date = datemat - averrage
        return new_date
    
    #求出协方差矩阵
    def cov_date(self, new_date):
        covdate = np.cov(new_date, rowvar = 0)
        return covdate
    
    #求出特征值和特征向量
    def chracters(self, covdate)：
        eigVals,eigVects=np.linalg.eig(np.mat(covdate))  
        return eigVals,eigVects
    
    #求出降维后的原始数据
    def cal(self):
        eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
        n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
        n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
        lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
        reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
        return lowDDataMat,reconMat 
    
    #求n
    def argsort(self, covdate):
        sortArray = np.sort(covdate)
        sortArray = sortArray[-1::-1]
        sumarray = sortArray.sum()
        self.percentage = 0.99
        n = 0
        for i in ndarray:
            tmpsum = tmpsum + i
            if tmpsum >= sumArray * self.percentage:
                n = np.where(sortArray == i)
                return n 
            

