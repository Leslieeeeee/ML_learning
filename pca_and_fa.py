
# coding: utf-8

# In[3]:

import numpy as np

def zero_mean(datemat): 
    averrage = np.mean(datemat, axis = 0)
    standard_date = datemat - averrage
    return standard_date,averrage  
    
    #求出协方差矩阵
    cov_date = np.cov(standard_date, rowvar = 0)

    
    #求出特征值和特征向量
    
    
    """
    #计算方差贡献率
    def cal_drate(self):
        rate_all = []
        for i in eigVal:
            rate = i / eigVal[0:n].sum()
            rate_all.append(rate)
        return rate_all
    """
    """
    #因子旋转
    def rotate_date(self):
        v_date = np.array[]
        ro_date = np.corrcoef(new_date)
        z_ date = new_date * ro_date
        #正交旋转
        rotate_date = v_date * ro_date
        for i <= n:
            for i <= n:
                for i <= n:
                    ai_1 = ai_1^2 + ai_1
                    ai_2 = ai_2^2 + ai_1
                u = (ai_1/h)^2 + (ai_2/h)^2
            a = a+ u

             for i < n:
                for i < n:
                    ai_1 = ai_1^2 + ai_1
                    ai_2 = ai_2^2 + ai_1
                v = 2*ai_1*ai_2/h
            b = b + (u^2 - v^2)
                d_ = u*v
            
            d_ = u*v + d
            d = 2 * d_       
        
        fi = atan((d - 2*a*b/n))/(4*(c - (a^2 - b*2))/n)

    """
        
        
    #求出降维后的原始数据
    def cal(self, datemat, n):
        standard_date, averrage = zero_mean(datemat)
        
        #求出协方差矩阵
        cov_date = np.cov(standard_date, rowvar = 0)
        
        #求出特征值
        eigVals,eigVects=np.linalg.eig(np.mat(cov_date))       
        eigValIndice = np.argsort(eigVals)            #对特征值从小到大排序  
        n_eigValIndice = eigValIndice[-1: -(n+1): -1]   #最大的n个特征值的下标  
        n_eigVect = eigVects[:, n_eigValIndice]        #最大的n个特征值对应的特征向量  
        lowDataMat = newData * n_eigVect               #低维特征空间的数据  
        reconMat = (lowDataMat * n_eigVect.T) + meanVal  #重构数据  
        return lowDataMat, reconMat 
        print(lowDataMat, reconMat)

    #求n(即公共因子数)
    def argsort(self, cov_date):
        sortArray = np.sort(cov_date)
        sortArray = sortArray[-1::-1]
        sumarray = sortArray.sum()
        percentage = 0.99
        n = 0
        for i in ndarray:
            tmpsum = tmpsum + i
            if tmpsum >= sumArray * self.percentage:
                n = np.where(sortArray == i)
                return n
            


# In[5]:

data = [[-1, -1, 0, 2, 0],[-2, 0, 0, 1, 1]]
datemat = np.array(data)

s = zero_mean(datemat)
print(s)


# In[ ]:



