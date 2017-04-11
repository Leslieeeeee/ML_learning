
# coding: utf-8

# In[ ]:

import numpy as np
class Perceptron(object):
    #初始化函数
    def __int__(self, rate = 0.01, n = 10):
        self.rate = rate
        self.n = n
        
    #输入样本和一个期望输出
    #初始化权重向量
    def Input_w(self, x):
        #x:shape[n_samples,n_features]
        xi = x.shape(1)
        self.w = np.zeros(1 + xi)
        w = self.w
        times = 0
        
        #重复训练十次
        for x in range(self.n):  
            
            #求出误差并与期望输出相比较
            #更新权值
            for x, target in add_up(x,y):
                new_e = self.rate * (target - Cal(xi))
                for i in w:
                    w[0] += new_e
                    w[1:] = new_e * xi
                
                if new_e = 0.0:
                    break
                else:
                    times ++
            print "This new curve is：%d", y
            print "err times is: %d", times
            
             
            
    
    #计算各层输出
    def Cal(self, x):
        y_before = w[1:].np.T * x
        # y = np.dot(w[1:], x) + w[0]
        y_mid = y_before.sum()
        if y_mid >= 0.0:
            y = 1
        else:
            y = -1
        return y
    
    def add_up(self, x, y):
        self.x = x 
        self.y = y
        hstack(x, y)
        

