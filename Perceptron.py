#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
