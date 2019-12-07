# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:58:29 2019

@author: asus
"""
import pandas as pd
import numpy as np
####封装

def pre_deal(data_period = 5, dif_preiod = 1):
    ###预处理函数： 将数据分为y1和y3数据然后进行存储整理成数表形式并返回
    
    ##读文件
    data = pd.read_csv('../data/financial data (.csv project).csv')

    ################################处理y1########################
    data_y1 = pd.DataFrame(data['y1'])
    data_y1 = data_y1.diff(1)

    ##构建数表
    i = 1
    while i <= data_period:
        if i != data_period:
            data_y1['X' + str(i)] = data_y1['y1'].shift(-i) 
        else:
            data_y1['X' + str(i)] = 0
            for j in range(0,dif_preiod):
                data_y1['X' + str(i)] = data_y1['X' + str(i)] + data_y1['y1'].shift(-i-j) 
        i = i + 1
    ##删除nan
    data_y1 = data_y1.dropna()
    ##更改column name
    data_y1.columns = ['X'+ str(i) for i in range(1,data_period + 1)] + ['y']
    ## 储存
    data_y1.to_csv('../data/data_y1.csv',index=False)
    #############################################################
    
    
    data_y3 = pd.DataFrame(data['y3'])
    data_y3 = data_y3.diff(1)
    i = 1
    while i <= data_period:
        if i !=data_period:
            data_y3['X' + str(i)] = data_y3['y3'].shift(-i) 
        else:
            data_y3['X' + str(i)] = 0
            for j in range(0,dif_preiod):
                data_y3['X' + str(i)] = data_y3['X' + str(i)] + data_y3['y3'].shift(-i-j) 

        i = i + 1  

    data_y3 = data_y3.dropna()
    data_y3.columns = ['X'+ str(i) for i in range(1,data_period + 1)] + ['y']
    data_y3.to_csv('../data/data_y3.csv',index=False)
    return data_y1,data_y3


def train_test_split(data,name):
    data.loc[1:1851,:].to_csv('../data/data_' + str(name) +  '_train.csv',index=False)
    data.loc[1852:,:].to_csv('../data/data_' + str(name) +  '_test.csv',index=False)
























