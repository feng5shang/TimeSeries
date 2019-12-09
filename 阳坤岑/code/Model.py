# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:02:18 2019

@author: asus
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import func
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras import losses
from sklearn.svm import SVR
################################################模型#####################################
'''
模型输入第一个参数是训练集，第二个是测试集，第三个特征名字列表

返回值第一个为预测的Dataframe，第二个是MSE，第三个是model对象


'''


def Liner_Model(data,test,features):
    model = linear_model.LinearRegression()
    model.fit(data[features],data['y'])         
    t=model.predict(test[features])      
    print(mean_squared_error(t,test['y']),mean_squared_error([0]*len(test),test['y']))  
    return pd.DataFrame(t).rename(columns ={ 0: 'pred'}),mean_squared_error(t,test['y']),model
def SVR_Model(data,test,features):
    model = SVR('linear')
    model.fit(data[features],data['y'])         
    t=model.predict(test[features])      
    print(mean_squared_error(t,test['y']),mean_squared_error([0]*len(test),test['y']))  
    return pd.DataFrame(t).rename(columns ={ 0: 'pred'}),mean_squared_error(t,test['y']),model
    
def LSTM_Model(data,test,features):
    train_X,train_Y = data[features].loc[:,:].values,data['y'][:].values
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test[features].values.reshape((test.shape[0], 1, len(features)))
    test_Y = test['y']
    model = Sequential()
    model.add(LSTM(5,  input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adadelta')
    _ = model.fit(train_X, train_Y, epochs=45, batch_size=100, verbose=2,validation_data=(test_X, test_Y) , shuffle=False)
    t = model.predict(test[features].values.reshape((test.shape[0], 1, len(features))))
    print(mean_squared_error(t,test['y']),mean_squared_error([0]*len(test),test['y']))      
    return pd.DataFrame(t).rename(columns ={ 0: 'pred'}),mean_squared_error(t,test['y']),model

'''

已有支持的模型
Liner_Model——线性回归
SVR_Model ——支持向量机回归
LSTM_Model —— LSTM网络

Liner_Model(data,test,features)
SVR_Model(data,test,features)
LSTM_Model(data,test,features)
#  
'''

##MSE得分列表
socre_list = []

for i in range(1,6):#循环5次    每一次调预测i期的model参数使得mse最小，即best最小的参数
    #i = 1
    best = 100##设置初始best分数
<<<<<<< HEAD
    for j in range(1,5):
        #j = 1
=======
    for j in range(1,20):
>>>>>>> 6370cb9141e1ca2e5ce9e57f0597caf24a0cf3d2
        data_y1,data_y3 = func.pre_deal(j,i)##func.pre_deal函数 进行数据预处理 目的是将数据整理成差分数表 ， j表示数表中X的个数，i表示滞后几期
        func.train_test_split(data_y1,'y1')#func.train_test_split划分测试集和训练集
        func.train_test_split(data_y3,'y3')#func.train_test_split划分测试集和训练集
        
        
        data = pd.read_csv('../data/data_y3_train.csv')#读取训练数据
        test = pd.read_csv('../data/data_y3_test.csv')#读取测试数据
        temp_data = pd.read_csv('../data/data_y1_train.csv')#读取 ||辅助|| 训练数据
        temp_test = pd.read_csv('../data/data_y1_test.csv')#读取 ||辅助|| 测试数据
        ########################### 需要你构建特征进行测试
#        举个例子
        data['Z'+str(j)] = temp_data['X'+str(j)]

        test['Z'+str(j)] = temp_test['X'+str(j)]
#        上诉例子表示用三年期或者一年期的最后一期作为X辅助变量
        
        
        
        ##########################
        
        data = data.drop(index =[i for i in list(range(800,1150))],axis = 0).reset_index(drop=True)   #删除高波动项
        features = [feat for feat in data.columns.values if feat not in ['y']]   #生成features的list
        t,socre,_ =SVR_Model(data,test,features) #调用模型

        if socre < best:#如果的分低则记录参数
            best = socre
            best_j =j    
    print('i = ',i,'best score = ',best,'j = ',best_j)      
    socre_list.append(best)
print(np.mean(socre_list))#整体得分



########################逐项预测################################
#socre_list = []
#data_y1,data_y3 = func.pre_deal(5,1)
#func.train_test_split(data_y1,'y1')
#func.train_test_split(data_y3,'y3')
#data = pd.read_csv('../data/data_y1_train.csv')
#temp_test = pd.read_csv('../data/data_y1_test.csv')
#test =  pd.read_csv('../data/data_y1_test.csv')
#data = data.drop(index =[i for i in list(range(780,1160))],axis = 0).reset_index(drop=True)
#features = [feat for feat in data.columns.values if feat not in ['y']]   
#t,_,model = Liner_Model(data,temp_test,features)
#for i in range(2,6):
#    for j in range(1,5):
#        test['X' + str(j)] = test['X' + str(j+1)]
#    test['X5'] = t
#    Sum = test['X5']
#    t = model.predict(test[features])
#    Sum += t
#    data_y1,data_y3 = func.pre_deal(5,i)
#    func.train_test_split(data_y1,'y1')
#    func.train_test_split(data_y3,'y3')
#    temp_test = pd.read_csv('../data/data_y1_test.csv')
#    socre_list.append(mean_squared_error(Sum[0:len(temp_test)]  ,temp_test['y']))
#print(np.mean(socre_list))
#
#



