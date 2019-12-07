# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""




import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn import linear_model

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
def pre_deal(dif_preiod=1):
    ###预处理函数： 将数据分为y1和y3数据然后进行存储整理成数表形式并返回
    
    ##读文件
    data = pd.read_csv('../data/financial data (.csv project).csv')

    ################################处理y1########################
    data_y1 = pd.DataFrame(data['y1'])
    data_y1 = data_y1.diff(1)

    ##构建数表
    i = 1
    while i <= 5:
        if i != 5:
            data_y1['X' + str(i)] = data_y1['y1'].shift(-i) 
        else:
            data_y1['X' + str(i)] = 0
            for j in range(0,dif_preiod):
                data_y1['X' + str(i)] = data_y1['X' + str(i)] + data_y1['y1'].shift(-i-j) 
        i = i + 1
    ##删除nan
    data_y1 = data_y1.dropna()
    ##更改column name
    data_y1.columns = ['X'+ str(i) for i in range(1,6)] + ['y']
    ## 储存
    data_y1.to_csv('../data/data_y1.csv',index=False)
    #############################################################
    
    
    data_y3 = pd.DataFrame(data['y3'])
    data_y3 = data_y3.diff(1)
    i = 1
    while i <= 5:
        if i !=5:
            data_y3['X' + str(i)] = data_y3['y3'].shift(-i) 
        else:
            data_y3['X' + str(i)] = 0
            for j in range(0,dif_preiod):
                data_y3['X' + str(i)] = data_y3['X' + str(i)] + data_y3['y3'].shift(-i-j) 

        i = i + 1  

    data_y3 = data_y3.dropna()
    data_y3.columns = ['X'+ str(i) for i in range(1,6)] + ['y']
    data_y3.to_csv('../data/data_y3.csv',index=False)
    return data_y1,data_y3


def hat_Count(data,features,name):
    ####hat估计函数 ，传入data 和 data中的features列表，和最后返回hat的名字
#    print('开始十折交叉验证')
    ##########################划分为10折###############################
    N = 10
    data_list = list(data.index)
    EVERY = int (len(data) / 10 )
    train_list = []
    test_list = []
    for i in range(N):
        if i == N-1:
            test_list.append(data.loc[i * EVERY : len(data)])
            train_list.append(data.drop(index =[i for i in list(range(i * EVERY, len(data))) if i in data_list],axis = 0))
        else:
            test_list.append(data.loc[i * EVERY : (i + 1 ) * EVERY - 1 ])
            train_list.append(data.drop(index =[i for i in list(range(i * EVERY, (i + 1 )*EVERY-1)) if i in data_list],axis = 0))
    ###################################################################
    
    ########################开始训练并生成hat的过程####################
    ##MSE得分list
    score = []
    ##hat的数据存在newdata中
    new_data = pd.DataFrame()
    ########################开始训练################################
    for i in range(len(train_list)):
        train_x,train_y = train_list[i][features], train_list[i]['y']
        test_x,test_y = test_list[i][features], test_list[i]['y']
        clf = linear_model.LinearRegression()
        clf.fit(train_x,train_y)
        pred = clf.predict(test_x)
        new_data = pd.concat([new_data,pd.DataFrame(pred)],axis = 0)
        print('第'+str(i)+"折 训练集MSE ：",mean_squared_error(clf.predict(train_x),train_y))
        print('第'+str(i)+"折 测试集MSE ：",mean_squared_error(pred,test_y))
        print('...............................')
        score.append(mean_squared_error(pred,test_y))
#    print('总体MSE均值',np.mean(score))
    ################################################################
    
    ##返回 newdata 和 score的均值
    return new_data.reset_index(drop=True).rename(columns={0:name}),np.mean(score)
    



##生成data_y1和data_y3
data_y1,data_y3 = pre_deal(5)

data_y1.loc[1:1851,:].to_csv('../data/data_y1_train.csv',index=False)
data_y1.loc[1852:,:].to_csv('../data/data_y1_test.csv',index=False)
data_y3.loc[1:1851,:].to_csv('../data/data_y3_train.csv',index=False)
data_y3.loc[1852:,:].to_csv('../data/data_y3_test.csv',index=False)




data_y1['std'] = data_y1[['X'+ str(i) for i in range(1,6)]].std(axis = 1)
data_y3['std'] = data_y3[['X'+ str(i) for i in range(1,6)]].std(axis = 1)
##copy一下方便待会反复迭代后更新
data_y1_ = data_y1.copy()
data_y3_ = data_y3.copy()

##原始features
features = ['X'+ str(i) for i in range(1,6)] 

###没有加入对方信息事的mse ，并称之为第0次迭代
data_y3_hat,y3_hat_score = hat_Count(data_y3,features,'y3_hat')
data_y1_hat,y1_hat_score = hat_Count(data_y1,features,'y1_hat')
print('第'+str(0)+"轮 y3_hat_score ：",y3_hat_score)
print('第'+str(0)+"轮 y1_hat_score ：",y1_hat_score)

import matplotlib.pyplot as plt


###反复3次
for i in range(1,4):
    data_y3_ = pd.concat([data_y3.reset_index(drop=True),data_y3_hat],axis = 1)
    data_y3_hat,y3_hat_score = hat_Count(data_y3_,['X'+ str(i) for i in range(1,6)] + ['y3_hat',],'y3_hat')
    print('第'+str(i)+"轮 y3_hat_score ：",y3_hat_score)   
    data_y1_ = pd.concat([data_y1.reset_index(drop=True),data_y3_hat.reset_index(drop=True)],axis = 1) 
    data_y1_hat,y1_hat_score = hat_Count(data_y1_,['X'+ str(i) for i in range(1,6)] + ['y3_hat',],'y1_hat')
    print('第'+str(i)+"轮 y1_hat_score ：",y1_hat_score)   
 
 
#
#开始十折交叉验证
#第0折 测试集MSE ： 0.004221270195360738
#第1折 测试集MSE ： 0.024283705644665293
#第2折 测试集MSE ： 0.055137987860180836
#第3折 测试集MSE ： 0.06742731954016053
#第4折 测试集MSE ： 0.17328313981296323
#第5折 测试集MSE ： 0.25342556659896626
#第6折 测试集MSE ： 0.0784640253588281
#第7折 测试集MSE ： 0.03701403521527298
#第8折 测试集MSE ： 0.03485733461865544
#第9折 测试集MSE ： 0.02518270046622148
#总体MSE均值 0.0753297085311275
#开始十折交叉验证
#第0折 测试集MSE ： 0.005682389996762313
#第1折 测试集MSE ： 0.02511110863624498
#第2折 测试集MSE ： 0.06435857325601681
#第3折 测试集MSE ： 0.1154865753960206
#第4折 测试集MSE ： 0.2701436243615123
#第5折 测试集MSE ： 0.4237549630321654
#第6折 测试集MSE ： 0.07106983275623265
#第7折 测试集MSE ： 0.036825680048653095
#第8折 测试集MSE ： 0.0246446430457489
#第9折 测试集MSE ： 0.01642805157366707
#总体MSE均值 0.1053505442103024
#第0轮 y3_hat_score ： 0.0753297085311275
#第0轮 y1_hat_score ： 0.1053505442103024
#开始十折交叉验证
#第0折 测试集MSE ： 0.004554058956630411
#第1折 测试集MSE ： 0.024238960283579445
#第2折 测试集MSE ： 0.05496421798523273
#第3折 测试集MSE ： 0.06740218605049565
#第4折 测试集MSE ： 0.1738498240770142
#第5折 测试集MSE ： 0.25464046864142614
#第6折 测试集MSE ： 0.07836242913105068
#第7折 测试集MSE ： 0.03687799838550455
#第8折 测试集MSE ： 0.03477591382426944
#第9折 测试集MSE ： 0.025260141704468532
#总体MSE均值 0.07549261990396718
#第1轮 y3_hat_score ： 0.07549261990396718
#开始十折交叉验证
#第0折 测试集MSE ： 0.005465691191151922
#第1折 测试集MSE ： 0.025097025759570037
#第2折 测试集MSE ： 0.06443009353176306
#第3折 测试集MSE ： 0.11541257182197029
#第4折 测试集MSE ： 0.27000429585331986
#第5折 测试集MSE ： 0.42376365963778
#第6折 测试集MSE ： 0.07281466638859967
#第7折 测试集MSE ： 0.037027939974415364
#第8折 测试集MSE ： 0.02501530252192102
#第9折 测试集MSE ： 0.016323588141238167
#总体MSE均值 0.10553548348217294
#第1轮 y1_hat_score ： 0.10553548348217294
#开始十折交叉验证
#第0折 测试集MSE ： 0.004546953018685545
#第1折 测试集MSE ： 0.024239475756649417
#第2折 测试集MSE ： 0.05497037349204776
#第3折 测试集MSE ： 0.06740096351096155
#第4折 测试集MSE ： 0.1738470576686277
#第5折 测试集MSE ： 0.2546804759912104
#第6折 测试集MSE ： 0.07837091209404916
#第7折 测试集MSE ： 0.036882077141473055
#第8折 测试集MSE ： 0.034776926070132365
#第9折 测试集MSE ： 0.02525768268795548
#总体MSE均值 0.07549728974317924
#第2轮 y3_hat_score ： 0.07549728974317924
#开始十折交叉验证
#第0折 测试集MSE ： 0.005465890000617806
#第1折 测试集MSE ： 0.025097024943052285
#第2折 测试集MSE ： 0.0644300901027923
#第3折 测试集MSE ： 0.11541259008241402
#第4折 测试集MSE ： 0.2700045302082583
#第5折 测试集MSE ： 0.42376374589925303
#第6折 测试集MSE ： 0.07281433031979022
#第7折 测试集MSE ： 0.03702788652105235
#第8折 测试集MSE ： 0.025015284593535313
#第9折 测试集MSE ： 0.016323643267750654
#总体MSE均值 0.10553550159385165
#第2轮 y1_hat_score ： 0.10553550159385165
#开始十折交叉验证
#第0折 测试集MSE ： 0.004546953820650488
#第1折 测试集MSE ： 0.024239475844552338
#第2折 测试集MSE ： 0.05497037378055226
#第3折 测试集MSE ： 0.06740096366771839
#第4折 测试集MSE ： 0.1738470591490224
#第5折 测试集MSE ： 0.25468047043510694
#第6折 测试集MSE ： 0.07837091355380912
#第7折 测试集MSE ： 0.03688207725705906
#第8折 测试集MSE ： 0.034776926131570775
#第9折 测试集MSE ： 0.025257682348989275
#总体MSE均值 0.07549728959890309
#第3轮 y3_hat_score ： 0.07549728959890309
#开始十折交叉验证
#第0折 测试集MSE ： 0.005465889985561005
#第1折 测试集MSE ： 0.025097024946828885
#第2折 测试集MSE ： 0.06443009010063788
#第3折 测试集MSE ： 0.11541259007827598
#第4折 测试集MSE ： 0.2700045301815382
#第5折 测试集MSE ： 0.42376374595006816
#第6折 测试集MSE ： 0.07281433033747528
#第7折 测试集MSE ： 0.03702788652334694
#第8折 测试集MSE ： 0.025015284625336073
#第9折 测试集MSE ： 0.016323643259001853
#总体MSE均值 0.10553550159880702
#第3轮 y1_hat_score ： 0.10553550159880702











