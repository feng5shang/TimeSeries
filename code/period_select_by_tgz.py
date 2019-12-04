# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:48:42 2019

@author: asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('..//data//financial data (.csv project).csv')['y3']

data = pd.DataFrame(data)

data_c = data.copy()

list1 = []

for j in range(5,21):
    data = data_c.copy()
    i = 1
    while i <=j:
        data['X' + str(i)] = data['y3'].shift(-i) 
        i = i+1

    data.columns = ['X'+ str(i) for i in range(1,j+1)] + ['y']

    data = data.dropna().reset_index(drop=True)

    features = ['X'+ str(i) for i in range(1,j+1)]

    train = data[features].loc[0:1600,:]
    train_y = data['y'].loc[0:1600]
    test = data[features].loc[1600:len(data),:]
    test_y =  data['y'].loc[1600:len(data)]

    from lightgbm import LGBMRegressor
    clf = LGBMRegressor(n_jobs=-1)
    clf.fit(train,train_y)


    from sklearn.metrics import mean_squared_error


    pred = clf.predict(test)


    mean_squared_error(pred,test_y)
    
    list1.append(mean_squared_error(pred,test_y))

print(list1)

result = pd.DataFrame(list1 , columns = ['result'])

result.index = [np.arange(start = 5 , stop = 21 , step = 1)]
plt.plot(list(result.index),result)

result.plot()























