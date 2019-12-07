# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:36:25 2019

@author: tangguoze
"""

import pandas as pd             #表格与数据处理
import numpy as np              #向量与矩阵运算
import matplotlib.pyplot as plt  #绘图
import seaborn as sns             #更多绘图功能
sns.set()

#from dateutil.relativedelta import relativedelta   # 日期数据处理

import statsmodels.formula.api as smf           #梳数理统计
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

##
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA,ARMA
#sm.tsa

import sklearn as sk

from random import randrange


#######################################################


##
data_YTM = pd.read_csv("..\\data\\financial data (.csv project).csv")
##

y3 = data_YTM['y3']

#y1.index = pd.Index(sm.tsa.datetools.dates_from_range())

y3.plot()  #原数据时间序列图

dy3 = y3.diff(1)    #将y1进行1阶差分

dy3 = dy3.dropna()  ##去空值

dy3.plot(figsize = (9,5))   #1阶差分后的时间序列图

plot_acf(dy3,lags = 20).show()  #1阶差分后的acf图

plot_pacf(dy3, lags = 20).show()  # 1阶差分后的pacf图

###########需要运算时间，可跳过改步骤，结果已经给出
#train_results = sm.tsa.arma_order_select_ic(dy3, ic=[ 'aic', 'bic'], trend= 'nc', max_ar= 6, max_ma= 5)   ## p、q的最优值
#print( 'AIC', train_results.aic_min_order)  #(5,4)
#print( 'BIC', train_results.bic_min_order)  #(1,0)
##############
model3 = ARMA(dy3, order = (2,4)) #设定模型阶数

result3 = model3.fit(disp = 0)  #拟合

pred3 = result3.predict()   #训练数据的估计

fore3 = result3.forecast(5)[0]   #后五期的预测


from sklearn.metrics import mean_squared_error

mse3 = mean_squared_error(dy3,pred3)  # 均方误差



######################################



#try 去掉最后5项后的均方误

y3_r5 = data_YTM['y3'].shift(-5)

dy3_r5 = y3_r5.diff(1)    #将y1进行1阶差分

dy3_r5 = dy3_r5.dropna()  ##抓取空值

###############同上
#train_results = sm.tsa.arma_order_select_ic(dy3_r5, ic=[ 'aic', 'bic'], trend= 'nc', max_ar= 5, max_ma= 5)   ## p、q的最优值
#print( 'AIC', train_results.aic_min_order) #(5,4)
#print( 'BIC', train_results.bic_min_order) #(1,0)
#################


model3_r5 = ARMA(dy3_r5, order = (1,1))

result3_r5 = model3_r5.fit()

pred3_r5 = result3_r5.predict()   #原始一阶差分模型估计值

mse3_r5 = mean_squared_error(dy3_r5,pred3_r5)   ##你和模型时的估计值

fore3_r5 = result3.forecast(5)[0]           ###后五期的预测值

#############
####try
list_y3_hat = []
m = y3[1951]
for i in range(5):
    m = m + fore3_r5[i]
    list_y3_hat.append(m)
    
y3_hat = pd.DataFrame(list_y3_hat)

mse3_nd = mean_squared_error(y3[1952:1957],y3_hat)     #后五期值的 均方误差
#####
####################








#####################################################

y1 = data_YTM['y1']

y1.plot()  #原数据时间序列图

dy1 = y1.diff(1)    #将y1进行1阶差分

dy1 = dy1.dropna()  ##去空值

dy1.plot(figsize = (9,5))   #1阶差分后的时间序列图

plot_acf(dy1,lags = 20).show()  #1阶差分后的acf图

plot_pacf(dy1, lags = 20).show()  # 1阶差分后的pacf图

###########需要运算时间，可跳过改步骤，结果已经给出
#train_results = sm.tsa.arma_order_select_ic(dy1, ic=[ 'aic', 'bic'], trend= 'nc', max_ar= 6, max_ma= 6)   ## p、q的最优值
#print( 'AIC', train_results.aic_min_order)  #(5,5)
#print( 'BIC', train_results.bic_min_order)  #(3,2)
##############
model1 = ARMA(dy1, order = (1,1)) #设定模型阶数

result1 = model1.fit(disp = 0)  #拟合

pred1 = result1.predict()   #训练数据的估计

fore1 = result1.forecast(5)[0]   #后五期的预测


from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(dy1,pred1)  # 均方误差






###################







y1_r5 = data_YTM['y1'].shift(-5)

dy1_r5 = y1_r5.diff(1)    #将y1进行1阶差分

dy1_r5 = dy1_r5.dropna()  ##抓取空值

###############同上
#train_results = sm.tsa.arma_order_select_ic(dy1_r5, ic=[ 'aic', 'bic'], trend= 'nc', max_ar= 7, max_ma= 7)   ## p、q的最优值
#print( 'AIC', train_results.aic_min_order) #(5,4)
#print( 'BIC', train_results.bic_min_order) #(3,2)
#################


model1_r5 = ARMA(dy1_r5, order = (1,1))

result1_r5 = model1_r5.fit()

pred1_r5 = result1_r5.predict()   #原始一阶差分模型估计值

mse1_r5 = mean_squared_error(dy1_r5,pred1_r5)

fore1_r5 = result1.forecast(5)[0]


######
###############
list_y1_hat = []
m = y1[1951]
for i in range(5):
    m = m + fore3_r5[i]
    list_y1_hat.append(m)
    
y1_hat = pd.DataFrame(list_y1_hat)

mse1_nd = mean_squared_error(y1[1952:1957],y1_hat)  ######

#####################
######

fore3_r5 = pd.DataFrame(fore3_r5)
df_new = pd.concat([pred3_r5,fore3_r5],ignore_index=True)

mse_ = mean_squared_error(dy3,df_new)
##









