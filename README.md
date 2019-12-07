#### 首先将一阶差分样本划分为训练集、验证集和测试集

##### 	划分规则：最后100位样本作为测试，其余用于交叉验证，进而测试基模型的可靠性。

> 划分数据集： 
>
> ​						交叉验证集 data_y3_train
>
> ​						测试集 data_y3_test

​	测试结果：Lgbm 测试集表现MSE 0.01004

​					   LinearRegression 测试集表现MSE 0.009975

​					   加权融合 0.7 *  LinearRegression + 0.3 *  Lgbm  测试集表现MSE 0.009970

​					   整体LinearRegression 测试集表现MSE 0.009975

***

##### 除去高波动位置重新进行回测——去除780-1160

测试结果：LSTM 测试集表现MSE 0.009782

​					Lgbm 测试集表现MSE 0.009964

​					LinearRegression 测试集表现MSE 0.0099635

​					整体LinearRegression 测试集表现MSE 0.0099630

***

##### 直接去除高波动后合并两个data训练

测试结果：

​					LSTM 测试集表现MSE 0.009950

​					Lgbm 测试集表现MSE 0.0099152

​					LinearRegression 测试集表现MSE 0.010002

​					整体LinearRegression 测试集表现MSE 0.00998675		

***

##### 去除高波动后加入对应项最后一项

测试结果：

​				   LSTM 测试集表现MSE 0.0096768

​				   Lgbm 测试集表现MSE 0.00987376

​				   LinearRegression 测试集表现MSE 0.00985596

​				   整体LinearRegression 测试集表现MSE 0.00985540		

***

##### 去除高波动后加入四大统计量

测试结果：

​				   LSTM 测试集表现MSE 0.009739

​				   Lgbm 测试集表现MSE 0.010052

​				   LinearRegression 测试集表现MSE 0.0099509

​				   整体LinearRegression 测试集表现MSE 0.0099506	

***



​				   

