install.packages("forecast")
library(forecast)

#########arma###########
data_YTM<-read.table(file = "C:\\Users\\tangguoze\\Desktop\\tgz\\data\\financial data (.csv project).csv",sep = ",",header = T)
#导入数据

##以最后5项test(即先去掉后5项，以前面数据预测该5项,然后用真实值评估模型)

data_YTM_d1<-diff(data_YTM$y1[1:(length(data_YTM$y1)-5)],1) 
#对切片后的y1做一阶差分
data_YTM_d3<-diff(data_YTM$y3[1:(length(data_YTM$y3)-5)],1) 
#对切片后的y3做一阶差分

model1_MLE<-arima(data_YTM_d1,order = c(1,0,4),method = "ML")  #拟合y1模型, 参数估计方法为MLE 
print(model1_MLE)

model3_MLE<-arima(data_YTM_d3,order = c(1,0,5),method = "ML")  #拟合y3模型, 参数估计方法为MLE
#arma(1,4) 0.0534   #arma(1,5)0.0516
print(model3_MLE)

forecast_y1<-forecast(model1_MLE,h=5,level = c(99.5))  #预测y1去掉的后五期,经检验,forecast与predict结果相同
forecast_y3<-forecast(model3_MLE,h=5,level = c(99.5))  #预测
print(forecast_y1)    #forecast_y1$mean 是预测值
print(forecast_y3)    #forecast_y3$mean 是预测值

#预测
pred_1<-predict(model1_MLE,5)  #predict预测y1
pred_3<-predict(model3_MLE,5)  #predict预测y3
print(pred_1)     #pred_1$pred 为预测值
print(pred_3)     #pred_3$pred 为预测值
#

######ignore###########
#r1<-model1_MLE$residuals
#r2<-model3_MLE$residuals
#模型内部残差
#rs1<-t(r1)%*%r1/length(r1)
#rs2<-t(r2)%*%r2/length(r2)
#模型均方误差


##########ignore#########
#data_YTM_d1_o<-diff(data_YTM$y1,1) 
#对y1做一阶差分
#data_YTM_d3_o<-diff(data_YTM$y3,1) 
##############

#d_1<-data_YTM_d1_o[1952:1956]
#d_3<-data_YTM_d3_o[1952:1956]
#mse1<-t(d_1-forecast_y1$mean)%*%(d_1-forecast_y1$mean)/5
#mse3<-t(d_3-forecast_y3$mean)%*%(d_3-forecast_y3$mean)/5


#############计算y3mse##########
y3_hat = rep(1,5)
m <- data_YTM$y3[1952]
for (i in 1:5) {
  m<-m+pred_3$pred[i]
  y3_hat[i]<-m
  
}

mse3<-t(data_YTM$y3[1953:1957]-y3_hat)%*%(data_YTM$y3[1953:1957]-y3_hat)/5
###############

#############计算y1mse#################
y1_hat = rep(1,5)
m <- data_YTM$y1[1952]
for (i in 1:5) {
  m<-m+forecast_y1$mean[i]
  y1_hat[i]<-m
  
}

mse1<-t(data_YTM$y1[1953:1957]-y1_hat)%*%(data_YTM$y1[1953:1957]-y1_hat)/5
#############






###############单独使用garch模型？？############
install.packages("fGarch")
library(fGarch)

ARCH效应检验
Portmanteau检验
for (i in 1:5) {
  print(Box.test(model3_MLE$residuals^2,type = "Ljung-Box",lag = 6*i))
  
}
#检验结果都显著，即拒绝原假设：no ARCH effects

#拟合garch(1,1)
GARCH.model_1<-garchFit(~garch(1,1),data = data_YTM_d3,trace = FALSE)
summary(garch_)
GARCH.model_2 <- garchFit(~garch(2,1), data=data_YTM_d3,trace=FALSE) # GARCH(1,2)-N模型
GARCH.model_3 <- garchFit(~garch(1,1), data=data_YTM_d3,cond.dist='std', trace=FALSE)  #GARCH(1,1)-t模型
GARCH.model_4 <- garchFit(~garch(1,1), data=data_YTM_d3,cond.dist='sstd', trace=FALSE)  #GARCH(1,1)-st模型
#model_4 is best , mse_g4 = 0.0597
GARCH.model_5 <- garchFit(~garch(1,1), data=data_YTM_d3,cond.dist='ged', trace=FALSE)  #GARCH(1,1)-GED模型
GARCH.model_6 <- garchFit(~garch(1,1), data=data_YTM_d3,cond.dist='sged', trace=FALSE)  #GARCH(1,1)-SGED模型


pred_g3<-predict(GARCH.model_1, n.ahead = 5, trace =FALSE, mse = 'cond', plot=FALSE)
#每次计算mse更改 GARCH.model_i 参数并重新运行下列函数计算

#############单独的garch模型拟合得分###############
y3_hat_g = rep(1,5)
m <- data_YTM$y3[1952]
for (i in 1:5) {
  m<-m+pred_g3$meanForecast[i]
  y3_hat_g[i]<-m
  
}

mse3_try<-t(data_YTM$y3[1953:1957]-y3_hat_g)%*%(data_YTM$y3[1953:1957]-y3_hat_g)/5
#为方便查看，每次更新计算时可以修改变量名 mse3_try 后缀

###############




#ignore it#
###############ARCH,not yeat,the packages is not allowed###########
install.packages("FinTS")
library(FinTS)
#进行ARCH效应检验
for (i in 1:5) print(ArchTest(data_YTM_d3,lag=i))
#检验结果都显著，即拒绝原假设：no ARCH effects
data_YTM_d3.fit=garch(data_YTM_d3,order=c(0,1))
summary(x.fit)

################################





#拟合Arma+garch模型计算得分
#########arma+garch#############

garch_try1<-garchFit(~arma(2,4)+garch(1,1),data = data_YTM_d3,trace = FALSE)
#arma(1,4)0.0577  arma(2,4)0.0585
garch_try2 <- garchFit(~arma(2,4)+garch(2,1), data=data_YTM_d3,trace=FALSE) # GARCH(1,2)-N模型
#0.0577  arma(2,4)0.0597
garch_try3 <- garchFit(~arma(2,4)+garch(1,1), data=data_YTM_d3,cond.dist='std', trace=FALSE)  #GARCH(1,1)-t模型
#0.0555   arma(2,4)0.0597
garch_try4 <- garchFit(~arma(1,5)+garch(1,1), data=data_YTM_d3,cond.dist='sstd', trace=FALSE)  #GARCH(1,1)-st模型
#arma(1,4)+garch(1,1)0.0554   arma(2,4)+garch(1,1)0.0538  arma(2,4)+garch(2,1)0.0538 
garch_try5 <- garchFit(~arma(2,4)+garch(1,1), data=data_YTM_d3,cond.dist='ged', trace=FALSE)  #GARCH(1,1)-GED模型
#0.0564   arma(2,4)0.0544
garch_try6 <- garchFit(~arma(2,4)+garch(1,1), data=data_YTM_d3,cond.dist='sged', trace=FALSE)  #GARCH(1,1)-SGED模型
#0.0604   arma(2,4)0.0544

pred_g3<-predict(garch_try4, n.ahead = 5, trace =FALSE, mse = 'cond', plot=FALSE)
#每次计算mse更改 garch_try4 参数并重新运行下列函数计算

#############
y3_hat_g = rep(1,5)
m <- data_YTM$y3[1952]
for (i in 1:5) {
  m<-m+pred_g3$meanForecast[i]
  y3_hat_g[i]<-m
  
}

mse3_best<-t(data_YTM$y3[1953:1957]-y3_hat_g)%*%(data_YTM$y3[1953:1957]-y3_hat_g)/5
#为方便查看，每次更新计算时可以修改变量名 mse3_best 后缀

#####################

#结论：得分结果显示，所有模型中，dy1:Arma（1,4）mse最小，dy3:Arma(1,5) mse最小











