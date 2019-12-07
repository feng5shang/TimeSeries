#  install.packages("forecast")
#  library(forecast)
data_YTM<-read.table(file = "C:\\Users\\tangguoze\\Desktop\\tgz\\data\\financial data (.csv project).csv",sep = ",",header = T)
#导入数据

##以最后5项test

data_YTM_d1<-diff(data_YTM$y1[1:(length(data_YTM$y1)-5)],1) 
#对y1做一阶差分
data_YTM_d3<-diff(data_YTM$y3[1:(length(data_YTM$y3)-5)],1) 
#对y3做一阶差分

model1_MLE<-arima(data_YTM_d1,order = c(1,0,1),method = "ML")  
print(model1_MLE)

model3_MLE<-arima(data_YTM_d3,order = c(1,0,1),method = "ML")  
print(model3_MLE)

forecast_y1<-forecast(model1_MLE,h=5,level = c(99.5))
forecast_y3<-forecast(model3_MLE,h=5,level = c(99.5))
print(forecast_y1)
print(forecast_y3)

#预测
pred_1<-predict(model1_MLE,5)
pred_3<-predict(model3_MLE,5)
#

#
r1<-model1_MLE$residuals
r2<-model3_MLE$residuals
#模型内部残差
rs1<-t(r1)%*%r1/length(r1)
rs2<-t(r2)%*%r2/length(r2)
#模型均方误差


##############
data_YTM_d1_o<-diff(data_YTM$y1,1) 
#对y1做一阶差分
data_YTM_d3_o<-diff(data_YTM$y3,1) 
##############


#d_1<-data_YTM_d1_o[1952:1956]
#d_3<-data_YTM_d3_o[1952:1956]
#mse1<-t(d_1-forecast_y1$mean)%*%(d_1-forecast_y1$mean)/5
#mse3<-t(d_3-forecast_y3$mean)%*%(d_3-forecast_y3$mean)/5


#############
y3_hat = rep(1,5)
m <- data_YTM$y3[1952]
for (i in 1:5) {
  m<-m+pred_3$pred[i]
  y3_hat[i]<-m
  
}

mse3<-t(data_YTM$y3[1953:1957]-y3_hat)%*%(data_YTM$y3[1953:1957]-y3_hat)/5
###############



#############
y1_hat = rep(1,5)
m <- data_YTM$y1[1952]
for (i in 1:5) {
  m<-m+pred_1$pred[i]
  y1_hat[i]<-m
  
}

mse1<-t(data_YTM$y1[1953:1957]-y1_hat)%*%(data_YTM$y1[1953:1957]-y1_hat)/5
#############


###############not yeat
#ARCH效应检验
#Portmanteau检验
#for (i in 1:5) {
#  print(Box.test(model3_MLE$residuals^2,type = "Ljung-Box",lag = 6*i))
  
#}

#拟合garch(1,1)
#r.fit<-garch(model3_MLE$residuals,order=c(1,1))
#summary(r.fit)
###################



