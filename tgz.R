#  install.packages("forecast")
#  library(forecast)
data_YTM<-read.table(file = "C:\\Users\\tangguoze\\Desktop\\3\\TS\\YTM.csv",sep = ",",header = T)
#导入数据
d_y1<-ndiffs(data_YTM$y1)
#判断需要几阶差分
d_y3<-ndiffs(data_YTM$y3)
#判断需要几阶差分
data_YTM_d1<-diff(data_YTM$y1,1) 
#对y1做一阶差分
data_YTM_d3<-diff(data_YTM$y3,1) 
#对y3做一阶差分
plot.ts(data_YTM_d1,ylab="y1") 
#画出y1的时间序列图
plot.ts(data_YTM_d3,ylab="y3") 
#画出y3的时间序列图
acf(data_YTM_d1)
#画出y1的acf图
acf(data_YTM_d3)   
#画出y3的acf图
pacf(data_YTM_d1)
#画出y1的pacf图
pacf(data_YTM_d3)
#画出y3的pacf图
model1_MLE<-arima(data_YTM_d1,order = c(1,0,4),method = "ML")  
print(model1_MLE)
#估计ARMA模型系数并估计
qqnorm(model1_MLE$residuals)
qqline(model1_MLE$residuals)
#残差正态性检验（QQ图）
print(Box.test(model1_MLE$residuals,type = "Ljung-Box")) 
#白噪声检验
model3_MLE<-arima(data_YTM_d3,order = c(1,0,4),method = "ML")  
print(model3_MLE)
qqnorm(model3_MLE$residuals)    
qqline(model3_MLE$residuals)    
print(Box.test(model3_MLE$residuals,type = "Ljung-Box"))
forecast_y1<-forecast(model1_MLE,h=5,level = c(99.5))
forecast_y3<-forecast(model3_MLE,h=5,level = c(99.5))
#预测后五期值
print(forecast_y1)
print(forecast_y3)
r1<-model1_MLE$residuals
r2<-model3_MLE$residuals
#模型内部残差
rs1<-t(r1)%*%r1/length(r1)
rs2<-t(r2)%*%r2/length(r2)
#模型均方误差


n<-length(data_YTM_d1)
#y1样本数
z_hat<-rep(0,n)
y_hat<-rep(0,n+5)
for (i in 1:4) {
  y_hat[i]<-data_YTM_d1[i]
}
z_hat[1]<-y_hat[1]
z_hat[2]<-y_hat[2]-0.41*y_hat[1]+0.0877*z_hat[1]
z_hat[3]<-y_hat[3]-0.41*y_hat[2]+0.0877*z_hat[2]-0.0153*z_hat[1]
z_hat[4]<-y_hat[4]-0.41*y_hat[3]+0.0877*z_hat[3]-0.0153*z_hat[2]-0.0346*z_hat[1]
#预设条件
for (i in 5:n) {
  z_hat[i]<-data_YTM_d1[i]-0.41*data_YTM_d1[i-1]+0.0877*z_hat[i-1]-0.0153*z_hat[i-2]-0.0346*z_hat[i-3]+0.0677*z_hat[i-4]
  #ARMA(1,0,4)模型，用真实值估计z_hat(i)
  y_hat[i]<-0.041*y_hat[i-1]+z_hat[i]-0.0877*z_hat[i-1]+0.0153*z_hat[i-2]+0.0364*z_hat[i-3]+0.0677*z_hat[i-4]
  #用z_hat估计y_hat(5~n)
}


prediction_<-rep(1,5)
prediction_[1]<-0.041*y_hat[n]+z_hat[n]-0.0877*z_hat[n-1]+0.0153*z_hat[n-2]+0.0364*z_hat[n-3]+0.0677*z_hat[n-4]
#预测y(n+1)
z_hat[n+1]<-prediction_[1]-0.41*y_hat[n]+0.0877*z_hat[n]-0.0153*z_hat[n-1]-0.0346*z_hat[n-2]+0.0677*z_hat[n-3]
prediction_[2]<-0.041*prediction_[1]+z_hat[n+1]-0.0877*z_hat[n]+0.0153*z_hat[n-1]+0.0364*z_hat[n-2]+0.0677*z_hat[n-3]
#预测z_hat(n+1)、y(n+2)
for (i in 3:5) {
  z_hat[n+i-1]<-prediction_[i-1]-0.41*prediction_[i-1]+0.0877*z_hat[n+i-2]-0.0153*z_hat[n+i-3]-0.0346*z_hat[n+i-4]+0.0677*z_hat[n+i-5]
  prediction_[i]<-0.041*prediction_[i-1]+z_hat[n+i-1]-0.0877*z_hat[n+i-2]+0.0153*z_hat[n+i-3]+0.0364*z_hat[n+i-4]+0.0677*z_hat[n+i-5]
}
#预测z_hat(n+2~n+4）、y(n+3~n+5)
for (i in 1:5) {
  y_hat[n+i]<-prediction_[i]
}
#将预测值加进y_hat


y_hat_notd<-rep(0,n+6)
y_hat_notd[1]=data_YTM$y1[1]
#没有差分的预设条件,notd意为没有进行差分
for (i in 2:(n+6)) {
  y_hat_notd[i]<-y_hat[i-1]+y_hat_notd[i-1]
}
#计算原样本的估计值


y_<-rep(1,5)
y_[1]<-prediction_[1]+y_hat[n]
#计算实际y(n+1)
for (i in 2:5) {
  y_[i]<-prediction_[i]+y_[i-1]
}
#计算实际y(n+2~n+5)


m1<-c(data_YTM$y1)
#将y1矩阵化
m2<-c(y_hat_notd[1:length(m1)])
#将y1的估计值矩阵化
mse1<-t(m1-m2)%*%(m1-m2)/length(m1)
#计算均方误差


