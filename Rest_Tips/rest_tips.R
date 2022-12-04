
install.packages('nnet')
install.packages('NeuralNetTools')

library('nnet')
library('NeuralNetTools')

getwd()
list.files()

rest_tips <- read.csv("Rest_Tips.csv", sep = ';', header = T)

View(rest_tips)
attach(rest_tips)

model <- nnet(CustomerWillTip ~ Service + Ambience + Food,
              size = 5,
              rang = 0.1,
              decay = 0.01,
              maxit = 5000)
print(model)
plotnet(model)
garson(model)
olden(model)

model1 <- nnet(CustomerWillTip ~ Service + Ambience + Food,
              skip = 1,
              size = 0,
              rang = 0.1,
              decay = 0.01,
              maxit = 5000)
print(model1)
plotnet(model1)
garson(model1)
olden(model1)

model2 <- nnet(CustomerWillTip ~ Service + Ambience + Food,
               size = 1,
               rang = 0.1,
               decay = 0.01,
               maxit = 5000)
print(model2)
plotnet(model2)
garson(model2)
olden(model2)

model2$fitted.values #предсказанные значения
model2$residuals #ошибки

t1 <- data.frame(model2$fitted.values, 
                 rest_tips$CustomerWillTip, 
                 model2$residuals)
View(t1)
t1

t2 <- data.frame(model$fitted.values,
                 rest_tips$CustomerWillTip,
                 model$residuals)
View(t2)
t2

MSE <- sqrt(sum((model$residuals)^2/length(model$residuals))) #среднеквадратическая ошибка
MSE

MSE1 <- sqrt(sum((model2$residuals)^2/length(model2$residuals))) #среднеквадратическая ошибка
MSE1
