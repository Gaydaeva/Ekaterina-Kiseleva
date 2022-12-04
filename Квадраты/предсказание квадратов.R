install.packages('neuralnet')
library('neuralnet')

getwd()

#предсказание квадратов чисел

a <- 1:10
sq <- data.frame(a, a^2)
sq

names(sq) <- c('input', 'output')
sq

#нейронная сеть

attach(sq)
set.seed(1) #точка отсчета для генератора случайных чисел

w <- rnorm(16, mean = 0, sd = 0.5) #генератор случайных чисел, первичные веса
w

m <-  neuralnet(output ~ input,
                data = sq,
                hidden = 5,
                startweights = w,
                threshold = 0.05,
                learningrate = 0.01)
m
plot(m)

result <- as.data.frame(m$net.result)
result

t3 <- data.frame(result,
                 output = sq$output,
                 residual = sq$output - result)
names(t3) <- c('res', 'output', 'residual')
t3

detach(sq)
test <- data.frame(input = c(11,12))

predict(m, test) #предсказываем 11 и 12 значения

#многослойная сеточка
mm <-  neuralnet(output ~ input,
                data = sq,
                hidden = c(3, 5, 5),
                threshold = 0.05,
                learningrate = 0.01)
mm
plot(mm)
