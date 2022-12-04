install.packages('kohonen')
library('kohonen')

data('wines')
View(wines)

set.seed(1)

som.wines <- som(scale(wines), grid = somgrid(5, 5, 'hexagonal'))
View(som.wines)
som.wines

plot(som.wines, main = 'wine kohonen') #визуализация

plot(som.wines, type = 'changes', main = 'changes') #ошибка, y-величина ошибки, х-количество эпох, 
# используется для оценки количества операций, ошибка упала и не возрастает

150/177
177*0.7
train <- sample(nrow(wines), 150) #тренировочная выборка. обычно 80 на 20 или 70 на 30
train

x_train <- scale(wines[train, ]) #отшкалированные данные (150) - известные наблюдения 
x_test <- scale(wines[-train, ],
                center = attr(x_train, 'scaled:center'),
                scale = attr(x_train, 'scaled:center')) #шкалирование по тренировочной выборке (27) - неизвестные наблюдения

vintages

train_data <- list(measurements = x_train,
                   vintages = vintages[train]) #матрица (150)
test_data <- list(measurements = x_test,
                  vintages = vintages[-train]) #тестовая матрица (27)

som_grid <- somgrid(5, 5, 'hexagonal')

som.wines <- supersom(train_data, grid = som_grid) #шаблон на основе train_data, обучение нейронной сети на базе som_grid
plot(som.wines) #сортировка по винограду

#оценка качества нейронной сети
som_predict <- predict(som.wines, newdata = test_data) #предсказание

table(vintages[-train], som_predict$predictions[['vintages']]) #количество правильных и неправильных аргументов

som_predict$predictions


som.wines[['distances']] #расстояние между наблюдением и нейроном

som.wines[['unit.classif']]
classif <- data.frame(vintages = vintages[train], class = som.wines[['unit.classif']])
classif

--------------------------
install.packages('MASS')
library('MASS')
data('bacteria')
bacteria

set.seed(1)


# преобразование вектора в числовой вид
a <- c(rep(1, times = 10), rep(0, times = 10))
a
b <- as.factor(a)

