#устанавливаем библиотеки
install.packages('devtools')
library('devtools')
devtools::install_github("rstudio/tensorflow")
1
devtools::install_github("rstudio/keras")

#обращаемся к библиотекам
library('keras')
library('tensorflow')

#вытаскиваем базу данных, присваиваем имя и разбиваем на четыре части
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#создаем скелет нейронной сети
network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = 'softmax')

#функция потерь и точности
network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))

#приводим к нужной размерности
train_images <- array_reshape(train_images, c(60000, 28*28))
train_images <- train_images/255
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

#создаем категории для ярлыков
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

#тренируем нейронную сеть
network %>% fit(train_images, train_labels, epochs = 25, batch_size = 128)

#точность модели составила 99,99% (acc: 0.9999)

metric <- network %>% evaluate(test_images, test_labels)
metric

#предскажем значения для первых 100 элементов тестовой матрицы и последних 100
pred_one <- network %>% predict_classes(test_images[1:100,])
pred_two <- network %>% predict_classes(test_images[9900:10000,])

#сравниваем предсказанные значения с реальными 
test_labels1 <- mnist$test$y
test_labels1[9900:10000]

one <- ifelse(pred_one == test_labels1[1:100], 1, 0) 
sum(one)/length(one)

two <- ifelse(pred_two == test_labels1[9900:10000], 1, 0) 
sum(two)/length(two)

#ошибок не выявлено

#вывод одной из единиц массива
img <- mnist$test$x[9999, 1:28, 1:28]
image(as.matrix(img))
