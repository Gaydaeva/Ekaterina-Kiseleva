install.packages('devtools')
library('devtools')
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")

install.packages('tensorflow')
install.packages('keras')

# запускаем библиотеки

library('keras')
library('tensorflow')

use_condaenv("r-tensorflow")
install_keras()
install_tensorflow(version = '1.12')

mnist <- dataset_mnist()

# для удобства разиваем их на 4 объекта
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# строим архитектуру нейронной сети

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = 'softmax')

# Добавляем функцию потерь и точность

network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))

# Приводим к необходимой размерности

train_images <- array_reshape(train_images, c(60000, 28*28)) 
train_images <- train_images/255 
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# создаем категории для ярлыков

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# поле подготовки данных тренируем нейронную сеть

network %>% fit(train_images, train_labels, epochs = 25, batch_size = 128)

# точность модели составила 99,9%
# при этом видим по динамике, что с 20 эпохи ошибка практически не меняется

metric <- network %>% evaluate(test_images, test_labels)
metric

# по тестовой выборке 98,12


# предскажем значения для первых 100 элементов тестовой матрицы и последних 100

pred_one <- network %>% predict_classes(test_images[1:100,])
pred_two <- network %>% predict_classes(test_images[9900:10000,])

# сравним предсказанные значения с реальными 
# и поделим на размер выборки, чтобы посчитать точность

test_labels1 <- mnist$test$y
test_labels1[9900:10000]

one <- ifelse(pred_one == test_labels1[1:100], 1, 0) 
sum(one)/length(one)

two <- ifelse(pred_two == test_labels1[9900:10000], 1, 0) 
sum(two)/length(two)

# точность - 1, ошибок в этих 201 наблюдении не выявлено

# вывод одной из единиц массива
img <- mnist$test$x[9999, 1:28, 1:28]
image(as.matrix(img))

