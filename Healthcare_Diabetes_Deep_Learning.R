install.packages(c("keras", "tidyverse", "DataExplorer", "SHAPforxgboost"))

library(keras)
library(tidyverse)
library(DataExplorer)
library(SHAPforxgboost)

dataset <- read.csv("C:/CarolineZiegler/Studium_DCU/8. Semester/Business Analytics Portfolio/Portfolio/03_Healthcare/Healthcare-Diabetes.csv")

# Display the first few rows of the dataset
head(dataset)

# Checking for missing values
plot_missing(dataset)

# No missing values

# Trying to understand the Feature Importance with Deep Learning

# Split the data into features and labels
set.seed(123) 
features <- dataset %>% select(-Id, -Outcome) %>% scale() # Scale features to have mean=0 and sd=1
labels <- dataset$Outcome

# Split into training and testing sets
index <- sample(1:nrow(features), round(0.8*nrow(features)))
x_train <- features[index, ]
y_train <- labels[index]
x_test <- features[-index, ]
y_test <- labels[-index]

# Define the model
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = 'relu', input_shape = c(ncol(x_train))) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)
