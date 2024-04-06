install.packages(c("tidyverse", "skimr", "ggplot2"))

library(tidyverse)
library(skimr)
library(ggplot2)

# Load the dataset
dataset <- read.csv("C:/CarolineZiegler/Studium_DCU/8. Semester/Business Analytics Portfolio/Portfolio/03_Healthcare/Healthcare-Diabetes.csv")

# Display the first few rows of the dataset to understand its structure
head(dataset)

# Getting a comprehensive summary of the dataset
skim(dataset)

# No missing values, really high std in Glucose, BloodPressure, SkinThickness and Insulin and age

# Check for outliers
boxplot(dataset$Glucose, main="Boxplot for Glucose Levels") 
boxplot(dataset$BloodPressure, main="Boxplot for BloodPressure")
# Outliers in BloodPressure identified
boxplot(dataset$SkinThickness, main="Boxplot for SkinThickness")
# Two outliers in the skinthickness feature
dataset <- dataset[dataset$SkinThickness <= 80, ]
boxplot(dataset$SkinThickness, main="Boxplot for SkinThickness")

# Visualize the distribution of each numeric feature
dataset %>% 
  select(-Id, -Outcome) %>% 
  gather(key = "features", value = "value") %>% 
  ggplot(aes(x = value)) + 
  geom_histogram(bins = 30, fill = "skyblue", color = "black") + 
  facet_wrap(~features, scales = "free") + 
  theme_minimal() + 
  labs(title = "Distribution of Numeric Features", x = NULL, y = "Frequency")

# Data is not normally distributed

# Examine how the different features relate to the diabetes outcome - glucose
dataset %>% 
  group_by(Outcome) %>% 
  summarise(Avg_Glucose = mean(Glucose, na.rm = TRUE)) %>% 
  ggplot(aes(x = as.factor(Outcome), y = Avg_Glucose, fill = as.factor(Outcome))) + 
  geom_col() + 
  labs(title = "Average Glucose Level by Diabetes Outcome", x = "Diabetes Outcome", y = "Average Glucose") + 
  scale_fill_manual(values = c("0" = "lightblue", "1" = "salmon"), labels = c("0" = "No Diabetes", "1" = "Diabetes")) + 
  theme_minimal()

# People with a higher glucose level on average are more likely to have diabetes
# Examine how the different features relate to the diabetes outcome - glucose
dataset %>% 
  group_by(Outcome) %>% 
  summarise(Avg_BloodPressure = mean(BloodPressure, na.rm = TRUE)) %>% 
  ggplot(aes(x = as.factor(Outcome), y = Avg_BloodPressure, fill = as.factor(Outcome))) + 
  geom_col() + 
  labs(title = "Average BloodPressure Level by Diabetes Outcome", x = "Diabetes Outcome", y = "Average BloodPressure") + 
  scale_fill_manual(values = c("0" = "lightblue", "1" = "salmon"), labels = c("0" = "No Diabetes", "1" = "Diabetes")) + 
  theme_minimal()

# Blood pressure has not a significant difference on diabetes as glucose level; however, people with a higher blood pressure on average are more likely to have diabetes

# Understanding the correlations better with a correlation matrix

# Install reshape2 if you haven't already
install.packages("reshape2")

# Load the reshape2 package
library(reshape2)

# Compute the correlation matrix
cor_matrix <- dataset %>% 
  select(-Id, -Outcome) %>% 
  cor()

# Visualize the correlation matrix
ggplot(data = melt(cor_matrix), aes(Var1, Var2, fill = value)) + 
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) + 
  theme_minimal() + 
  labs(title = "Correlation Matrix of Features", x = NULL, y = NULL) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# - Glucose is the most strongly correlated with the Outcome variable, indicating it is a key predictor of diabetes.
# - BMI shows significant positive correlations with SkinThickness and Insulin levels, hinting at a potential relationship with insulin resistance in the context of higher BMI.
# - Age has a moderate positive correlation with Pregnancies, which is an expected finding as the possibility of having more pregnancies increases with age.
# - Blood Pressure has moderate correlations with other features but not as strongly associated with the Outcome variable, suggesting it might be a less critical predictor in the context of diabetes.
# - The heatmap shows varying degrees of correlations among features, but not all are strongly associated with the Outcome. This can guide feature selection by indicating which variables might be the most informative for predicting diabetes.

