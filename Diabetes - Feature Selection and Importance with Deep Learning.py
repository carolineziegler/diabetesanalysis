#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


inpPath = "C:/CarolineZiegler/Studium_DCU/8. Semester/Business Analytics Portfolio/Portfolio/03_Healthcare/"
DiabetesDf = pd.read_csv(inpPath + "Healthcare-Diabetes.csv", delimiter =  ",", header = 0, index_col = 0)
DiabetesDf


# In[3]:


# Get a summary of the dataframe
print(DiabetesDf.info())


# In[4]:


# Check for missing values
print(DiabetesDf.isnull().sum())


# In[5]:


# Statistical summary to identify potential outliers
print(round(DiabetesDf.describe()),2)


# In[6]:


# The Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns have minimum values of 0, which may not be plausible in a real-world scenario and could indicate missing or incorrect data


# In[7]:


# Handling zero values: replacing zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI by imputing the median of the respective columns
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# In[8]:


for column in columns_with_zeros:
    median_value = DiabetesDf[column].median()  # Compute median excluding zeros
    DiabetesDf[column] = DiabetesDf[column].mask(DiabetesDf[column] == 0, median_value)


# In[9]:


print(round(DiabetesDf.describe()),2)


# In[14]:


import seaborn as sns


# In[15]:


# Distribution of features
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 20))
fig.tight_layout(pad=5.0)

for i, ax in enumerate(axes.flatten()):
    sns.histplot(DiabetesDf[features[i]].dropna(), kde=True, ax=ax)
    ax.set_title(f'Distribution of {features[i]}')
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')


# In[17]:


# Analyizing relationships between the feature with a correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = DiabetesDf.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')

plt.show()


# In[10]:


# Since deep learning models perform better with normalized data, the data will be normalized to the numerical features


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


# Separating features and target feature
X = DiabetesDf.drop('Outcome', axis=1)
y = DiabetesDf['Outcome']


# In[13]:


# Applying standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[21]:


# Feedforward neural network will be used as it is suitable for binary classification tasks
# The model will have an input layer, a couple of hidden layers, and an output layer with a sigmoid activation function helping with the binary classification
import tensorflow as tf


# In[ ]:


# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary to see the structure
model.summary()


# In[ ]:


history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)


# In[ ]:


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy}")


# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


# Function to calculate permutation feature importance
def permutation_feature_importance(model, X_test, y_test):
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test).round())
    feature_importances = []
    for i in range(X_test.shape[1]):
        X_test_shuffled = X_test.copy()
        X_test_shuffled[:, i] = shuffle(X_test_shuffled[:, i])
        shuffled_accuracy = accuracy_score(y_test, model.predict(X_test_shuffled).round())
        importance = baseline_accuracy - shuffled_accuracy
        feature_importances.append(importance)
    return feature_importances

# Calculate and print feature importances
feature_importances = permutation_feature_importance(model, X_test, y_test)
print(feature_importances)

