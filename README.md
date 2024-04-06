# **Diabetes Analysis Repository**

This repository includes comprehensive analysis files for diabetes research, utilizing both Python and R programming languages for data analysis and machine learning. Below are the details of the files contained within the repository:
Files and Descriptions
- Healthcare-Diabetes.csv: The primary dataset used for the analysis, containing various health indicators that are potential predictors for diabetes.
- Diabetes - Feature Selection and Importance with Python.ipynb: A Python Jupyter notebook detailing the feature selection process and the importance of various features in predicting diabetes outcomes.
- Diabetes - Feature Selection and Importance with R.R: An R script for conducting feature selection and evaluating the importance of different variables in the context of diabetes.
- Healthcare_Diabetes_Deep_Learning.R: An R script dedicated to building and evaluating deep learning models for diabetes prediction.
- Healthcare_Diabetes_Statistical_Analysis.R: This R script focuses on statistical analysis of the data to uncover significant trends and relationships.

## **Project Description**
### **Objective**

The project aims to leverage statistical analysis and machine learning to understand the factors that contribute to diabetes outcomes, with a focus on predictive modeling to identify high-risk individuals based on their health indicators.

### **Data Analysis**
- Data Cleaning: Addressed missing and implausible values by imputing medians for critical parameters such as Glucose and BloodPressure.
- Exploratory Data Analysis (EDA): Utilized Python (matplotlib, seaborn) for initial data exploration and R for advanced visualizations to understand the distribution and relationship of variables.
- Feature Engineering: Identified and selected significant features influencing diabetes outcomes for modeling.
- Predictive Modeling: Applied a neural network using TensorFlow in Python to predict diabetes outcomes. Also explored permutation feature importance to understand feature relevance.
- Statistical Analysis in R: Conducted detailed statistical analysis to explore correlations and performed visualizations to support findings.

### **Insights and Outputs**

- Visualizations illustrating the distribution of various health indicators and their relationships.
- Predictive models indicating the likelihood of diabetes based on health indicators.
- Identification of key predictors of diabetes through feature importance analysis.
- Comprehensive correlation analysis using advanced visual techniques in R.

## **Installation**

To set up the project environment:

- Clone this repository to your local machine.
- Ensure you have Python and R installed on your system.
- Install necessary Python libraries:

    pip install pandas numpy matplotlib seaborn tensorflow scikit-learn

Install necessary R packages:

    install.packages(c("tidyverse", "skimr", "ggplot2", "reshape2"))

## **Usage**

- Data Analysis in Python: Open the Jupyter notebook to view the Python-based data analysis process.
- Data Analysis in R: Run the R script to execute the data visualization and statistical analysis.
