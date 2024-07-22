# Health Indicators and Diabetes Prediction Project
Introduction
This project aims to predict diabetes using various health indicators. The dataset used includes multiple health-related attributes collected from individuals. The goal is to build a predictive model that can accurately classify individuals as diabetic or non-diabetic based on these health indicators.

# Project Overview and Goals
Project Overview
The project involves the use of a dataset containing health indicators to predict diabetes status. The dataset includes various attributes such as blood pressure, cholesterol levels, body mass index (BMI), smoking status, physical activity, and other health-related metrics. By analyzing this dataset, we aim to understand the relationships between these health indicators and the presence of diabetes.

# Goals
Data Exploration and Cleaning: Perform exploratory data analysis (EDA) to understand the dataset and clean the data by handling missing values and removing duplicates.
# Feature Analysis:
Analyze the distribution and significance of each feature in relation to the target variable (diabetes status).

# Model Training:
Train multiple machine learning models to predict diabetes status, including, KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, and SVC.
Based on the provided metrics, the Gradient Boosting Classifier appears to be the best-performing model overall. Here are the key points that support this conclusion:

## High Accuracy:

The Gradient Boosting Classifier has an overall accuracy of 0.85, which is on par with the best performing models.
Balanced Performance:

The precision, recall, and F1-score for the majority class (class 0) are high (0.86, 0.98, 0.92).
Although the minority class (class 1) still has lower metrics (0.57, 0.16, 0.24), the performance is better than the other models in terms of precision.
Macro and Weighted Averages:

The macro average precision, recall, and F1-score are 0.72, 0.57, and 0.58 respectively, indicating a better balance between the classes compared to the other models.
The weighted average precision, recall, and F1-score are 0.82, 0.85, and 0.81, which are among the highest.
Gradient Boosting Classifier
Training Set Score: Estimated to be close to 0.85 based on the accuracy.
Test Set Score: 0.85
Classification Report:
Precision, Recall, F1-Score for class 0: 0.86, 0.98, 0.92
Precision, Recall, F1-Score for class 1: 0.57, 0.16, 0.24
Accuracy: 0.85
Macro Avg Precision, Recall, F1-Score: 0.72, 0.57, 0.58
Weighted Avg Precision, Recall, F1-Score: 0.82, 0.85, 0.81

# Model Evaluation:
Evaluate the performance of each model using metrics such as accuracy, mean squared error, classification report, and confusion matrix.
## Decision Tree Classifier
Training Set Score: 0.9924
Test Set Score: 0.7769
Classification Report:
Precision, Recall, F1-Score for class 0: 0.87, 0.86, 0.87
Precision, Recall, F1-Score for class 1: 0.29, 0.32, 0.31
Accuracy: 0.78
Macro Avg Precision, Recall, F1-Score: 0.58, 0.59, 0.59
Weighted Avg Precision, Recall, F1-Score: 0.78, 0.78, 0.78

## Logistic Regression
Training Set Score: 0.8507
Test Set Score: 0.8487
Classification Report:
Precision, Recall, F1-Score for class 0: 0.86, 0.98, 0.92
Precision, Recall, F1-Score for class 1: 0.53, 0.14, 0.22
Accuracy: 0.85
Macro Avg Precision, Recall, F1-Score: 0.70, 0.56, 0.57
Weighted Avg Precision, Recall, F1-Score: 0.81, 0.85, 0.81

## K-Neighbors Classifier
Training Set Score: 0.8768
Test Set Score: 0.8308
Classification Report:
Precision, Recall, F1-Score for class 0: 0.87, 0.94, 0.90
Precision, Recall, F1-Score for class 1: 0.40, 0.21, 0.27
Accuracy: 0.83
Macro Avg Precision, Recall, F1-Score: 0.64, 0.58, 0.59
Weighted Avg Precision, Recall, F1-Score: 0.80, 0.83, 0.81

## Gradient Boosting Classifier
Training Set Score: Not provided, but assumed to be close to 0.85 based on accuracy.
Test Set Score: Not explicitly provided, but overall accuracy is 0.85.
Classification Report:
Precision, Recall, F1-Score for class 0: 0.86, 0.98, 0.92
Precision, Recall, F1-Score for class 1: 0.57, 0.16, 0.24
Accuracy: 0.85
Macro Avg Precision, Recall, F1-Score: 0.72, 0.57, 0.58
Weighted Avg Precision, Recall, F1-Score: 0.82, 0.85, 0.81
# Visualization: 
Create visualizations to illustrate the distribution of the target variable and the relationships between features and the target.
Optimization: Fine-tune the models to improve their performance and select the best model for predicting diabetes status



[the dataset can be found at the](https://www.archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The dataset contains the following columns:

Diabetes_binary: Indicates whether the individual has diabetes (0: No, 1: Yes)

HighBP: High blood pressure status.

HighChol: High cholesterol status.

CholCheck: Cholesterol check in the past five years.

BMI: Body Mass Index.

Smoker: Smoking status.

Stroke: History of stroke.

HeartDiseaseorAttack: History of heart disease or heart attack.

PhysActivity: Physical activity status.

Fruits: Fruit consumption frequency.

Veggies: Vegetable consumption frequency.

HvyAlcoholConsump: Heavy alcohol consumption status.

AnyHealthcare: Healthcare coverage status.

NoDocbcCost: Unable to see a doctor due to cost.

GenHlth: General health condition.

MentHlth: Number of days mental health was not good in the past 30 days.

PhysHlth: Number of days physical health was not good in the past 30 days.

DiffWalk: Difficulty walking or climbing stairs.

Sex: Gender of the individual.

Age: Age category.

Education: Education level.

Income: Income level.

For more details, visit the UCI Machine Learning Repository.
