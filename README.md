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
![image](https://github.com/user-attachments/assets/7c3f4ca6-050a-4259-8609-e533798253ec)
observations:

Categories: There are two categories represented on the x-axis:
0.0: Likely indicating individuals who do not have diabetes.
1.0: Likely indicating individuals who have diabetes.

![image](https://github.com/user-attachments/assets/bca1d174-5417-4f2d-a3db-ea9ea1b3d8a7)
The chart consists of six bar plots showing the count of diabetes cases (Diabetes_binary) by various past medical conditions: HighBP, HighChol, CholCheck, Smoker, Stroke, and HeartDiseaseorAttack.

HighBP (High Blood Pressure):

Green bars (0.0) indicate individuals without diabetes, and orange bars (1.0) indicate those with diabetes.
Most individuals do not have high blood pressure or diabetes.
The black line shows the proportion of diabetics.
HighChol (High Cholesterol):

Similar to HighBP, with most individuals not having high cholesterol or diabetes.
CholCheck (Cholesterol Check):

Shows counts based on whether individuals had a cholesterol check, segmented by diabetes status.
Most who had a cholesterol check do not have diabetes.
Smoker:

Displays the distribution of smokers and non-smokers by diabetes status.
Non-smokers without diabetes have the highest count.
Stroke:

Shows counts of individuals with and without a stroke, segmented by diabetes status.
Most individuals without a stroke do not have diabetes.
HeartDiseaseorAttack:

Displays counts based on history of heart disease or attack, segmented by diabetes status.
Most individuals without heart disease or attack do not have diabetes.
In all plots, green bars (0.0) consistently show a larger group compared to orange bars (1.0), indicating more individuals without diabetes. The black lines highlight the proportion of diabetics in each category, providing insight into the relationship between these medical conditions and diabetes.

![image](https://github.com/user-attachments/assets/45a4f959-002c-4c5a-af6e-c91eed29d475)

This chart contains eight bar plots, each illustrating the count of diabetes cases (Diabetes_binary) by various demographic and lifestyle factors: Age, Sex, Education, Income, PhysActivity, Fruits, Veggies, and HvyAlcoholConsump.

# Age:

The distribution of diabetes cases by age, segmented into different age groups.
Green bars (0.0) indicate non-diabetics, and orange bars (1.0) indicate diabetics.
The black line shows the proportion of diabetics across age groups, with a noticeable increase in older age groups.
# Sex:
Shows the count of diabetes cases by sex.
More individuals without diabetes in both male and female groups.
# Education:
Displays the distribution of diabetes cases by education level.
The highest counts are among individuals with lower education levels.
# Income:
Shows the count of diabetes cases by income level.
Higher income levels generally have fewer diabetes cases.
# PhysActivity (Physical Activity):

Illustrates the count of diabetes cases based on physical activity.
Individuals with physical activity (1.0) have fewer diabetes cases.
# Fruits:

Shows the distribution of diabetes cases based on fruit consumption.
More individuals who consume fruits (1.0) do not have diabetes.
# Veggies:

Displays the count of diabetes cases based on vegetable consumption.
Higher vegetable consumption (1.0) is associated with fewer diabetes cases.
# HvyAlcoholConsump (Heavy Alcohol Consumption):

Shows the distribution of diabetes cases based on heavy alcohol consumption.
Higher counts of non-diabetics are seen among those with heavy alcohol consumption (1.0).
In all plots, green bars (0.0) consistently represent a larger group compared to orange bars (1.0), indicating more individuals without diabetes. The black lines indicate the proportion of diabetics within each category, providing insights into the relationship between these factors and diabetes prevalence.

![image](https://github.com/user-attachments/assets/1fff8f49-c210-4ca6-a25f-2e8a43a3a841)

The image is a correlation heatmap showing the relationships between various features in the dataset, such as Diabetes_binary, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, and Income.

Key Points:

Correlation Coefficients:

Range from -1 to 1, indicating the strength and direction of relationships.
Positive values indicate a direct relationship, negative values an inverse relationship.
Color Scale:

Dark red indicates high positive correlation, yellow indicates low or negative correlation.
Darker colors show stronger correlations.
Notable Correlations:

HighBP and HeartDiseaseorAttack: 0.63
BMI and HighBP: 0.28
Age and PhysActivity: -0.24
Education and Income: 0.42
Diabetes_binary:

Positive correlation with HighBP (0.25) and BMI (0.19).
Negative correlation with Veggies (-0.06) and Fruits (-0.025), suggesting higher consumption is linked to lower diabetes prevalence.

![image](https://github.com/user-attachments/assets/e3efa88b-3f9a-4892-826b-cde6a03c2397)
The box plot compares the ages of individuals with and without diabetes (Diabetes_binary).

No Diabetes (0.0):

Green box.
Median age: ~8.
IQR: 6 to 10.
Whiskers: 2 to 12.
With Diabetes (1.0):

Orange box.
Median age: ~10.
IQR: 8 to 12.
Whiskers: 4 to 13, with outliers below 4.
Overall, individuals with diabetes are generally older, indicated by the higher median and IQR.

![image](https://github.com/user-attachments/assets/b0ea4ce4-0f39-4377-afa7-bac0d0cc8984)

# Conclusion
Low Multicollinearity: The VIF values for all features are below 2, suggesting that there is low multicollinearity in this dataset. This is good for regression models, as it implies that the features are relatively independent of each other.
Feature Selection: Since multicollinearity is low, all features can be included in regression models without the risk of inflated standard errors or unstable coefficient estimates.


![image](https://github.com/user-attachments/assets/8b9fc8b7-c12c-4d62-a077-efa1fff1328f)

High Importance Features: Features like Diabetes_binary, PhysHlth, BMI, MentHlth, and Age have high scores, indicating strong predictive power.
Low Importance Features: Features like CholCheck, AnyHealthcare, and Sex have low scores and are thus dropped from the dataset.













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
