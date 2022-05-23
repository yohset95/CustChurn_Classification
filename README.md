# **Insurance Customer Churn Prediction**

* Written by Yohanes Setiawan
* This is my final project from IBM Machine Learning: Classification
* For further reading of the code, please open the `.ipynb` file

# **Business Understanding**

## Introduction
* Insurance companies around the world operate in a very competitive environment
* With various aspects of data collected from millions of customers, it is painstakingly hard to analyse and understand the reason for a customer’s decision to switch to a different insurance provider
* Knowing whether a customer is possibly going to switch beforehand gives Insurance companies an opportunity to come up with strategies to prevent it from actually happening

## Problem Statement
Customer churn is a serious problem in insurance companies

## Goal
To prevent customer churn in insurance companies

## Research Questions
* How to predict customer churn in an insurance company?
* What are influenced factors of customer churn in an insurance company?

## Objective Statement
* Creating a machine learning model (classification model) for customer churn prediction
* Analyzing influenced factors of customer churn in an insurance company

# **Analytical Approach**

* Descriptive analysis
* Graph analysis
* Table analysis
* Predictive analysis

# **Data Understanding**

## Dataset Description

* The dataset is taken from https://www.kaggle.com/datasets/mukulsingh/insurance-churn-prediction?select=Test.csv (Insurance Churn Prediction)
* Provided are **16 anonymized factors** (feature_0 to feature 15) that influence the churn of customers in the insurance industry

The unzipped folder will have the following files.

Train.csv – 33908 observations.

Test.csv – 11303 observations.

Target Variable: labels (0: Not Churn, 1: Churn)

## Dataset Information (Train.csv)
![](https://drive.google.com/uc?export=view&id=10hs5Tm6UBqW0vdKkXy0DPn2ffAndQF7r) </br>
* There is no missing value in the dataset
* All of columns are numerical values

## Checking Duplicated Data
![](https://drive.google.com/uc?export=view&id=1ORirO6pUOlMyPIMIf1vDbocRp2kohNTI) </br>
* There is no duplicated data in the dataset

# **Exploratory Data Analysis (EDA)**

## Descriptive Statistics
* All of columns have different range of values
* Therefore, scaling features is needed

## Univariate Distribution of Observations
![](https://drive.google.com/uc?export=view&id=1zZV3Wo67B3d5ObbVlvFZ2IPomXqFIjkF) </br>
* Skewed: `feature_0`,`feature_1`, `feature_3`,`feature_4`,`feature_5`,`feature_6`,`feature_10`,`feature_11`,`feature_13`

## Box Plot Analysis
![](https://drive.google.com/uc?export=view&id=1CRXa8A_RBJNpZQRibDrfygxGNCpVBXiT) </br>
* Highly influenced by outliers: `feature_0`,`feature_1`,`feature_3`,`feature_4`,`feature_5`,`feature_6`
* Lower influenced by outliers: `feature_10`,`feature_12`,`feature_15`

## Categorical Plot (Integers)
* I plot the int64 columns
![](https://drive.google.com/uc?export=view&id=1-hPSY-zpf9J_iCwEelfSkPzefixMQuW1) </br>
* I plot non "float" columns
* I have severe imbalanced class in `labels`

## Correlation
![](https://drive.google.com/uc?export=view&id=1xFeBondJ1n9K12mBVY38veENDlwCgAiL) </br>
* `feature_5` and `feature_6` are highly positive correlated (+0.55)
* `feature_5` and `feature_15` are highly negative correlated (-0.86)
* `feature_6` and `feature_15` are highly negative correlated (-0.59)
* Therefore, `feature_5` and `feature_6` should be removed to prevent multicollinearity

# **Data Preparation and Feature Engineering**

## Handling Outlier (Interquartile Range Analysis)
* Count of rows before removing outlier: 33908
* Count of rows after removing outlier: 21113

## Removing Multicollinear Features
* Removing `feature_5` and `feature_6` to prevent multicollinearity

## Train Test Split
* 70% training data, 30% testing data from cleaned “Train.csv”
* Using stratified train-test split for balancing proportion of classes in training and testing set
* Class from training data df_train Counter({0: 13942, 1: 837})
* Class from testing data df_test Counter({0: 5975, 1: 359})

## Feature Transformation
* Compared between Box Cox and Log Transformation
* I choose Log Transformation because the features look more like normal
distribution 

![](https://drive.google.com/uc?export=view&id=1ZaUuvZ6ET3hEKQ9AsLUPho_76p_IXWKo) </br>

## Feature Scaling
* Feature scaling is done by standardization

## Handling Imbalanced Class
* Severe imbalanced class

![](https://drive.google.com/uc?export=view&id=1BNniZ_huhJRTs1ZTAXZ0Qt-k-pFBLa6N) </br>
* Oversampling using SMOTE
* Combining Oversampling and Undersampling using SMOTE + TomekLinks

# **Modelling**

## Machine Learning Algorithms
* Logistic Regression
* Naive Bayes
* Decision Tree
* Random Forest
* Gradient Boosting Trees
* XGB

## Evaluation metrics
* Recall from class "Churn" or "1" (Main)
* Recall from class "Not Churn" or "0" (Main)
* PR AUC Score (Additional)
* ROC AUC Score (Additional)

## Research Scenario
![](https://drive.google.com/uc?export=view&id=1W_MeNrfjwV9OqBdmD3d4bE6EvZVTa1R0) </br>

# Model Evaluation

## Machine Learning Algorithms

### Logistic Regression
![](https://drive.google.com/uc?export=view&id=1mXMXpCWYa6XNq8KBQ1wYsnx-UgqgXBbE) </br>

### Naive Bayes
![](https://drive.google.com/uc?export=view&id=1cOKUHN30tXvINvtTPCgnEHubNULY4Ih_) </br>

### Decision Tree
![](https://drive.google.com/uc?export=view&id=1YhcwNia9SwUw6pmDFr7e5iGWQTTDgBba) </br>

### Random Forest
![](https://drive.google.com/uc?export=view&id=1VXzLqZ8XwYue_aGZ3a1OpR4dW1XgWfVP) </br>

### Gradient Boosting Trees
![](https://drive.google.com/uc?export=view&id=1XcgDzxfBNJrHQ1TTOhl8PqrWhStMeIBT) </br>

### XGB
![](https://drive.google.com/uc?export=view&id=1Hmk822gMVFuW3OkcbClyUPvTrijf6rwV) </br>

## Brief Explanation
* In imbalanced dataset without any resampling method, I focus on recall on both classes, such that I know how well the classifier can minimize the false negatives
* Although I have higher accuracy, the highest recall (lower false negatives) is achieved by negative class (Not Churn). While the positive one (Churn) get the lowest recall (close to zero) which means I still have many false negatives
* Logistic Regression achieved its best recall from both classes through SMOTE and SMOTE+TomekLinks. As a result, precision from positive class (Churn) was slowed down
* Naïve Bayes also achieved its great recall from SMOTE and SMOTE+TomekLinks even it is not as good as Logistic Regression
* Gradient Boosting Trees achieved its best recall from positive class (Churn) comparing with other tree-based classifier models (Decision Tree, Random Forest, and XGB) through resampling the training data with SMOTE+TomekLinks even though the recall from negative class is only in range of 0.5.
* SMOTE+TomekLinks did not give a significant impact to the model, especially in Logistic Regression
* PR AUC goes down when there is huge difference between precision and recall
* ROC AUC always performs well even though recall and precision from positive class (Churn) below 30%. Therefore, we can not rely on ROC AUC when we are dealing with imbalanced class

## Selected Model
* Finally, I have chosen the Logistic Regression with SMOTE as the selected model to develop Insurance Price Prediction
* Finding the best hyperparameter from the selected model (Hyperparameter Tuning)
* I used the optimal hyperparameter to the 5-Fold Validation

![](https://drive.google.com/uc?export=view&id=1TD0Ldi6TRV6vCfWSIAz-mgTnC3le9K2b) </br>

## Final Report of the Selected Model
![](https://drive.google.com/uc?export=view&id=1uk3i7WbCZmHK4C3OYyNRHpwTNdFmq2bf) </br>
![](https://drive.google.com/uc?export=view&id=1s37wsU4ZZbcjPrSsmRfQZI9RPMSAMswz) </br>
* Hyperparameter tuning optimizes the selected model
* In confusion matrix, we have seen that lower false negatives which consumes higher false positives
* In customer churn model, we can “omit” the false positives because it is more dangerous with unpredicted customer churn rather than not churned customers are classified as churned customers

# **Get Insights from the Selected Model**
![](https://drive.google.com/uc?export=view&id=1UTRAq6pYgqn3HfEFL49sji9KCqH4bQR-) </br>
* feature_3 has been the most important features in predicting insurance customer churn in this company
* feature_1 and feature_2 have similar importance 

# Data Validation Prediction (Test.csv)
* I still have “Test.csv” to be predicted its churned customers as evaluation
* After re-doing the feature engineering and data preparation steps (through parameter from previous training models), I evaluate the file with the selected model (Logistic Regression)
![](https://drive.google.com/uc?export=view&id=1Jwrbt7_oONF2Epyr6l8GzEhDZpy-jLYE) </br>

# **Summary of Key Findings and Insights**
* Identity of the features in this dataset is hidden such that it is quite difficult to analyze and get insights from exploratory data analysis
* This model has severe imbalanced dataset (5% positive class vs 95% positive class)
* Outliers exist almost in all features and get removed by IQR Method
* I have also removed two features which can cause multicollinearity because of its higher positive and negative correlation (> 0.5 for positive correlation and < -0.5 for negative correlation)
* I splitted the dataset into 70% training set and 39% testing set with stratified train-test-split
* I used Log Transformation to normalize skewed features
* Feature scaling is done using standardization
* I used six machine learning models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting Trees, and XGB comparing with three research scenarios: without balancing class, SMOTE, and SMOTE + Tomek Links
* The best model is achieved by Logistic Regression with SMOTE through highest recall from positive class (82%) and negative class (77%)
* I get insight from the selected model by checking the feature importances: I found that the most important feature is feature_3. Company should look forward to analyzing feature_3 as the important factor in order to predict churned customers from the insurance company
* Finally, with selected model, I have predicted 35% churned customers from the unseen data (Test.csv) such that company should pay attention to these customers and ensure them to not leave the insurance easily

# **Suggestions**
* Deploying the model to give predictive modelling to the company
* Another classification method can be used as comparison, such that K-Nearest Neighbors
* Adding feature explanation for better data understanding

Thank you for reading!