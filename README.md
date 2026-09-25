# Health Insurance Cost Predictor

A machine learning application that predicts estimated health insurance charges based on demographic and health-related features.

## Project Overview

This project uses regression-based machine learning models to estimate medical insurance costs using factors such as age, BMI, smoking status, number of children, sex, and geographic region.

The project covers the complete data science workflow, including data exploration, preprocessing, feature engineering, model training, evaluation, and deployment through Streamlit.

## Dataset

The project uses the Medical Cost Personal Dataset available on Kaggle.

Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance

The dataset contains the following features:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The target variable is:

- `charges` - medical insurance charges

## Exploratory Data Analysis

The exploratory analysis includes:

- Dataset structure and statistical analysis
- Missing-value analysis
- Duplicate detection
- Feature distributions
- Correlation analysis
- Insurance charges by smoking status
- Age versus insurance charges
- BMI versus insurance charges
- Regional analysis
- Analysis of categorical variables

## Machine Learning

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The model with the highest R² score was selected for the deployed prediction application.

## Streamlit Application

The Streamlit application allows users to enter customer information and receive an estimated insurance charge.

Input features include:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The trained machine learning model processes these inputs and generates the predicted insurance cost.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Google Colab

## Project Structure

```text
Health-Insurance-Cost-Predictor/
│
├── app.py
├── insurance_model.pkl
├── requirements.txt
├── Medical_Cost_Personalization_Insurance_Cost_....ipynb
└── README.md
