# HW3 - AFT Models for Survival Analysis

## Overview
This project applies **Accelerated Failure Time (AFT) models** to predict customer **churn** using the **telco** dataset. The goal is to model the time to churn and understand which factors influence churn risk. Various AFT models were implemented and compared, including:
- **Weibull AFT**
- **Log-Logistic AFT**
- **Log-Normal AFT**
- **Generalized Gamma Regression**
- **Piecewise Exponential Regression**

## Features
- **Data Preprocessing**: Handling missing values, converting categorical columns to numerical codes, and one-hot encoding.
- **Modeling**: The project compares several AFT models to predict customer churn.
- **Model Comparison**: The models are evaluated using survival curves, **AIC**, and **Concordance Index**.
- **Customer Lifetime Value (CLV)**: CLV is calculated for each customer based on the survival functions predicted by the AFT models.

## How to Run
1. **Clone this repository**:
   ```bash
   git clone https://github.com/AnzhelaDavityan/Homeworks.git
   ```
2. **Navigate to the project directory**:
  ```bash
  cd Homeworks/HW3
  ```
3.1. **Create and activate a virtual environment On macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
3.2. **Create and activate a virtual environment On Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
4. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
5. **Run the Python script**:
  ```bash
  python AFT_model.py
  ```


