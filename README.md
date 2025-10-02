☕ Coffee Productivity Forecasting in West Sumatra

This repository contains an R project that forecasts coffee productivity (tons/ha) in West Sumatra using time series analysis and ARIMA models.
The project applies statistical tests, transformations, and forecasting techniques to evaluate and select the best ARIMA model for predicting future coffee yields.

📌 Objectives
Test for homoscedasticity (equal variance) using Levene’s test.
Apply Box-Cox transformation to stabilize variance.
Test for stationarity using the Augmented Dickey-Fuller (ADF) test.
Fit and compare different ARIMA models.
Perform residual diagnostics and model evaluation.
Forecast coffee productivity for the next 4 years and validate with test data.

📂 Dataset
The dataset is stored in an Excel file:
Data Productivity Coffee _ 2.xlsx
Sheet: Productivity
Variables:

Year (2000–2023)

Productivity (tons/ha)

🛠️ Packages Required
# Core packages
library(tidyverse)
library(forecast)
library(ggplot2)
library(tseries)
library(readxl)
library(MASS)
library(lmtest)
library(car)

# Additional packages
library(lawstat) # for Levene’s test
library(fpp2)    # for forecasting

🔎 Methodology
1. Data Preparation
Import dataset from Excel.
Split into training data (2000–2019) and testing data (2020–2023).
Convert training data into time series (ts).

2. Statistical Tests
Levene’s Test → checks homogeneity of variance.
Box-Cox Transformation → stabilizes variance.
ADF Test → checks stationarity before and after differencing.

3. Model Building
Fit ARIMA(1, 3, 0), ARIMA(1, 3, 1), ARIMA(1, 3, 2), and ARIMA(1, 3, 3).
Compare models using AIC and BIC.
Check parameter significance (p-values).

4. Diagnostics
Residual plots (time series, ACF, PACF).
Ljung-Box Test for white noise.

6. Forecasting & Evaluation
Forecast 5 years ahead (h=5).
Compare forecasts with test data.

Evaluation metrics:
MAPE (Mean Absolute Percentage Error)

📊 Results
Best model :
ARIMA(1,3,0)
