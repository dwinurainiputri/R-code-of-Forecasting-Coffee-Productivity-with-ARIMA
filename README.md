☕ Coffee Productivity Forecasting with ARIMA

This repository contains an R project that forecasts coffee productivity (tons/ha) in West Sumatra using time series analysis and ARIMA models.
The workflow includes data preparation, statistical tests, model fitting, diagnostics, and forecasting with evaluation metrics.

📌 Project Objectives
1. Analyze the coffee productivity trend from 2000–2023.
2. Test for homogeneity of variance (Levene’s Test).
3. Apply Box-Cox transformation for variance stabilization.
4. Check stationarity using the Augmented Dickey-Fuller (ADF) test.
5. Fit and compare ARIMA models: ARIMA(1,1,0); ARIMA(1,1,1); ARIMA(1,1,2)
6. Perform residual diagnostics (ACF, PACF, Ljung-Box).
7. Forecast coffee productivity for the next 4 years.
8. Evaluate accuracy using MAE, RMSE, and MAPE.

📂 Dataset
File: Data Productivity Coffee _ 2.xlsx
Sheet: Productivity
Variables:
Year (2000–2023)
Productivity (tons/ha)

🛠️ R Packages Required
library(car)
library(tidyverse)
library(forecast)
library(ggplot2)
library(tseries)
library(readxl)
library(MASS)
library(lmtest)
library(lawstat) # for Levene's Test
library(fpp2)    # for forecasting functions

🔎 Methodology
1. Data Preparation
Train data: 2000–2019
Test data: 2020–2023
Converted into time series (ts).

2. Tests Applied
Levene’s Test → checks variance homogeneity.
Box-Cox Transformation → stabilizes variance.
ADF Test → checks stationarity before and after differencing.

3. Model Building
Fitted ARIMA(1,1,0), ARIMA(1,1,1), ARIMA(1,1,2).
Compared using AIC, BIC and p-values of parameters.

4. Diagnostics
Residual plots (time series, ACF, PACF).
Ljung-Box Test for white noise.

5. Forecasting
Forecasted 4 years ahead (h=4).
Compared forecast values with actual test data (2020–2023).
Computed MAE, RMSE, MAPE.

📊 Results
The best model (based on AIC/BIC, significance, and diagnostics):
ARIMA(1,1,0)

🚀 How to Run
1. Clone this repository:
git clone https://github.com/yourusername/coffee-forecasting.git
cd coffee-forecasting
2. Open the R script in RStudio.
3. Adjust the dataset path inside read_excel() to your local file.
4. Run the script to reproduce results.

🏆 Author
Name: Dwi Nur’aini Putri
Project: Coffee Productivity Forecasting (Thesis Project ⚠️ Simulation only – data and results are not real, used for privacy reasons)
Methods: Time Series Analysis, ARIMA Modeling
