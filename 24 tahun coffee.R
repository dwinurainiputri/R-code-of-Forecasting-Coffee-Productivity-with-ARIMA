# Load the necessary packages
if (!require(forecast)) install.packages("forecast", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
if (!require(tseries)) install.packages("tseries", dependencies = TRUE)
if (!require(readxl)) install.packages("readxl", dependencies = TRUE)
if (!require(MASS)) install.packages("MASS", dependencies = TRUE)
if (!require(lmtest)) install.packages("lmtest", dependencies = TRUE)
if (!require(car)) install.packages("car", dependencies = TRUE)

library(car)
library(tidyverse)
library(forecast)
library(ggplot2)
library(tseries)
library(readxl)
library(MASS)
library(lmtest)

# Install and load the 'lawstat' package for Levene's test
if (!requireNamespace("lawstat", quietly = TRUE)) {
  install.packages("lawstat")
}
library(lawstat)

# Load the Excel data (adjust the file path and sheet name as needed)
Coffee_data<- read_excel("D:/Project/Skripsi R1/kopi/pasca semhas/Data Productivity Coffee _ 2.xlsx", 
                                        sheet = "Productivity")
# View the data
print(Coffee_data)

# Split the data
train_data <- Coffee_data[1:20, ]  # Training data
test_data <- Coffee_data[21:24, ]  # Test data

# Convert training data to time series
coffee_ts <- ts(train_data, start = 2000, frequency = 1)

# Plot the time series data
plot(coffee_ts, main = "Coffee Productivity in West Sumatra", 
     xlab = "Year", ylab = "Coffee Productivity (tons/ha)", col = "black", , type = "o", lwd = 1)

# Create the group variable for Levene's test based on the updated years
group <- ifelse(train_data$Year <= 2010, "2000-2010", "2011-2020")

# Perform Levene's test on the original data
levene_result <- levene.test(train_data$`Productivity (tons/ha)`, group, location = "mean")
print("Levene's test result on original data:")
print(levene_result)


# Load necessary package for Box-Cox transformation
if (!require(forecast)) install.packages("forecast", dependencies = TRUE)
library(forecast)

# Extract productivity values from training data
train_values <- train_data$`Productivity (tons/ha)`

# Apply Box-Cox transformation
# Box-Cox requires all values to be positive, ensure this condition
if (min(train_values) <= 0) {
  train_values <- train_values + abs(min(train_values)) + 0.01
}

# Use Box-Cox transformation to find the optimal lambda
box_cox_result <- BoxCox.lambda(train_values, method = "loglik")

# Perform the actual Box-Cox transformation
transformed_train_values <- BoxCox(train_values, lambda = box_cox_result)

# Check the transformation
print(paste("Optimal lambda for Box-Cox transformation:", box_cox_result))
print("Transformed training data:")
print(transformed_train_values)

# Plot transformed data
plot(transformed_train_values, main = "Transformed Data (Box-Cox)", ylab = "Transformed Productivity", xlab = "Year", col = "black", type = "o")

# **Re-evaluate Homoscedasticity after Transformation**

# Convert transformed data to time series
transformed_ts <- ts(transformed_train_values, frequency = 1)

# Create grouping variable for transformed data
group_transformed <- ifelse(2001:2019 <= 2009, "2001-2009", "2010-2019")

# Perform Levene's test on transformed data
levene_result_transformed <- levene.test(transformed_ts, group_transformed, location = "mean")

# Print the result
print(levene_result_transformed)

# Perform the ADF Test on the transformed data
adf_test_transformed <- adf.test(transformed_ts)
print("ADF Test on Transformed Data:")
print(adf_test_transformed)

# Perform first differencing on the transformed data
diff_transformed_ts <- diff(transformed_ts, differences = 1)

# Perform ADF test on the differenced data
adf_test_diff <- adf.test(diff_transformed_ts)
print("ADF Test on Differenced Transformed Data:")
print(adf_test_diff)

# Plot the differenced data
plot(diff_transformed_ts, main = "Differenced Transformed Data", 
     xlab = "Year", ylab = "Differenced Transformed Productivity", col = "black", type = "o")

# Calculate ACF and PACF of the differenced data
acf(diff_transformed_ts, lag.max = 10, main = "ACF of Differenced Transformed Data")
pacf(diff_transformed_ts, lag.max = 10, main = "PACF of Differenced Transformed Data")

# Fit ARIMA(1,1,0)
arima_110 <- arima(coffee_ts, order = c(1, 1, 0), method = "ML")
summary(arima_110)

# Fit ARIMA(1,1,1)
arima_111 <- arima(coffee_ts, order = c(1, 1, 1), method = "ML")
summary(arima_111)

# Fit ARIMA(1,1,2)
arima_112 <- arima(coffee_ts, order = c(1, 1, 2), method = "ML")
summary(arima_112)


# Compare AIC values
aic_values <- c(arima_110$aic, arima_111$aic, arima_112$aic)
names(aic_values) <- c("ARIMA(1,1,0)", "ARIMA(1,1,1)", "ARIMA(1,1,2)")
print(aic_values)

# Select the model with the lowest AIC

# Compare models using AIC and BIC
models <- list(ARIMA_110 = arima_110, ARIMA_111 = arima_111, ARIMA_112 = arima_112)
compare_models <- data.frame(
  Model = names(models),
  AIC = sapply(models, AIC),
  BIC = sapply(models, BIC)
)
compare_models <- compare_models[order(compare_models$AIC), ]

# Print result comparation
print(compare_models)

# Function to calculate p-values for ARIMA model coefficients
get_p_values <- function(arima_model) {
  coefs <- arima_model$coef  # Coefficients
  se <- sqrt(diag(arima_model$var.coef))  # Standard errors
  z_values <- coefs / se  # Z-values
  p_values <- 2 * pnorm(-abs(z_values))  # Two-tailed p-values
  
# Combine coefficients, standard errors, Z-values, and p-values in a data frame
result <- data.frame(Coefficient = coefs, Std_Error = se, Z_value = z_values, P_value = p_values)
return(result)
}

# Function to select model based on significance of parameters (p-value < 0.05)
select_significant_model <- function(p_values_df) {
  return(all(p_values_df$P_value < 0.05))  # Check if all p-values are below 0.05
}

# Check P-values for ARIMA models:
p_values_110 <- get_p_values(arima_110)
p_values_111 <- get_p_values(arima_111)
p_values_112 <- get_p_values(arima_112)

print(p_values_110)
print(p_values_111)
print(p_values_112)


# Model Significance Check
is_significant_110 <- select_significant_model(p_values_110)
is_significant_111 <- select_significant_model(p_values_111)
is_significant_112 <- select_significant_model(p_values_112)

cat("\nModel Significance:\n")
cat("ARIMA(1,1,0):", is_significant_110, "\n")
cat("ARIMA(1,1,1):", is_significant_111, "\n")
cat("ARIMA(1,1,2):", is_significant_112, "\n")

# Residual diagnostics for the ARIMA model choosen (based on AIC/BIC and significance)
best_model <- arima_110  # best model
par(mfrow=c(3,1))
ts.plot(residuals(best_model), main="Residuals of ARIMA (1,1,0)", ylab="Residuals")
acf(residuals(best_model), main="ACF of Residuals")
pacf(residuals(best_model), main="PACF of Residuals")

# Ljung-Box test on residuals for white noise
ljung_box_test <- Box.test(residuals(best_model), lag = 10, type = "Ljung-Box")
cat("\nLjung-Box test for white noise:\n")
print(ljung_box_test)

#forecasting for 4 years to see the error

#Training and testing Data 19 years and 4 Years
# Training data from 2001 to 2019 (19 years)

#Package for training data
if (!require(fpp2)) install.packages("fpp2", dependencies = TRUE)
library(fpp2)

# Now use the forecast function
forecast(Arima(coffee_ts, order = c(1,1,0)), h = 4)
forecast(Arima(coffee_ts, order = c(1,1,1)), h = 4)
forecast(Arima(coffee_ts, order = c(1,1,2)), h = 4)

print(test_data)

# Define the test data
test_data <- c(0.600, 0.588, 0.998, 0.805)

# Function to calculate evaluation metrics
evaluate_forecast <- function(model, test_data, h) {
  # Generate the forecast
  forecast_result <- forecast(model, h = h)
  
  # Extract point forecasts
  point_forecasts <- forecast_result$mean
  
  # Calculate metrics
  mae <- mean(abs(point_forecasts - test_data))
  rmse <- sqrt(mean((point_forecasts - test_data)^2))
  mape <- mean(abs((point_forecasts - test_data) / test_data)) * 100
  
  return(c(MAE = mae, RMSE = rmse, MAPE = mape))
}

# Fit ARIMA models
model_1 <- Arima(coffee_ts, order = c(1, 1, 0))
model_2 <- Arima(coffee_ts, order = c(1, 1, 1))
model_3 <- Arima(coffee_ts, order = c(1, 1, 2))

# Evaluate models
metrics_1 <- evaluate_forecast(model_1, test_data, h = 4)
metrics_2 <- evaluate_forecast(model_2, test_data, h = 4)
metrics_3 <- evaluate_forecast(model_3, test_data, h = 4)

# Combine results into a table
results <- data.frame(
  Model = c("ARIMA(1,1,0)", "ARIMA(1,1,1)", "ARIMA(1,1,2)"),
  MAE = c(metrics_1["MAE"], metrics_2["MAE"], metrics_3["MAE"]),
  RMSE = c(metrics_1["RMSE"], metrics_2["RMSE"], metrics_3["RMSE"]),
  MAPE = c(metrics_1["MAPE"], metrics_2["MAPE"], metrics_3["MAPE"])
)

# Print results
print(results)

