### Developed by : RAGUNATH R
### Register no :212222240081
### Date: 

# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Astrobiological dataset.
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# Load the dataset
df = pd.read_csv('/content/dataset.csv')

# Display the first few rows of the dataset to see the data in row and column format
print("Initial Data Preview:")
print(df.head())

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Convert 'Daily minimum temperatures' column to numeric, if not already
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')

# Drop any NaN values
df.dropna(inplace=True)

# Perform Augmented Dickey-Fuller test for stationarity on 'Daily minimum temperatures'
result = adfuller(df['Daily minimum temperatures'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Plot ACF and PACF to visualize autocorrelations
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_acf(df['Daily minimum temperatures'], lags=13, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Daily minimum temperatures'], lags=13, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df['Daily minimum temperatures'][:train_size], df['Daily minimum temperatures'][train_size:]

# Fit AutoRegressive model with 13 lags (based on PACF observation)
model = AutoReg(train, lags=13)
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the test data and the predictions
plt.plot(test.index, test, label='Actual', marker='o')
plt.plot(test.index, predictions, label='Predicted', marker='x')
plt.title('Daily Minimum Temperatures - Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

```

### OUTPUT:


PACF - ACF :

![WhatsApp Image 2024-10-16 at 13 50 53_b7d238c8](https://github.com/user-attachments/assets/dc8af820-2aa5-4d99-bc0c-db75dc7c5970)



PREDICTION :

![WhatsApp Image 2024-10-16 at 13 51 14_24ca083b](https://github.com/user-attachments/assets/b61b1835-57f8-4db0-909b-1aa5829c24af)


### RESULT:
Thus the python code successfully implemented the auto regression function.
