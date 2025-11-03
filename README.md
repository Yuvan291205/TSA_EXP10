# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 03.11.2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('cars (1).csv')


def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

target_variable = 'CO2' # Changed to 'CO2'
train_size = int(len(data) * 0.8)
train, test = data[target_variable][:train_size], data[target_variable][train_size:]

# Since there is no inherent seasonality in this dataset based on the column names,
# I will remove the seasonal_order from the SARIMAX model for now.
sarima_model = SARIMAX(train, order=(1, 1, 1)) # Removed seasonal_order
sarima_result = sarima_model.fit(disp=False)
predictions = sarima_result.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Testing Data')
plt.plot(test.index, predictions, label='Predicted Data', color='orange')
plt.title(f"SARIMA Forecast for {target_variable}")
plt.xlabel("Index") # Changed xlabel as there is no Date column
plt.ylabel(target_variable)
plt.legend()
plt.show()

# Re-adding the stationarity checks and plots for the 'CO2' column
check_stationarity(data[target_variable])
plot_acf(data[target_variable])
plt.title(f"ACF for {target_variable}")
plt.show()
plot_pacf(data[target_variable])
plt.title(f"PACF for {target_variable}")
plt.show()

```
### OUTPUT:

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/ef86a358-dae7-458b-a0c3-dc65ea8f39bf" />
ADF Statistic: -1.927441468616781
p-value: 0.3192639129359401
Critical Values:
	1%: -3.6327426647230316
	5%: -2.9485102040816327
	10%: -2.6130173469387756
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/1605c1a7-fe06-4bb9-a419-5e039e217f4d" />
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/662c9c9f-b1bd-49b3-8a5b-bee296012d4c" />

  

### RESULT:
Thus the program run successfully based on the SARIMA model.
