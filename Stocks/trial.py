import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

# Load the preprocessed data
data = pd.read_csv('D:/Program Files/project/Data_Windowing/SBILIFE.csv', parse_dates=['Date'])

# Sort data by date
data = data.sort_values('Date')

# Set the date as the index
data.set_index('Date', inplace=True)

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Keep only the last 150 days of data
data_last_150_days = data

# Prepare the dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into train and test sets
training_size = int(len(X) * 0.8)
test_size = len(X) - training_size
X_train, X_test = X[0:training_size], X[training_size:len(X)]
y_train, y_test = y[0:training_size], y[training_size:len(y)]

# Create the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate the RMSE
train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))
test_accuracy = 100 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))) * 100)

# Predicting the next 5 days
x_input = scaled_data[-time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = time_step
for i in range(5):  # Predicting for 5 days
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

future_predictions = scaler.inverse_transform(lst_output)

# Plotting the results
def plot_predictions(data_last_150_days, train_predict, test_predict, future_predictions):
    train_data_len = len(train_predict)
    test_data_len = len(test_predict)
    future_data_len = len(future_predictions)

    plt.figure(figsize=(14, 8))

    # Plotting actual prices for the last 150 days
    plt.plot(data_last_150_days.index, data_last_150_days['Close'], color='red', label='Actual Price')

    # Plotting predicted prices
    plt.plot(data.index[time_step:train_data_len + time_step], train_predict, color='blue', linestyle='solid', label='Train Predict Price')
    test_start_index = train_data_len + time_step
    plt.plot(data.index[test_start_index:test_start_index + test_data_len], test_predict, color='blue', linestyle='solid', label='Test Predict Price')

    # Plotting future predictions
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=6)[1:]  # Adjusted to start from the last date and predict for 30 days
    plt.plot(future_dates, future_predictions, color='green', linestyle='solid', label='Future 5 Days Predict Price')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit app
st.title('Stock Price Prediction')
st.write(f'Train RMSE: {train_rmse}')
st.write(f'Test RMSE: {test_rmse}')
st.write(f'Test Accuracy: {test_accuracy:.2f}%')

# Plot
plot_predictions(data_last_150_days, train_predict, test_predict, future_predictions)

st.write('## Future Predictions for the Next 5 Days')
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])
st.write(future_df)










# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# # Load the preprocessed data
# data = pd.read_csv('D:\Program Files\project\Data_Windowing\SBILIFE.csv', parse_dates=['Date'])

# # Sort data by date
# data = data.sort_values('Date')

# # Set the date as the index
# data.set_index('Date', inplace=True)

# # Feature Scaling
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data[['Close']])

# # Prepare the dataset for LSTM
# def create_dataset(dataset, time_step=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - time_step - 1):
#         a = dataset[i:(i + time_step), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + time_step, 0])
#     return np.array(dataX), np.array(dataY)

# time_step = 30
# X, y = create_dataset(scaled_data, time_step)

# # Reshape input to be [samples, time steps, features] which is required for LSTM
# X = X.reshape(X.shape[0], X.shape[1], 1)

# # Split the data into train and test sets
# training_size = int(len(X) * 0.8)
# test_size = len(X) - training_size
# X_train, X_test = X[0:training_size], X[training_size:len(X)]
# y_train, y_test = y[0:training_size], y[training_size:len(y)]

# # Create the LSTM model
# model = Sequential()
# model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
# model.add(LSTM(100, return_sequences=False))
# model.add(Dense(50))
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Define early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)

# # Predictions
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Transform back to original scale
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)

# # Calculate the RMSE
# train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
# test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))
# print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

# # Calculate the accuracy
# test_accuracy = 100 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))) * 100)
# print(f'Test Accuracy: {test_accuracy:.2f}%')

# # Predicting the next 100 days
# x_input = scaled_data[-time_step:].reshape(1, -1)
# temp_input = list(x_input)
# temp_input = temp_input[0].tolist()

# lst_output = []
# n_steps = time_step
# for i in range(10):  # Predicting for 100 days
#     if len(temp_input) > time_step:
#         x_input = np.array(temp_input[1:])
#         x_input = x_input.reshape(1, -1)
#         x_input = x_input.reshape((1, n_steps, 1))
#         yhat = model.predict(x_input, verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         temp_input = temp_input[1:]
#         lst_output.extend(yhat.tolist())
#     else:
#         x_input = x_input.reshape((1, n_steps, 1))
#         yhat = model.predict(x_input, verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         lst_output.extend(yhat.tolist())

# future_predictions = scaler.inverse_transform(lst_output)

# # Plotting the results
# train_data_len = len(train_predict)
# test_data_len = len(test_predict)
# future_data_len = len(future_predictions)

# plt.figure(figsize=(14, 8))

# # Plotting actual prices
# plt.plot(data.index, data['Close'], color='red', label='Actual Price')

# # Plotting predicted prices
# plt.plot(data.index[time_step:train_data_len + time_step], train_predict, color='blue', linestyle='dashed', label='Train Predict Price')
# test_start_index = train_data_len + time_step
# plt.plot(data.index[test_start_index:test_start_index + test_data_len], test_predict, color='blue', linestyle='dashed', label='Test Predict Price')

# # Plotting future predictions
# last_date = data.index[-1]
# future_dates = pd.date_range(start=last_date, periods=11)[1:]  # Adjusted to start from the last date and predict for 100 days
# plt.plot(future_dates, future_predictions, color='green', linestyle='solid', label='Future 100 Days Predict Price')

# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.title('Stock Price Prediction')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
