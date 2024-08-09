

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the preprocessed data
data = pd.read_csv('D:\Program Files\STOCK_PREDICTION\Data_Windowing\CIPLA.csv', parse_dates=['Date'])

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
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))  # Reduced complexity
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate the RMSE
train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))
print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

# Calculate the accuracy
test_accuracy = 100 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))) * 100)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Predicting the next 30 days
x_input = scaled_data[-time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = time_step
for i in range(5):  # Predicting for 30 days
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
plt.show()











# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# # Load the preprocessed data
# data = pd.read_csv('D:\Program Files\project\Data_Windowing\CIPLA.csv', parse_dates=['Date'])

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

# # Predicting the next 10 days
# x_input = scaled_data[-time_step:].reshape(1, -1)
# temp_input = list(x_input)
# temp_input = temp_input[0].tolist()

# lst_output = []
# n_steps = time_step
# for i in range(10):  # Predicting for 10 days
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
# future_dates = pd.date_range(start=last_date, periods=11)[1:]  # Adjusted to start from the last date
# plt.plot(future_dates, future_predictions, color='green', linestyle='solid', label='Future 10 Days Predict Price')

# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.title('Stock Price Prediction')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


















# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# # Load the preprocessed data
# data = pd.read_csv('D:\Program Files\project\Data_Windowing\CIPLA.csv', parse_dates=['Date'])

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

# time_step = 60
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
# model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
# model.add(LSTM(50, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, batch_size=1, epochs=1)

# # Predictions
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Transform back to original scale
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)

# # Calculate the RMSE
# train_rmse = np.sqrt(np.mean((train_predict - scaler.inverse_transform(y_train.reshape(-1, 1)))**2))
# test_rmse = np.sqrt(np.mean((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
# print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

# # Predicting the next 3 days
# x_input = scaled_data[-time_step:].reshape(1, -1)
# temp_input = list(x_input)
# temp_input = temp_input[0].tolist()

# lst_output = []
# n_steps = time_step
# i = 0
# while (i < 3):
#     if (len(temp_input) > time_step):
#         x_input = np.array(temp_input[1:])
#         x_input = x_input.reshape(1, -1)
#         x_input = x_input.reshape((1, n_steps, 1))
#         yhat = model.predict(x_input, verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         temp_input = temp_input[1:]
#         lst_output.extend(yhat.tolist())
#         i = i + 1
#     else:
#         x_input = x_input.reshape((1, n_steps, 1))
#         yhat = model.predict(x_input, verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         lst_output.extend(yhat.tolist())
#         i = i + 1

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
# future_dates = pd.date_range(start=last_date, periods=4)[1:]  # Exclude the start date to match future_predictions
# plt.plot(future_dates, future_predictions, color='green', label='Future 3 Days Predict Price')

# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.title('Stock Price Prediction')
# plt.legend()
# plt.show()































































































# import numpy as np
# import pandas as pd
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from keras.layers import LSTM, Dense
# from keras import Sequential
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # Load the preprocessed data
# data = pd.read_csv(r'D:\Program Files\project\Data_Windowing\BPCL.csv')

# # Convert date column to datetime
# data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# # Drop the date column for scaling and model training
# data = data.drop(columns=['Date'])

# # Split the data into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# # Scale the data
# # scaler = MinMaxScaler()
# # train_data_scaled = scaler.fit_transform(train_data)
# # test_data_scaled = scaler.transform(test_data)

# scaler = StandardScaler()
# train_data_scaled = scaler.fit_transform(train_data)
# test_data_scaled = scaler.transform(test_data)

# # Prepare the input and output data for LSTM model
# X_train, y_train = train_data_scaled[:, :-1], train_data_scaled[:, -1]
# X_test, y_test = test_data_scaled[:, :-1], test_data_scaled[:, -1]

# # Reshape the input data for LSTM model
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(1024, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mse')

# # Train the LSTM model
# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# # Evaluate the model on test data
# loss = model.evaluate(X_test, y_test)

# # Make predictions using the trained model
# predictions = model.predict(X_test)

# # Inverse transform the scaled data for the target variable
# y_test_inverse = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[1]), y_test.reshape(-1, 1)), axis=1))[:, -1]
# predictions_inverse = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[1]), predictions), axis=1))[:, -1]

# # Plot the actual and predicted values
# plt.plot(y_test_inverse, label='Actual')
# plt.plot(predictions_inverse, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.title('Actual vs Predicted Stock Prices')
# plt.legend()
# plt.show()



# # import numpy as np
# # import pandas as pd
# # from tensorflow import keras
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.layers import LSTM, Dense, Dropout
# # from tensorflow.keras.models import Sequential
# # import matplotlib.pyplot as plt
# # from tensorflow.keras.callbacks import EarlyStopping

# # # Load the preprocessed data
# # data = pd.read_csv(r'D:\Program Files\project\Data_Windowing\BPCL.csv')

# # # Convert date column to datetime
# # data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# # # Drop the date column for scaling and model training
# # data = data.drop(columns=['Date'])

# # # Scale the data
# # scaler = MinMaxScaler()
# # data_scaled = scaler.fit_transform(data)

# # # Function to create sequences
# # def create_sequences(data, seq_length):
# #     X = []
# #     y = []
# #     for i in range(len(data) - seq_length):
# #         X.append(data[i:i + seq_length, :-1])
# #         y.append(data[i + seq_length - 1, -1])
# #     return np.array(X), np.array(y)

# # # Define sequence length
# # seq_length = 60  # Use 60 past days to predict the next day's stock price

# # # Create sequences
# # X, y = create_sequences(data_scaled, seq_length)

# # # Split the data into training and testing sets
# # split = int(0.8 * len(X))
# # X_train, X_test = X[:split], X[split:]
# # y_train, y_test = y[:split], y[split:]

# # # Build the LSTM model
# # model = Sequential()
# # model.add(LSTM(512, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
# # model.add(Dropout(0.2))
# # model.add(LSTM(512, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(1))

# # # Compile the model
# # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# # # Implement Early Stopping
# # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # # Train the LSTM model
# # history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# # # Evaluate the model on test data
# # loss = model.evaluate(X_test, y_test)

# # # Make predictions using the trained model
# # predictions = model.predict(X_test)

# # # Inverse transform the scaled data for the target variable
# # y_test_expanded = np.zeros((len(y_test), data.shape[1]))
# # y_test_expanded[:, :-1] = X_test[:, -1, :]
# # y_test_expanded[:, -1] = y_test
# # y_test_inverse = scaler.inverse_transform(y_test_expanded)[:, -1]

# # predictions_expanded = np.zeros((len(predictions), data.shape[1]))
# # predictions_expanded[:, :-1] = X_test[:, -1, :]
# # predictions_expanded[:, -1] = predictions.flatten()
# # predictions_inverse = scaler.inverse_transform(predictions_expanded)[:, -1]

# # # Plot the actual and predicted values
# # plt.plot(y_test_inverse, label='Actual')
# # plt.plot(predictions_inverse, label='Predicted')
# # plt.xlabel('Time')
# # plt.ylabel('Stock Price')
# # plt.title('Actual vs Predicted Stock Prices')
# # plt.legend()
# # plt.show()

# # # Plot training and validation loss
# # plt.plot(history.history['loss'], label='Training Loss')
# # plt.plot(history.history['val_loss'], label='Validation Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.title('Training and Validation Loss')
# # plt.legend()
# # plt.show()
