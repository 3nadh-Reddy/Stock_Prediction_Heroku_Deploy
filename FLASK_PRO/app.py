from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import io
import base64
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

with open('FLASK_PRO/nifty50_list.pickle', 'rb') as f:
    tickers = pickle.load(f)

time_step = 60

def load_model_and_predict(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    training_size = int(len(X) * 0.8)
    test_size = len(X) - training_size
    X_train, X_test = X[0:training_size], X[training_size:len(X)]
    y_train, y_test = y[0:training_size], y[training_size:len(y)]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[early_stopping], verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
    test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))
    test_accuracy = 100 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))) * 100)

    x_input = scaled_data[-time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []

    for i in range(5):
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
        else:
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())

    future_predictions = scaler.inverse_transform(lst_output)
    return data, train_predict, test_predict, future_predictions, test_accuracy

def plot_predictions(data, train_predict, test_predict, future_predictions):
    train_data_len = len(train_predict)
    test_data_len = len(test_predict)
    future_data_len = len(future_predictions)

    train_trace = go.Scatter(
        x=data.index[time_step:train_data_len + time_step],
        y=train_predict.flatten(),
        mode='lines',
        name='Train Predict Price',
        line=dict(color='blue')
    )

    test_trace = go.Scatter(
        x=data.index[train_data_len + time_step:train_data_len + time_step + test_data_len],
        y=test_predict.flatten(),
        mode='lines',
        name='Test Predict Price',
        line=dict(color='orange')
    )

    future_trace = go.Scatter(
        x=pd.date_range(start=data.index[-1], periods=6)[1:],
        y=future_predictions.flatten(),
        mode='lines',
        name='Future 5 Days Predict Price',
        line=dict(color='green')
    )

    actual_trace = go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='red')
    )

    layout = go.Layout(
        title='Stock Price Prediction',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price'),
        hovermode='closest'
    )

    fig = go.Figure(data=[actual_trace, train_trace, test_trace, future_trace], layout=layout)

    plot_html = pio.to_html(fig, full_html=False)

    return plot_html

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected_ticker = request.form['ticker']
        return redirect(url_for('results', ticker=selected_ticker))
    
    return render_template('index.html', tickers=tickers)

@app.route('/results/<ticker>')
def results(ticker):
    data = pd.read_csv(f'D:/Program Files/STOCK_PREDICTION/Data_Windowing/{ticker}.csv', parse_dates=['Date'])
    data = data.sort_values('Date')
    data.set_index('Date', inplace=True)
    data, train_predict, test_predict, future_predictions, accuracy = load_model_and_predict(data)
    plot_html = plot_predictions(data, train_predict, test_predict, future_predictions)
    predicted_values = future_predictions.flatten().tolist()

    return render_template('results.html', ticker=ticker, plot_html=plot_html, accuracy=accuracy, predicted_values=predicted_values)

if __name__ == '__main__':
    app.run(debug=True)




















# from flask import Flask, render_template, request, redirect, url_for
# import pandas as pd
# import numpy as np
# import io
# import base64
# import pickle
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import plotly.graph_objs as go
# import plotly.io as pio

# app = Flask(__name__)

# with open('nifty50_list.pickle', 'rb') as f:
#     tickers = pickle.load(f)

# time_step = 60

# def load_model_and_predict(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data[['Close']])

#     def create_dataset(dataset, time_step=1):
#         dataX, dataY = [], []
#         for i in range(len(dataset) - time_step - 1):
#             a = dataset[i:(i + time_step), 0]
#             dataX.append(a)
#             dataY.append(dataset[i + time_step, 0])
#         return np.array(dataX), np.array(dataY)

#     X, y = create_dataset(scaled_data, time_step)
#     X = X.reshape(X.shape[0], X.shape[1], 1)

#     training_size = int(len(X) * 0.8)
#     test_size = len(X) - training_size
#     X_train, X_test = X[0:training_size], X[training_size:len(X)]
#     y_train, y_test = y[0:training_size], y[training_size:len(y)]

#     model = Sequential()
#     # model.add(LSTM(60, return_sequences=True, input_shape=(time_step, 1)))
#     # model.add(Dropout(0.3))
#     # model.add(LSTM(120, return_sequences=False))
#     # model.add(Dropout(0.3))
#     # model.add(Dense(20))
#     # model.add(Dense(1))
#     model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
#     model.add(LSTM(100, return_sequences=False))
#     model.add(Dense(50))
#     model.add(Dense(1))

#     model.compile(optimizer='adam', loss='mean_squared_error')

#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)

#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)

#     train_predict = scaler.inverse_transform(train_predict)
#     test_predict = scaler.inverse_transform(test_predict)

#     train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
#     test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))
#     test_accuracy = 100 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))) * 100)
#     x_input = scaled_data[-time_step:].reshape(1, -1)
#     temp_input = list(x_input)
#     temp_input = temp_input[0].tolist()
#     lst_output = []
#     for i in range(5):
#         if len(temp_input) > time_step:
#             x_input = np.array(temp_input[1:])
#             x_input = x_input.reshape(1, -1)
#             x_input = x_input.reshape((1, time_step, 1))
#             yhat = model.predict(x_input, verbose=0)
#             temp_input.extend(yhat[0].tolist())
#             temp_input = temp_input[1:]
#             lst_output.extend(yhat.tolist())
#         else:
#             x_input = x_input.reshape((1, time_step, 1))
#             yhat = model.predict(x_input, verbose=0)
#             temp_input.extend(yhat[0].tolist())
#             lst_output.extend(yhat.tolist())
#     future_predictions = scaler.inverse_transform(lst_output)
#     return data, train_predict, test_predict, future_predictions, test_accuracy

# def plot_predictions(data, train_predict, test_predict, future_predictions):
#     train_data_len = len(train_predict)
#     test_data_len = len(test_predict)
#     future_data_len = len(future_predictions)

#     train_trace = go.Scatter(
#         x=data.index[time_step:train_data_len + time_step],
#         y=train_predict.flatten(),
#         mode='lines',
#         name='Train Predict Price',
#         line=dict(color='blue')
#     )

#     test_trace = go.Scatter(
#         x=data.index[train_data_len + time_step:train_data_len + time_step + test_data_len],
#         y=test_predict.flatten(),
#         mode='lines',
#         name='Test Predict Price',
#         line=dict(color='orange')
#     )

#     future_trace = go.Scatter(
#         x=pd.date_range(start=data.index[-1], periods=6)[1:],
#         y=future_predictions.flatten(),
#         mode='lines',
#         name='Future 5 Days Predict Price',
#         line=dict(color='green')
#     )

#     actual_trace = go.Scatter(
#         x=data.index,
#         y=data['Close'],
#         mode='lines',
#         name='Actual Price',
#         line=dict(color='red')
#     )

#     layout = go.Layout(
#         title='Stock Price Prediction',
#         xaxis=dict(title='Date'),
#         yaxis=dict(title='Stock Price'),
#         hovermode='closest'
#     )

#     fig = go.Figure(data=[actual_trace, train_trace, test_trace, future_trace], layout=layout)

#     plot_html = pio.to_html(fig, full_html=False)

#     return plot_html

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         selected_ticker = request.form['ticker']
#         return redirect(url_for('results', ticker=selected_ticker))
#     return render_template('index.html', tickers=tickers)

# @app.route('/results/<ticker>')
# def results(ticker):
#     data = pd.read_csv(f'D:/Program Files/STOCK_PREDICTION/Data_Windowing/{ticker}.csv', parse_dates=['Date'])
#     data = data.sort_values('Date')
#     data.set_index('Date', inplace=True)
#     data, train_predict, test_predict, future_predictions, accuracy = load_model_and_predict(data)
#     plot_html = plot_predictions(data, train_predict, test_predict, future_predictions)
#     predicted_values = future_predictions.flatten().tolist()

#     return render_template('results.html', ticker=ticker, plot_html=plot_html, accuracy=accuracy, predicted_values=predicted_values)

# if __name__ == '__main__':
#     app.run(debug=True)































# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# import io
# import base64
# import pickle
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# import plotly.graph_objs as go
# import plotly.io as pio

# app = Flask(__name__)

# with open('nifty50_list.pickle', 'rb') as f:
#     tickers = pickle.load(f)

# time_step = 60

# def load_model_and_predict(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data[['Close']])

#     def create_dataset(dataset, time_step=1):
#         dataX, dataY = [], []
#         for i in range(len(dataset) - time_step - 1):
#             a = dataset[i:(i + time_step), 0]
#             dataX.append(a)
#             dataY.append(dataset[i + time_step, 0])
#         return np.array(dataX), np.array(dataY)

#     X, y = create_dataset(scaled_data, time_step)
#     X = X.reshape(X.shape[0], X.shape[1], 1)

#     training_size = int(len(X) * 0.8)
#     test_size = len(X) - training_size
#     X_train, X_test = X[0:training_size], X[training_size:len(X)]
#     y_train, y_test = y[0:training_size], y[training_size:len(y)]

#     model = Sequential()
#     model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
#     model.add(LSTM(100, return_sequences=False))
#     model.add(Dense(50))
#     model.add(Dense(1))

#     model.compile(optimizer='adam', loss='mean_squared_error')

#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)

#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)

#     train_predict = scaler.inverse_transform(train_predict)
#     test_predict = scaler.inverse_transform(test_predict)

#     train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
#     test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))
#     test_accuracy = 100 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))) * 100)
#     x_input = scaled_data[-time_step:].reshape(1, -1)
#     temp_input = list(x_input)
#     temp_input = temp_input[0].tolist()
#     lst_output = []
#     for i in range(5):
#         if len(temp_input) > time_step:
#             x_input = np.array(temp_input[1:])
#             x_input = x_input.reshape(1, -1)
#             x_input = x_input.reshape((1, time_step, 1))
#             yhat = model.predict(x_input, verbose=0)
#             temp_input.extend(yhat[0].tolist())
#             temp_input = temp_input[1:]
#             lst_output.extend(yhat.tolist())
#         else:
#             x_input = x_input.reshape((1, time_step, 1))
#             yhat = model.predict(x_input, verbose=0)
#             temp_input.extend(yhat[0].tolist())
#             lst_output.extend(yhat.tolist())
#     future_predictions = scaler.inverse_transform(lst_output)
#     return data, train_predict, test_predict, future_predictions, test_accuracy

# def plot_predictions(data, train_predict, test_predict, future_predictions):
#     train_data_len = len(train_predict)
#     test_data_len = len(test_predict)
#     future_data_len = len(future_predictions)

#     train_trace = go.Scatter(
#         x=data.index[time_step:train_data_len + time_step],
#         y=train_predict.flatten(),
#         mode='lines',
#         name='Train Predict Price',
#         line=dict(color='blue')
#     )

#     test_trace = go.Scatter(
#         x=data.index[train_data_len + time_step:train_data_len + time_step + test_data_len],
#         y=test_predict.flatten(),
#         mode='lines',
#         name='Test Predict Price',
#         line=dict(color='orange')
#     )

#     future_trace = go.Scatter(
#         x=pd.date_range(start=data.index[-1], periods=6)[1:],
#         y=future_predictions.flatten(),
#         mode='lines',
#         name='Future 5 Days Predict Price',
#         line=dict(color='green')
#     )

#     actual_trace = go.Scatter(
#         x=data.index,
#         y=data['Close'],
#         mode='lines',
#         name='Actual Price',
#         line=dict(color='red')
#     )

#     layout = go.Layout(
#         title='Stock Price Prediction',
#         xaxis=dict(title='Date'),
#         yaxis=dict(title='Stock Price'),
#         hovermode='closest'
#     )

#     fig = go.Figure(data=[actual_trace, train_trace, test_trace, future_trace], layout=layout)

#     plot_html = pio.to_html(fig, full_html=False)

#     return plot_html

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     plot_html = None
#     accuracy = None
#     selected_ticker = None
#     predicted_values = None

#     if request.method == 'POST':
#         selected_ticker = request.form['ticker']
#         data = pd.read_csv(f'D:/Program Files/STOCK_PREDICTION/Data_Windowing/{selected_ticker}.csv', parse_dates=['Date'])
#         data = data.sort_values('Date')
#         data.set_index('Date', inplace=True)
#         data, train_predict, test_predict, future_predictions, accuracy = load_model_and_predict(data)
#         plot_html = plot_predictions(data, train_predict, test_predict, future_predictions)
#         predicted_values = future_predictions.flatten().tolist()

#     return render_template('index.html', tickers=tickers, plot_html=plot_html, accuracy=accuracy, predicted_values=predicted_values)

# if __name__ == '__main__':
#     app.run(debug=True)