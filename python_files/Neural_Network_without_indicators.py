# Εισαγωγή βιβλιοθηκών
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Λήψη δεδομένων από το Yahoo Finance
symbol = 'AMZN'  # Παρ΄΄αδειγμα μετοχ΄ής για την Amazon
start_date = '2018-01-01'
end_date = '2023-11-30'
df = yf.download(symbol, start=start_date, end=end_date)
df['Date'] = df.index

# Καθαρισμός των δεδομένων και προεπεξεργασία
df.isnull().sum()
df.dropna(inplace=True)

close_prices = df['Close'].values.reshape(-1, 1)

# Κανονικοποίηση δεδομένων
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Προετοιμασία δεδομένων για το μοντέλο


def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


# Διαχωρισμός σε train και test set
time_step = 60
X, y = create_dataset(scaled_data, time_step)
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size, :], X[train_size:len(X), :]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Μετατροπή σε 3D μορφή για το LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Χτίσιμο του μοντέλου LSTM
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True),
          input_shape=(time_step, 1)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(25))
model.add(Dense(1))

# Εκπαίδευση του μοντέλου
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=5)  # Εκπαίδευση για 5 εποχές

# Αξιολόγηση του μοντέλου
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Μετατροπή των δεδομένων σε αρχική μορφή
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Υπολογισμός του start index
test_start_index = len(scaled_data) - len(test_predict) - 1

trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[test_start_index:test_start_index +
                len(test_predict), :] = test_predict

# Υπολογισμός του MAE, MSE και RMSE για το train set
train_mae = mean_absolute_error(
    y_train, scaler.inverse_transform(model.predict(X_train)))
train_mse = mean_squared_error(
    y_train, scaler.inverse_transform(model.predict(X_train)))
train_rmse = np.sqrt(train_mse)

# Υπολογισμός του MAE, MSE και RMSE για το test set
test_mae = mean_absolute_error(
    y_test, scaler.inverse_transform(model.predict(X_test)))
test_mse = mean_squared_error(
    y_test, scaler.inverse_transform(model.predict(X_test)))
test_rmse = np.sqrt(test_mse)

print("Training Data Evaluation:")
print("MAE:", train_mae)
print("MSE:", train_mse)
print("RMSE:", train_rmse)

print("\nTesting Data Evaluation:")
print("MAE:", test_mae)
print("MSE:", test_mse)
print("RMSE:", test_rmse)

# Οπτικοποίηση των αποτελεσμάτων
plt.figure(figsize=(15, 7))
plt.plot(df.index, scaler.inverse_transform(
    scaled_data), label='Real Price', color='blue')
plt.plot(df.index, trainPredictPlot,
         label='Train Predict Price', color='orange')
plt.plot(df.index, testPredictPlot, label='Test Predict Price', color='green')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Πρόβλεψη τιμών για τις επόμενες 5 μέρες
future_days = 5
future_predictions = []

# Παίρνουμε τα δεδομένα για την τελευταία μέρα
last_time_step_data = scaled_data[-time_step:]

# Μετατροπή των δεδομένων σε 3D μορφή για το LSTM
current_batch = last_time_step_data.reshape((1, time_step, 1))

# Πρόβλεψη τιμών για τις επόμενες 5 μέρες
for i in range(future_days):
    future_pred = model.predict(current_batch)[0]
    future_predictions.append(future_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

# Αντιστροφή της κανονικοποίησης
future_predictions = scaler.inverse_transform(future_predictions)

# Δημιουργία Pandas Series για τις μελλοντικές ημερομηνίες και τις προβλεπόμενες τιμές
# 'B' denotes business day frequency
future_dates = pd.date_range(start=df.index[-1], periods=future_days, freq='B')
future_dates_series = pd.Series(future_dates, name='Date')
future_predictions_series = pd.Series(np.reshape(
    future_predictions, (future_days,)), name='Predicted Close')

# Συνδυασμός των dates και των προβλεπόμενων τιμών σε ένα DataFrame
future_predictions_df = pd.concat(
    [future_dates_series, future_predictions_series], axis=1)

# Οπτικοποίηση των προβλεπόμενων τιμών
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Actual Prices')

plt.plot(future_predictions_df.set_index('Date'),
         label='Future Predictions', color='red')

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()
