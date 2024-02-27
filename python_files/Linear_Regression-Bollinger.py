# Εισαγωγή απαραίτητων βιβλιοθηκών
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Λήψη δεδομένων από το χρηματιστήριο
symbol = 'AMZN'  # Παράδειγμα με μετοχή Amazon
start_date = '2018-11-30'
end_date = '2023-11-30'
data = yf.download(symbol, start=start_date, end=end_date)

# Καθαρισμός και επεξεργασία δεδομένων
data.dropna(inplace=True)
data['Return'] = data['Close'].pct_change()

# Υπολογισμός Bollinger Bands
n = 20
data['MA'] = data['Close'].rolling(n).mean()
data['Upper_Band'] = data['MA'] + 2 * data['Close'].rolling(n).std()
data['Lower_Band'] = data['MA'] - 2 * data['Close'].rolling(n).std()

# Παραμετροποίηση βάσει Bollinger Bands
data['Overbought'] = np.where(data['Close'] > data['Upper_Band'], 1, 0)
data['Oversold'] = np.where(data['Close'] < data['Lower_Band'], 1, 0)

# Δημιουργία μεταβλητών για το μοντέλο μηχανικής μάθησης
data['Previous_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

# Διαχωρισμός σε σετ εκπαίδευσης και δοκιμής
X = data[['Previous_Close', 'Overbought', 'Oversold']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Εκπαίδευση του μοντέλου
model = LinearRegression()
model.fit(X_train, y_train)

# Πρόβλεψη τιμών
predictions = model.predict(X_test)

# Αξιολόγηση του μοντέλου
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Οπτικοποίηση των προβλέψεων
plt.scatter(X_test['Previous_Close'], y_test,
            color='black', label='Πραγματικές τιμές')
plt.scatter(X_test['Previous_Close'], predictions,
            color='blue', label='Προβλέψεις')
# plt.scatter(X_test['Overbought'], y_test, color='red', label='Πραγματικές τιμές (Overbought)')
# plt.scatter(X_test['Overbought'], predictions, color='green', label='Προβλέψεις (Overbought)')
# plt.scatter(X_test['Oversold'], y_test, color='purple', label='Πραγματικές τιμές (Oversold)')
# plt.scatter(X_test['Oversold'], predictions, color='orange', label='Προβλέψεις (Oversold)')
plt.xlabel('Τιμή Κλεισίματος Προηγούμενης Ημέρας / Bollinger Bands')
plt.ylabel('Τιμή Κλεισίματος')
plt.title('Πραγματικές τιμές vs Προβλέψεις')
plt.legend()
plt.show()
