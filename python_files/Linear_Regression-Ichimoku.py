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

# Υπολογισμός Ichimoku Cloud
conversion_line_period = 9
base_line_period = 26
lagging_span_period = 52

data['Conversion_Line'] = (data['High'].rolling(window=conversion_line_period).max() + data['Low'].rolling(
    window=conversion_line_period).min()) / 2
data['Base_Line'] = (data['High'].rolling(window=base_line_period).max() + data['Low'].rolling(
    window=base_line_period).min()) / 2
data['Leading_Span_A'] = (data['Conversion_Line'] + data['Base_Line']) / 2
data['Leading_Span_B'] = (data['High'].rolling(window=lagging_span_period).max() + data['Low'].rolling(
    window=lagging_span_period).min()) / 2

# Παραμετροποίηση βάσει Ichimoku Cloud
cloud_data = data[['Leading_Span_A', 'Leading_Span_B']].shift()
data['Above_Cloud'] = np.where(data['Close'] > cloud_data.min(axis=1), 1, 0)
data['Below_Cloud'] = np.where(data['Close'] < cloud_data.min(axis=1), 1, 0)


# Δημιουργία μεταβλητών για το μοντέλο μηχανικής μάθησης
data['Previous_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

# Διαχωρισμός σε σετ εκπαίδευσης και δοκιμής
X = data[['Previous_Close', 'Above_Cloud', 'Below_Cloud']]
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
# plt.scatter(X_test['Above_Cloud'], y_test, color='red', label='Πραγματικές τιμές (Above Cloud)')
# plt.scatter(X_test['Above_Cloud'], predictions, color='green', label='Προβλέψεις (Above Cloud)')
# plt.scatter(X_test['Below_Cloud'], y_test, color='purple', label='Πραγματικές τιμές (Below Cloud)')
# plt.scatter(X_test['Below_Cloud'], predictions, color='orange', label='Προβλέψεις (Below Cloud)')
plt.xlabel('Τιμή Κλεισίματος Προηγούμενης Ημέρας / Ichimoku Cloud')
plt.ylabel('Τιμή Κλεισίματος')
plt.title('Πραγματικές τιμές vs Προβλέψεις')
plt.legend()
plt.show()
