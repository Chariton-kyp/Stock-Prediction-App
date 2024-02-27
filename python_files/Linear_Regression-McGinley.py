# Εισαγωγή απαραίτητων βιβλιοθηκών
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Λήψη δεδομένων από το χρηματιστήριο
symbol = 'AMZN'  # Παράδειγμα με μετοχή Amazon
start_date = '2018-11-30'
end_date = '2023-11-30'
data = yf.download(symbol, start=start_date, end=end_date)

# Καθαρισμός και επεξεργασία δεδομένων
data.dropna(inplace=True)  # Διαγραφή απουσιάζουσων τιμών
data['Return'] = data['Close'].pct_change()  # Υπολογισμός ημερήσιας απόδοσης

# Υπολογισμός του McGinley Dynamic


def mcginley_dynamic(df, n):
    df['MD'] = df['Close'] * \
        (1 + (df['Close'] - df['Close'].expanding(n).mean()) /
         df['Close'].expanding(n).mean()) ** 2
    return df


# Εφαρμογή του McGinley Dynamic στα δεδομένα
data = mcginley_dynamic(data, 24)

# Οπτικοποίηση δεδομένων
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Κλείσιμο')
plt.plot(data['MD'], label='McGinley Dynamic')
plt.title(f'Ιστορική Τιμή Κλεισίματος {symbol}')
plt.legend()
plt.show()

# Δημιουργία μεταβλητών για το μοντέλο μηχανικής μάθησης
data['Previous_Close'] = data['Close'].shift(
    1)  # Τιμή κλεισίματος προηγούμενης ημέρας
data.dropna(inplace=True)  # Διαγραφή απουσιάζουσων τιμών μετά τη μετατόπιση

# Διαχωρισμός σε σετ εκπαίδευσης και δοκιμής
X = data[['Previous_Close', 'MD']]
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
# plt.scatter(X_test['MD'], y_test, color='red', label='Πραγματικές τιμές (McGinley Dynamic)')
# plt.scatter(X_test['MD'], predictions, color='green', label='Προβλέψεις (McGinley Dynamic)')
plt.xlabel('Τιμή Κλεισίματος Προηγούμενης Ημέρας / McGinley Dynamic')
plt.ylabel('Τιμή Κλεισίματος')
plt.title('Πραγματικές τιμές vs Προβλέψεις μετοχής APPLE')
plt.legend()
plt.show()
