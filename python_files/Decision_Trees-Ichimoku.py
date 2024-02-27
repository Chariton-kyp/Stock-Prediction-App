import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Λήψη δεδομένων μετοχής από το yfinance
symbol = 'AMZN'  # Παράδειγμα με μετοχή Amazon
start_date = '2018-11-30'
end_date = '2023-11-30'
data = yf.download(symbol, start=start_date, end=end_date)

# Υπολογισμός στοιχείων Ichimoku
data['Conversion_Line'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
data['Base_Line'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
data['Leading_Span_A'] = (data['Conversion_Line'] + data['Base_Line']) / 2
data['Leading_Span_B'] = (data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2

# Επιλογή τιμής κλεισίματος ως χαρακτηριστικό
features = data[['Close', 'Conversion_Line', 'Base_Line', 'Leading_Span_A', 'Leading_Span_B']]

# Εισαγωγή της μελλοντικής τιμής κλεισίματος ως μεταβλητή πρόβλεψης
features['Next_Close'] = features['Close'].shift(-1)

# Αφαίρεση των τελευταίων γραμμών που περιέχουν NaN λόγω του shift
features = features.dropna()

# Επιλογή των χαρακτηριστικών και της μεταβλητής πρόβλεψης
X = features[['Close', 'Conversion_Line', 'Base_Line', 'Leading_Span_A', 'Leading_Span_B']]
y = features['Next_Close']

# Διαίρεση των δεδομένων σε σύνολα εκπαίδευσης και ελέγχου
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Δημιουργία και εκπαίδευση του μοντέλου Δέντρα Απόφασης
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
predictions = model.predict(X_test)

# Αξιολόγηση του μοντέλου
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Παρουσίαση των πραγματικών τιμών και των προβλέψεων
plt.scatter(X_test['Close'], y_test, color='black', label='Πραγματικές τιμές')
plt.scatter(X_test['Close'], predictions, color='blue', label='Προβλέψεις')
plt.xlabel('Τιμή Κλεισίματος')
plt.ylabel('Επόμενη Τιμή Κλεισίματος')
plt.title('Πραγματικές τιμές vs Προβλέψεις')
plt.legend()
plt.show()
