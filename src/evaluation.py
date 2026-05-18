from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report

# Funkcje oceny modeli klasyfikacyjnych i regresyjnych
def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)  # Obliczenie dokładności jako proporcji poprawnych predykcji
    report = classification_report(y_true, y_pred)  # Generowanie raportu klasyfikacji
    return acc, report  

# Funkcja oceny modeli regresyjnych: oblicza MSE i MAE
def evaluate_regression(y_true, y_pred):
    
    # Oblicza podstawowe metryki regresji:
    # - MSE: średni błąd kwadratowy
    # - MAE: średni błąd bezwzględny
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae