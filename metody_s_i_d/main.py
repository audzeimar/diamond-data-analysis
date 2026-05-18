from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.eda import (
    plot_bar_cut,
    plot_box_categorical,
    plot_boxplot,
    plot_carat_hist,
    plot_correlation,
    plot_histogram,
    plot_regression,
    plot_scatter,
)
from src.evaluation import evaluate_classification, evaluate_regression
from src.model import predict_complex, predict_regression, predict_simple

# Wczytanie zbioru danych
df = load_data()

# EDA - analiza eksploracyjna danych
plot_histogram(df)  # histogram cen
plot_scatter(df)    # zależność ceny od masy
plot_boxplot(df)    # boxplot ceny
plot_correlation(df)    # macierz korelacji
plot_carat_hist(df)     # histogram masy
plot_box_categorical(df)    # boxploty dla color i clarity
plot_bar_cut(df)    # średnia cena względem cut


# Podział danych
# 80% danych przeznaczamy na "trening", a 20% na test końcowy
train_df, test_df = train_test_split(df.copy(), test_size=0.2, random_state=42)

# Tworzenie targetu 
# Za próg przyjmujemy cenę większą niż 5000 USD
train_df['target'] = (train_df['price'] > 5000).astype(int)
test_df['target'] = (test_df['price'] > 5000).astype(int)

# Klasyfikacja
# Sprawdzamy działanie dwóch wersji modelu: prostego i bardziej złożonego
y_pred_simple = test_df.apply(predict_simple, axis=1)
acc_simple = evaluate_classification(test_df['target'], y_pred_simple)

y_pred_complex = test_df.apply(predict_complex, axis=1)
acc_complex = evaluate_classification(test_df['target'], y_pred_complex)

print('Accuracy (simple):', acc_simple)
print('Accuracy (complex):', acc_complex)

# REGRESJA
# Model regresyjny szacuje przybliżoną cenę diamentu na podstawie prostych progów masy
y_pred_reg = test_df.apply(predict_regression, axis=1)
mse, mae = evaluate_regression(test_df['price'], y_pred_reg)

print('MSE:', mse)
print('MAE:', mae)

# Generujemy wykres: ceny rzeczywiste vs przewidywane
plot_regression(test_df['price'], y_pred_reg)

# Zapis wyników
with open('results/metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f'Accuracy simple: {acc_simple}\n')
    f.write(f'Accuracy complex: {acc_complex}\n')
    f.write(f'MSE: {mse}\n')
    f.write(f'MAE: {mae}\n')
