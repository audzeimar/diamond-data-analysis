import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from src.data_loader import load_data  # Wczytywanie zbioru danych diamentów
from src.evaluation import evaluate_classification, evaluate_regression  # Funkcje oceny modeli
from src.models_scratch import calculate_entropy, information_gain, CustomLinearRegression  # Własne implementacje
from src.eda import (
    plot_feature_importance,  # Wykres ważności cech
    plot_weight_comparison,  # Porównanie wag przed/po skalowaniu
    plot_regression_results,  # Wyniki regresji
    plot_bias_variance_curve  # Krzywe bias-variance
)

def main():
    # Bufor na wyniki do raportu tekstowego
    report_lines = []
    
    def log(text=""):
        print(text)  # Wyświetlanie w konsoli
        report_lines.append(text)  # Dodanie do bufora raportu

    # 1. Wczytanie danych i przygotowanie zbiorów treningowych/testowych
    log("=== RAPORT Z ANALIZY DANYCH DIAMENTÓW (LISTA 2) ===\n")
    df = load_data()  # Wczytanie zbioru danych diamentów z seaborn
    df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
    # Kodowanie zmiennych kategorycznych na dummy variables (0/1), drop_first usuwa pierwszą kategorię aby uniknąć multicollinearity
    
    train_df, test_df = train_test_split(df_encoded.copy(), test_size=0.2, random_state=42)
    # Podział na zbiór treningowy (80%) i testowy (20%) z ustalonym ziarnem losowości
    
    # Przygotowanie danych dla klasyfikacji: cena > 5000 USD jako klasa 1 (drogie diamenty)
    y_train_clf = (train_df['price'] > 5000).astype(int)  # Zmienna docelowa binarna dla treningu
    y_test_clf = (test_df['price'] > 5000).astype(int)  # Zmienna docelowa binarna dla testu
    X_train_clf = train_df.drop(columns=['price'])  # Cechy treningowe (bez ceny)
    X_test_clf = test_df.drop(columns=['price'])  # Cechy testowe (bez ceny)

    # Przygotowanie danych dla regresji: przewidywanie dokładnej ceny
    y_train_reg = train_df['price'].values  # Cena jako tablica numpy dla treningu
    y_test_reg = test_df['price'].values  # Cena jako tablica numpy dla testu

    X_train_reg = train_df.drop(columns=['price']).values  # Cechy jako tablica numpy
    X_test_reg = test_df.drop(columns=['price']).values  # Cechy jako tablica numpy

    # 2. Drzewo decyzyjne - Klasyfikacja
    log("Drzewo decyzyjne - Klasyfikacja")
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # Drzewo o maksymalnej głębokości 5
    tree_clf.fit(X_train_clf, y_train_clf)  # Trenowanie modelu na danych treningowych
    
    y_pred_tree = tree_clf.predict(X_test_clf)  # Predykcje na zbiorze testowym
    acc = accuracy_score(y_test_clf, y_pred_tree)  # Obliczenie dokładności
    f1 = f1_score(y_test_clf, y_pred_tree)  # Obliczenie F1-score (ważne dla niezbalansowanych klas)
    
    log(f"Accuracy na zbiorze testowym: {acc:.4f}")
    log(f"F1-Score na zbiorze testowym: {f1:.4f}\n")
    
    plot_feature_importance(tree_clf.feature_importances_, X_train_clf.columns)

    # Interpretacja drzewa i ważność cech
    log(">>> Interpretacja drzewa i ważność cech")
    log("Fragment struktury reguł drzewa:")  # Wyświetlenie pierwszych poziomów drzewa
    tree_rules = export_text(tree_clf, feature_names=list(X_train_clf.columns), max_depth=2)
    log(tree_rules)
    
    # Ręczne obliczenie Information Gain dla demonstracji
    mask_left = X_train_clf['carat'] <= 1.0  # Podział na lewą część (carat <= 1.0)
    mask_right = X_train_clf['carat'] > 1.0  # Podział na prawą część (carat > 1.0)
    ig = information_gain(y_train_clf.values, y_train_clf[mask_left].values, y_train_clf[mask_right].values)

    # Obliczenie zysku informacyjnego dla podziału na carat przy progu 1.0
    log(f"Ręczny Information Gain dla 'carat' przy progu 1.0: {ig:.4f}")  # Wyświetlenie IG
    log(f"Feature Importance ze scikit-learn dla 'carat': {tree_clf.feature_importances_[0]:.4f}\n")

    # 3. Własna implementacja regresji liniowej
    log("Własna implementacja regresji liniowej")
    scaler = StandardScaler()  
    X_tr_s = scaler.fit_transform(X_train_reg)  # Dopasowanie skalera i transformacja danych treningowych
    X_te_s = scaler.transform(X_test_reg)  # Transformacja danych testowych (bez ponownego dopasowania)

    # Trenowanie trzech modeli regresji liniowej
    my_reg = CustomLinearRegression().fit_analytical(X_tr_s, y_train_reg)  # Własna implementacja analityczna
    my_reg_gd = CustomLinearRegression().fit_gradient_descent(X_tr_s, y_train_reg, learning_rate=0.1, epochs=1000)
    # Własna implementacja z spadkiem gradientu
    sk_reg = LinearRegression().fit(X_tr_s, y_train_reg) 
    
    # Obliczenie błędów MSE dla porównania modeli
    mse_my = mean_squared_error(y_test_reg, my_reg.predict(X_te_s))  # Błąd własnej implementacji
    mse_sk = mean_squared_error(y_test_reg, sk_reg.predict(X_te_s))  # Błąd scikit-learn
    
    log("Porównanie wyuczonych wag (intercept + waga 'carat'):")
    log(f"Analitycznie (NumPy):     W0={my_reg.weights[0]:.2f}, W1(carat)={my_reg.weights[1]:.2f}")
    log(f"Spadek Gradientu (NumPy): W0={my_reg_gd.weights[0]:.2f}, W1(carat)={my_reg_gd.weights[1]:.2f}")
    log(f"Scikit-Learn (Gotowiec):  W0={sk_reg.intercept_:.2f}, W1(carat)={sk_reg.coef_[0]:.2f}\n")
    
    log(f"Błąd MSE (Moja Implementacja Analityczna): {mse_my:,.2f}")
    log(f"Błąd MSE (Scikit-Learn LinearRegression):  {mse_sk:,.2f}\n")
    
    plot_regression_results(y_test_reg, my_reg.predict(X_te_s), "Moja Implementacja")  # Wykres wyników

    # 4. Interpretacja wag regresji przed i po skalowaniu
    log("Interpretacja regresji przed i po skalowaniu")
    reg_unscaled = LinearRegression().fit(X_train_reg, y_train_reg)  # Model na danych nieskalowanych
    
    log(f"Waga dla cechy 'carat' BEZ SKALOWANIA: {reg_unscaled.coef_[0]:.2f} USD")
    log(f"Waga dla cechy 'carat' PO SKALOWANIU:  {sk_reg.coef_[0]:.2f} USD")
    log("(Skalowanie pozwala na sprawiedliwe porównywanie wag między różnymi cechami!)\n")
    
    plot_weight_comparison(reg_unscaled.coef_, sk_reg.coef_, X_train_clf.columns)

    # 5. Bias-Variance oraz problem Czarnego Łabędzia
    log("Analiza Bias-Variance Tradeoff oraz problem Czarnego Łabędzia")
    log("Generowanie modeli od stopnia 1 do 12 (Trwa obliczanie...)")
    
    X_train_1d = train_df[['carat']].values  # Tylko cecha 'carat' dla regresji wielomianowej
    X_test_1d = test_df[['carat']].values

    train_errors = []
    test_errors = []
    degrees = range(1, 13)

    # pętla po stopniach wielomianu, trenowanie modelu i obliczanie błędów dla każdego stopnia
    for d in degrees:
        poly = PolynomialFeatures(degree=d)
        X_poly_tr = poly.fit_transform(X_train_1d)
        X_poly_te = poly.transform(X_test_1d)
        
        m_poly = LinearRegression().fit(X_poly_tr, y_train_reg)  # Trenowanie modelu wielomianowego
        train_errors.append(mean_squared_error(y_train_reg, m_poly.predict(X_poly_tr)))  # Błąd treningowy
        test_errors.append(mean_squared_error(y_test_reg, m_poly.predict(X_poly_te)))  # Błąd testowy

    plot_bias_variance_curve(degrees, train_errors, test_errors)
    log(">> Wykres krzywych złożoności zapisany.")
    
    black_swan = np.array([[15.0]]) # Anomalia - 15 karatów
    poly_overfit = PolynomialFeatures(degree=11)
    m_overfit = LinearRegression().fit(poly_overfit.fit_transform(X_train_1d), y_train_reg)
    m_normal = LinearRegression().fit(X_train_1d, y_train_reg)
    
    pred_normal = m_normal.predict(black_swan)[0]  # Predykcja modelu liniowego
    pred_swan = m_overfit.predict(poly_overfit.transform(black_swan))[0]  # Predykcja modelu przeuczonego
    
    log("\nEksperyment: Przewidywanie ceny diamentu o ogromnej masie 15 Karatów")
    log(f"Predykcja (Zwykły model Liniowy):       {pred_normal:,.2f} USD")
    log(f"Predykcja (Przeuczony Wielomian st. 11): {pred_swan:,.2f} USD")
    log("Wniosek: Przeuczony model traci zdolność generalizacji (Czarne Łabędzie!).\n")

    with open("WYNIKI_DO_RAPORTU.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    

if __name__ == "__main__":
    main()