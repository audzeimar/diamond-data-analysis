import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import load_data
from src.eda import plot_regularization_weights, plot_tree_regularization
from src.models_scratch import (
    CustomLinearRegression, CustomBaggingClassifier, 
    CustomGradientBoostingRegressor, MixtureOfExperts
)

def main():
    print("="*60)
    print(" URUCHAMIANIE PROJEKTU: CZĘŚĆ 3 (Ensemble & Regularization) ")
    print("="*60 + "\n")

    # 1. PREPROCESSING
    df = load_data()
    df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
    train_df, test_df = train_test_split(df_encoded.copy(), test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    
    # Target 1: Klasyfikacja (Cena > 5000)
    y_train_clf = (train_df['price'] > 5000).astype(int).values
    y_test_clf = (test_df['price'] > 5000).astype(int).values
    X_train_clf = scaler.fit_transform(train_df.drop(columns=['price']))
    X_test_clf = scaler.transform(test_df.drop(columns=['price']))

    # Target 2: Regresja (Dokładna cena)
    y_train_reg = train_df['price'].values
    y_test_reg = test_df['price'].values
    X_train_reg = scaler.fit_transform(train_df.drop(columns=['price']))
    X_test_reg = scaler.transform(test_df.drop(columns=['price']))

    features = train_df.drop(columns=['price']).columns

    print(">>> REGULARYZACJA")
    
    # Regresja (Moja Implementacja z models_scratch.py)
    unreg = CustomLinearRegression().fit_gradient_descent(X_train_reg, y_train_reg, learning_rate=0.1, epochs=500)
    l1_model = CustomLinearRegression().fit_gradient_descent(X_train_reg, y_train_reg, penalty='l1', alpha=15000.0, learning_rate=0.1, epochs=500)
    l2_model = CustomLinearRegression().fit_gradient_descent(X_train_reg, y_train_reg, penalty='l2', alpha=1000.0, learning_rate=0.1, epochs=500)
    
    plot_regularization_weights(unreg.weights[1:10], l1_model.weights[1:10], l2_model.weights[1:10], features[:9])
    print("- Wykres wag L1/L2 wygenerowany w results/plots/")

    # Regularyzacja drzewa (min_samples_leaf)
    leaves = [1, 5, 10, 50, 100, 500]
    train_acc, test_acc = [], []
    for l in leaves:
        dt = DecisionTreeClassifier(min_samples_leaf=l, random_state=42).fit(X_train_clf, y_train_clf)
        train_acc.append(accuracy_score(y_train_clf, dt.predict(X_train_clf)))
        test_acc.append(accuracy_score(y_test_clf, dt.predict(X_test_clf)))
    
    plot_tree_regularization(leaves, train_acc, test_acc, "min_samples_leaf")
    print("- Wykres Sweet-Spot dla drzewa wygenerowany.\n")

    
    dt_base = DecisionTreeClassifier(max_depth=5).fit(X_train_clf, y_train_clf)
    my_bagging = CustomBaggingClassifier(n_estimators=10)
    my_bagging.fit(X_train_clf, y_train_clf)
    rf_sklearn = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42).fit(X_train_clf, y_train_clf)
    
    print(f"Pojedyncze drzewo (Acc):  {accuracy_score(y_test_clf, dt_base.predict(X_test_clf)):.4f}")
    print(f"Mój Bagging (10 drzew):   {accuracy_score(y_test_clf, my_bagging.predict(X_test_clf)):.4f}")
    print(f"Scikit-Learn RandomForest: {accuracy_score(y_test_clf, rf_sklearn.predict(X_test_clf)):.4f}\n")


    print(">>> STACKING (Klasyfikacja)")
    
    m1 = DecisionTreeClassifier(max_depth=5)
    m2 = LogisticRegression(max_iter=1000)
    m3 = KNeighborsClassifier(n_neighbors=5)
    
    # Używamy cross_val_predict aby zapobiec wyciekowi danych (Data Leakage)
    print("Trenowanie meta-modelu na predykcjach CV...")
    pred_m1 = cross_val_predict(m1, X_train_clf, y_train_clf, cv=3)
    pred_m2 = cross_val_predict(m2, X_train_clf, y_train_clf, cv=3)
    pred_m3 = cross_val_predict(m3, X_train_clf, y_train_clf, cv=3)
    
    X_meta_train = np.column_stack((pred_m1, pred_m2, pred_m3))
    meta_model = LogisticRegression().fit(X_meta_train, y_train_clf)
    
    # Test
    m1.fit(X_train_clf, y_train_clf)
    m2.fit(X_train_clf, y_train_clf)
    m3.fit(X_train_clf, y_train_clf)
    X_meta_test = np.column_stack((m1.predict(X_test_clf), m2.predict(X_test_clf), m3.predict(X_test_clf)))
    
    print(f"Wagi zaufania meta-modelu (Drzewo, LogReg, KNN): {np.round(meta_model.coef_[0], 2)}")
    print(f"Dokładność (Stacking): {accuracy_score(y_test_clf, meta_model.predict(X_meta_test)):.4f}\n")


    print(">>> GRADIENT BOOSTING (Regresja)")
    
    my_gbm = CustomGradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    my_gbm.fit(X_train_reg, y_train_reg)
    lin_reg = LinearRegression().fit(X_train_reg, y_train_reg)
    
    print(f"MSE (Zwykła Regresja Liniowa): {mean_squared_error(y_test_reg, lin_reg.predict(X_test_reg)):,.2f}")
    print(f"MSE (Mój Gradient Boosting):   {mean_squared_error(y_test_reg, my_gbm.predict(X_test_reg)):,.2f}\n")


    print(">>> MIXTURE OF EXPERTS (Regresja)")
    
    moe = MixtureOfExperts(n_experts=3)
    moe.fit(X_train_reg, y_train_reg)
    
    print(f"MSE (MoE - Sieć z 3 Ekspertami): {mean_squared_error(y_test_reg, moe.predict(X_test_reg)):,.2f}")
    
    print("\n" + "="*60)
    print(" ZAKOŃCZONO POMYŚLNIE - ZAJRZYJ DO FOLDERU results/plots/")
    print("="*60)

if __name__ == "__main__":
    main()
