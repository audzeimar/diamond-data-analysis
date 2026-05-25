from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  

# Folder, do którego zapisywane są wszystkie wygenerowane wykresy
PLOTS_DIR = Path('results/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_STYLE = {
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 140,
    'savefig.dpi': 220,
}

# Uporządkowana kolejność kategorii na wykresach
CUT_ORDER = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COLOR_ORDER = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
CLARITY_ORDER = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

def _setup_style():
    sns.set_theme(style='whitegrid', context='notebook') 
    plt.rcParams.update(BASE_STYLE) 

def _finalize_plot(path: Path):
    # Dopasowuje układ wykresu, zapisuje go do pliku i zamyka figurę
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()


# WYKRESY Z LISTY 1 (EDA) - podstawowe wykresy eksploracyjne

# Tworzy histogram rozkładu ceny diamentów
def plot_histogram(df):
    _setup_style()  # Ustawienie stylu wykresu
    plt.figure(figsize=(8.2, 4.8))  # Ustawienie rozmiaru figury
    ax = sns.histplot(df, x='price', bins=35, kde=True, color='#2A9D8F', edgecolor='white', linewidth=0.4)
    ax.set_title('Rozkład ceny diamentów')
    ax.set_xlabel('Cena [USD]')
    ax.set_ylabel('Liczba obserwacji')
    _finalize_plot(PLOTS_DIR / 'hist_price.png')

# Tworzy wykres punktowy zależności ceny od masy diamentu
def plot_scatter(df):
    _setup_style()
    plt.figure(figsize=(7.2, 5.2))
    ax = sns.scatterplot(data=df, x='carat', y='price', alpha=0.18, s=16, color='#3D5A80', edgecolor=None)
    ax.set_title('Zależność ceny od masy diamentu')
    ax.set_xlabel('Masa [carat]')
    ax.set_ylabel('Cena [USD]')
    _finalize_plot(PLOTS_DIR / 'scatter_carat_price.png')

# Tworzy boxplot rozkładu ceny diamentów (bez outliers)
def plot_boxplot(df):
    _setup_style()
    plt.figure(figsize=(4.8, 5.4))
    ax = sns.boxplot(data=df, y='price', color='#8AB17D', width=0.45, showfliers=False)
    ax.set_title('Rozkład ceny (boxplot)')
    ax.set_xlabel('')
    ax.set_ylabel('Cena [USD]')
    _finalize_plot(PLOTS_DIR / 'boxplot_price.png')

# Tworzy wykres słupkowy średniej ceny względem jakości szlifu
def plot_correlation(df):
    _setup_style()
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(8.6, 6.8))
    ax = sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='YlGnBu', square=True, linewidths=0.5, cbar_kws={'shrink': 0.9}, annot_kws={'size': 8})
    ax.set_title('Macierz korelacji cech numerycznych')
    _finalize_plot(PLOTS_DIR / 'correlation_matrix.png')

# Tworzy histogram rozkładu masy diamentów
def plot_carat_hist(df):
    _setup_style()
    plt.figure(figsize=(8.2, 4.8))
    ax = sns.histplot(df, x='carat', bins=32, kde=True, color='#E76F51', edgecolor='white', linewidth=0.35)
    ax.set_title('Rozkład masy diamentów')
    ax.set_xlabel('Masa [carat]')
    ax.set_ylabel('Liczba obserwacji')
    _finalize_plot(PLOTS_DIR / 'hist_carat.png')

# Tworzy boxploty ceny względem kategorii kolor i przejrzystość
def plot_box_categorical(df):
    _setup_style()
    plt.figure(figsize=(8.6, 5.2))
    ax = sns.boxplot(data=df, x='color', y='price', order=COLOR_ORDER, hue='color', palette=sns.color_palette('crest', n_colors=len(COLOR_ORDER)), showfliers=False, width=0.65, dodge=False, legend=False)
    ax.set_title('Cena a kolor diamentu')
    ax.set_xlabel('Kolor')
    ax.set_ylabel('Cena [USD]')
    _finalize_plot(PLOTS_DIR / 'boxplot_color.png')

    plt.figure(figsize=(8.8, 5.2))
    ax = sns.boxplot(data=df, x='clarity', y='price', order=CLARITY_ORDER, hue='clarity', palette=sns.color_palette('mako', n_colors=len(CLARITY_ORDER)), showfliers=False, width=0.65, dodge=False, legend=False)
    ax.set_title('Cena a przejrzystość diamentu')
    ax.set_xlabel('Przejrzystość')
    ax.set_ylabel('Cena [USD]')
    _finalize_plot(PLOTS_DIR / 'boxplot_clarity.png')

# Tworzy wykres słupkowy średniej ceny względem jakości szlifu
def plot_bar_cut(df):
    _setup_style()
    plt.figure(figsize=(7.6, 5.0))
    ax = sns.barplot(data=df, x='cut', y='price', order=CUT_ORDER, hue='cut', estimator='mean', errorbar=('ci', 95), palette=sns.color_palette('viridis', n_colors=len(CUT_ORDER)), dodge=False, legend=False)
    ax.set_title('Średnia cena a jakość szlifu')
    ax.set_xlabel('Jakość szlifu')
    ax.set_ylabel('Średnia cena [USD]')
    _finalize_plot(PLOTS_DIR / 'bar_cut_price.png')


# NOWE WYKRESY DO RAPORTU (LISTA 2)

# Zapisuje wykres słupkowy ważności cech dla drzewa decyzyjnego
def plot_feature_importance(importances, columns):
    _setup_style()
    plt.figure(figsize=(10, 6))
    feat_imp = pd.Series(importances, index=columns).sort_values(ascending=False).head(10)
    
    # Wybór top 10 najważniejszych cech, posortowanych malejąco
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Top 10 najważniejszych cech (Decision Tree)')
    plt.xlabel('Ważność (Feature Importance)')
    _finalize_plot(PLOTS_DIR / 'feature_importance.png')

# Porównuje wagi cech dla regresji liniowej przed i po skalowaniu
def plot_weight_comparison(weights_unscaled, weights_scaled, columns):
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Wybieramy tylko pierwsze 5 cech (w tym carat), żeby wykres był czytelny
    cols = columns[:5] 
    
    ax1.bar(cols, weights_unscaled[:5], color='skyblue')
    ax1.set_title('Wagi BEZ skalowania')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(cols, weights_scaled[:5], color='salmon')
    ax2.set_title('Wagi PO skalowaniu (StandardScaler)')
    ax2.tick_params(axis='x', rotation=45)
    
    _finalize_plot(PLOTS_DIR / 'weights_comparison.png')

# Tworzy wykres porównujący rzeczywiste i przewidziane ceny dla regresji liniowej
def plot_regression_results(y_true, y_pred, title):
    _setup_style()
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='purple', s=10)
    
    # Czerwona linia idealnej predykcji (y=x)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    
    plt.title(f'Regresja: {title}\n(Rzeczywiste vs Przewidziane)')
    plt.xlabel('Cena rzeczywista [USD]')
    plt.ylabel('Cena przewidziana [USD]')
    # Dodajemy zabezpieczenie na nazwę pliku - usuwamy znaki specjalne
    safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _finalize_plot(PLOTS_DIR / f'regression_results_{safe_title}.png')

# Wykres krzywych bias-variance tradeoff dla regresji wielomianowej
def plot_bias_variance_curve(degrees, train_errors, test_errors):
    _setup_style()
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', label="Błąd treningowy (MSE)", color='blue')

    # Krzywa błędu treningowego z markerami kółek
    plt.plot(degrees, test_errors, 'x-', label="Błąd testowy (MSE)", color='red')

    # Krzywa błędu testowego z markerami krzyżyków
    plt.yscale('log')  # Skala logarytmiczna na osi Y, aby lepiej pokazać duże różnice błędów
    
    plt.xlabel("Stopień wielomianu (Skomplikowanie modelu)")
    plt.ylabel("Błąd MSE (skala logarytmiczna)")
    plt.title("Analiza Bias-Variance Tradeoff")
    plt.legend()
    _finalize_plot(PLOTS_DIR / 'bias_variance_curve.png')

# --------------------- Lista 3 --------------------
def plot_regularization_weights(w_unreg, w_l1, w_l2, features):
    # Pokazuje wyzerowanie wag przez Lasso i zmniejszenie przez Ridge
    _setup_style()
    x = np.arange(len(features))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, w_unreg, width, label='Brak regularyzacji', color='gray', alpha=0.6)
    ax.bar(x, w_l1, width, label='Lasso (L1) - Zeruje wagi', color='#E76F51')
    ax.bar(x + width, w_l2, width, label='Ridge (L2) - Zmniejsza wagi', color='#2A9D8F')

    ax.set_ylabel('Wartość Wagi')
    ax.set_title('Wpływ Regularyzacji na Wagi Modelu Liniowego')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'regularization_weights.png')
    plt.close()

def plot_tree_regularization(param_values, train_scores, test_scores, param_name):
    # Wykres obrazujący sweet-spot regularyzacji strukturalnej drzewa
    _setup_style()
    plt.figure(figsize=(9, 5))
    plt.plot(param_values, train_scores, label='Zbiór treningowy', marker='o', color='blue')
    plt.plot(param_values, test_scores, label='Zbiór testowy', marker='x', color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Dokładność (Accuracy)')
    plt.title(f'Optymalizacja parametru: {param_name} (Sweet Spot)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'tree_regularization_{param_name}.png')
    plt.close()
