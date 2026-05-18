# System wspomagania decyzji w wycenie diamentów

## Opis projektu
Projekt polega na analizie zbioru danych dotyczącego diamentów oraz stworzeniu prostego systemu wspomagania decyzji opartego na regułach.

Model wykorzystuje zarówno cechy numeryczne (masa, cena), jak i kategorialne (kolor, jakość szlifu, przejrzystość).

**Projekt został podzielony na dwie części:**
* **Część 1:** Analiza eksploracyjna danych (EDA) oraz budowa manualnego systemu regułowego (if/else) do klasyfikacji i prostej regresji.
* **Część 2 (Machine Learning):** Automatyzacja systemu z wykorzystaniem klasycznych algorytmów uczenia maszynowego. Obejmuje m.in.:
  - **Drzewa Decyzyjne** (z weryfikacją zysku informacyjnego / Information Gain).
  - **Regresję Liniową** zaimplementowaną od zera za pomocą biblioteki NumPy (Rozwiązanie analityczne MNK oraz iteracyjne Spadek Gradientu) i porównaną z modelem `scikit-learn`.
  - Badanie wpływu standaryzacji (`StandardScaler`) na interpretację wag modelu.
  - Analizę zjawiska **Bias-Variance Tradeoff** (kompromisu obciążenie-wariancja) oraz testy błędu ekstrapolacji (**Problem "Czarnego Łabędzia"**) przy pomocy wielomianów wysokiego stopnia.

## Struktura projektu

metody_s_i_d/
│
├── src/
│   ├── data_loader.py       # Pobieranie i przygotowanie zbioru danych
│   ├── eda.py               # Generowanie wykresów (EDA, ważne cechy, Bias-Variance)
│   ├── evaluation.py        # Funkcje metryk (MSE, Accuracy, F1-Score)
│   └── models_scratch.py    # Autorskie implementacje algorytmów (NumPy)
│
├── results/
│   └── plots/               # Tu automatycznie zapisują się wygenerowane wykresy
│
├── main_lista2.py           # Główny skrypt dla Części 2 (Machine Learning)
├── main.py                  # Skrypt archiwalny z Części 1 (Modele regułowe)
├── WYNIKI_DO_RAPORTU.txt    # Automatycznie generowany plik tekstowy z wynikami eksperymentów
├── requirements.txt         # Lista wymaganych bibliotek
└── README.md                # Dokumentacja projektu

## Instalacja
Zainstaluj wymagane biblioteki:
pip install -r requirements.txt

## Uruchomienie
Aby uruchomić projekt, wpisz:
python main.py / main_lista2.py

## Autor
Maryia Audzei