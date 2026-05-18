# Diamond Valuation Decision Support System

## Project Overview
This project focuses on the exploratory data analysis (EDA) of a diamond dataset and the development of a decision support system for diamond valuation. The model utilizes both numerical features (carat weight, price) and categorical features (color, cut quality, clarity) to estimate values and make data-driven classifications.

**The project is structured into two main phases:**
* **Part 1: Exploratory Data Analysis & Rule-Based Modeling** Initial data exploration and the construction of a manual, rule-based system (if/else logic) for basic classification and simple regression tasks.
* **Part 2: Machine Learning & Statistical Modeling** Automation of the decision-making process using classic machine learning algorithms. This phase includes:
  * **Decision Trees:** Implementation with a focus on Information Gain verification.
  * **Linear Regression:** Implemented entirely from scratch using **NumPy** (covering both the analytical Ordinary Least Squares solution and iterative Gradient Descent) and benchmarked against the `scikit-learn` library.
  * **Feature Scaling Analysis:** Evaluating the impact of standardization (`StandardScaler`) on the interpretability of model weights.
  * **Bias-Variance Tradeoff:** In-depth statistical analysis of model complexity using high-degree polynomials, including testing for extrapolation errors (the "Black Swan" problem).

## Project Structure

```text
├── src/
│   ├── data_loader.py       # Pobieranie i przygotowanie zbioru danych
│   ├── eda.py                # Generowanie wykresów (EDA, ważne cechy, Bias-Variance)
│   ├── evaluation.py         # Funkcje metryk (MSE, Accuracy, F1-Score)
│   └── models_scratch.py     # Autorskie implementacje algorytmów (NumPy)
├── results/
│   └── plots/                # Tu automatycznie zapisują się wygenerowane wykresy
├── main_lista2.py            # Główny skrypt dla Części 2 (Machine Learning)
├── main.py                   # Skrypt archiwalny z Części 1 (Modele regułowe)
├── WYNIKI_DO_RAPORTU.txt     # Automatycznie generowany plik tekstowy z wynikami eksperymentów
├── requirements.txt          # Lista wymaganych bibliotek
└── README.md                 # Dokumentacja projektu
```

## Installation
Clone the repository and install the required dependencies using pip: pip install -r requirements.txt

## Usage
To run the specific parts of the project, execute the corresponding main scripts from your terminal:

# To run the Machine Learning and Statistical Analysis phase (Part 2):
python main_lista2.py

# To run the initial Rule-Based System phase (Part 1):
python main.py

## Autor
Maryia Audzei
