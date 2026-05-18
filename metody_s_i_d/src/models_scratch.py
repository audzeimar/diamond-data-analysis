import numpy as np

# Obliczanie entropii i zysku informacyjnego dla drzewa decyzyjnego
def calculate_entropy(y):
    counts = np.bincount(y)  # Zlicza wystąpienia każdej klasy
    probs = counts / len(y)  # Oblicza prawdopodobieństwa klas 
    probs = probs[probs > 0]  # Usuwa zera przy klasach nieobecnych
    return -np.sum(probs * np.log2(probs))  # Wzór na entropię

# Zysk informacyjny dla podziału drzewa decyzyjnego
def information_gain(y, y_left, y_right):

    # proporcje próbek po lewej i prawej stronie podziału
    p_left = len(y_left) / len(y)  
    p_right = len(y_right) / len(y)  
    
    # entropia przed podziałem i po podziale
    entropy_parent = calculate_entropy(y)  
    entropy_children = p_left * calculate_entropy(y_left) + p_right * calculate_entropy(y_right)  
    
    return entropy_parent - entropy_children  # Zysk informacyjny

# Własna implementacja regresji liniowej
class CustomLinearRegression:
    def __init__(self):
        self.weights = None 

    # Dopasowanie modelu metodą analityczną (wzór normalny) 
    def fit_analytical(self, X, y):

        # Dodanie kolumny jedynek dla wyrazu wolnego 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Kształt: (n_samples, n_features + 1)

        # Wzór normalny dla metody najmniejszych kwadratów
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        return self 

    # Dopasowanie modelu metodą spadku gradientu    
    def fit_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Dodanie intercept
        m = len(y)  # Liczba próbek treningowych
        self.weights = np.zeros(X_b.shape[1])  # Inicjalizacja wag na zero
        
        # Pętla treningowa spadku gradientu
        for epoch in range(epochs):
            predictions = X_b.dot(self.weights)  # Obliczenie predykcji dla wszystkich próbek
            errors = predictions - y 
            gradients = (2/m) * X_b.T.dot(errors) 
            self.weights -= learning_rate * gradients  # Aktualizacja wag w kierunku przeciwnym do gradientu
        return self

    # Predykcja wartości na podstawie wyuczonych wag    
    def predict(self, X):
       
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Dodanie intercept
        return X_b.dot(self.weights)  # Obliczenie predykcji liniowej