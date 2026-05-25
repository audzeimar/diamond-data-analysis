import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

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

    # Dopasowanie modelu metodą spadku gradientu Z REGULARYZACJĄ L1/L2
    def fit_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000, penalty=None, alpha=1.0):
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Dodanie intercept
        m = len(y)  # Liczba próbek treningowych
        self.weights = np.zeros(X_b.shape[1])  # Inicjalizacja wag na zero
        
        # Pętla treningowa spadku gradientu
        for epoch in range(epochs):
            predictions = X_b.dot(self.weights)  # Obliczenie predykcji dla wszystkich próbek
            errors = predictions - y 
            gradients = (2/m) * X_b.T.dot(errors) 
            
            # --- LOGIKA REGULARYZACJI Z LISTY 3 ---
            # Dodajemy karę do gradientów (pomijając wyraz wolny: self.weights[0])
            if penalty == 'l2':
                gradients[1:] += (2 * alpha / m) * self.weights[1:]
            elif penalty == 'l1':
                gradients[1:] += (alpha / m) * np.sign(self.weights[1:])
                
            self.weights -= learning_rate * gradients  # Aktualizacja wag
        return self

    # Predykcja wartości na podstawie wyuczonych wag    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Dodanie intercept
        return X_b.dot(self.weights)  # Obliczenie predykcji liniowej
    
# ----------------- Lista 3 -----------------
class CustomBaggingClassifier:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        self.models = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap: losowanie ze zwracaniem
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Słaby klasyfikator
            tree = DecisionTreeClassifier(max_depth=5, random_state=np.random.randint(1000))
            tree.fit(X_sample, y_sample)
            self.models.append(tree)

    def predict(self, X):
        # Zebranie głosów od wszystkich drzew
        predictions = np.array([model.predict(X) for model in self.models])
        # Twarde głosowanie (najczęstsza klasa w każdej kolumnie - próbce)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.base_pred = 0

    def fit(self, X, y):
        # 1. Start od średniej wartości
        self.base_pred = np.mean(y)
        F_m = np.full(y.shape, self.base_pred)

        # 2. Uczenie na resztach
        for _ in range(self.n_estimators):
            residuals = y - F_m
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self.models.append(tree)
            
            F_m += self.learning_rate * tree.predict(X)

    def predict(self, X):
        F_m = np.full(X.shape[0], self.base_pred)
        for tree in self.models:
            F_m += self.learning_rate * tree.predict(X)
        return F_m
    
class MixtureOfExperts:
    def __init__(self, n_experts=3):
        self.n_experts = n_experts
        # Bramka (Router)
        self.cluster = KMeans(n_clusters=n_experts, random_state=42)
        self.gate = RandomForestClassifier(random_state=42)
        # Eksperci (modele Liniowe)
        self.experts = [LinearRegression() for _ in range(n_experts)]

    def fit(self, X, y):
        # Bramka dzieli dane i uczy się przypisywania
        clusters = self.cluster.fit_predict(X)
        self.gate.fit(X, clusters)

        # Trening ekspertów wąskodziedzinowych
        for i in range(self.n_experts):
            X_k, y_k = X[clusters == i], y[clusters == i]
            if len(X_k) > 0:
                self.experts[i].fit(X_k, y_k)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        routes = self.gate.predict(X)
        
        # Odesłanie przypadku do konkretnego eksperta
        for i in range(self.n_experts):
            mask = routes == i
            if np.any(mask):
                predictions[mask] = self.experts[i].predict(X[mask])
        return predictions
