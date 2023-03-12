import numpy as np

def RMSE(y_test: np.array, y_pred: np.array) -> float:
    # using this bc its equiv to classic RMSE formula
    # but i like the code for this one better
    n = len(y_test)
    rmse = np.linalg.norm(y_test - y_pred) * n**-0.5
    return round(rmse, 5)

# REVIEW: this?
def cross_val_score(estimator, x: np.array, y: np.array, k: int):
    # im sure theres a better way to do this with numpy 
    # but i dont wanna find out
    scores = list()
    n = len(x)
    for i in range(k):
        l, r = i*n//k, (i+1)*n//k
        X_train = np.array(list(x[0:l])+list(x[r:]))
        y_train = np.array(list(y[0:l])+list(y[r:]))
        X_valid = x[l:r]
        y_valid = y[l:r]
        
        estimator.fit(X_train, y_train)
        score = estimator.score(X_valid, y_valid)
        scores.append(score)
    
    return np.mean(scores)
        

class RidgeRegressor:
    def __init__(self, λ: float):
        self.λ = λ
        self.w = None

    # TODO: this
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        # self.w = (X_train.T * X_train.)
        pass
    
    # TODO: this
    def predict(self, X_test) -> np.array:
        pass
    
    # REVIEW: if this is even right lol
    def score(self, X_test, y_test) -> float:
        # gives MSE
        return RMSE(y_test, self.predict(X_test))**2
    
class StandardScaler:
    def __init__(self) -> None:
        pass
    
    def fit(self, x) -> None:
        self.μ = np.mean(x)
        self.σ = np.std(x)
    
    def transform(self, x) -> np.array:
        return (x - self.μ) / self.σ
    
    # REVIEW: when do we ever use this bro
    def untransform(self, x) -> np.array:
        return x * self.σ + self.μ
    
    def fit_transform(self, x) -> np.array:
        self.fit(x)
        return self.transform(x)

# TODO: this
# REVIEW: maybe just dont make this
class PolynomialFeatures:
    def __init__(self, d: int) -> None:
        self.degree = d
        
    def fit(self, x: np.array) -> None:
        pass
    
    def transform(self, x: np.array) -> np.array:
        pass
    
class Pipeline:
    def __init__(self, *steps) -> None:
        """Pipeline assumes last object is predictor and all others are transformers
        """
        self.transformers = steps[:-1]
        self.predictor = steps[-1]
        self.w = self.predictor.w
    
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        for transformer in self.transformers[:-1]:
            X_train = transformer.fit_transform(X_train)
        
        self.predictor.fit(X_train, y_train)
    
    def predict(self, x: np.array) -> np.array:
        for transformer in self.transformers[:-1]:
            x = transformer.transform(x)
        
        return self.predictor.predict(x)