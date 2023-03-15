import numpy as np
from functools import reduce
        

# █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀
# █░▀░█ █▄█ █▄▀ ██▄ █▄▄ ▄█

class LinearRegressor:
    def __init__(self):
        self.w = None

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        pseudo_inv = np.matmul(
            np.linalg.inv(
                np.matmul(X_train.T, X_train)
            ), 
            X_train.T
        )
        self.w = np.matmul(pseudo_inv, y_train)
    
    def predict(self, X_test) -> np.array:
        return np.matmul(X_test, self.w) 
    
    def score(self, X_test, y_test) -> float:
        """perform RMSE on estimator, requires that model has been fit
        """
        y_pred = self.predict(X_test)
        return RMSE(y_test, y_pred)
    
class RidgeRegressor(LinearRegressor):
    def __init__(self, λ: float) -> None:
        self.w = None
        self.λ = λ
        
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        pseudo_inv = np.matmul(
            np.linalg.inv(
                np.matmul(X_train.T, X_train) + self.λ*np.identity(n=X_train.shape[1])
            ), 
            X_train.T
        )
        self.w = np.matmul(pseudo_inv, y_train)

class Pipeline:
    def __init__(self, *steps) -> None:
        """Pipeline assumes last object is predictor and all others are transformers
        """
        self.transformers = steps[:-1]
        self.predictor = steps[-1]
        self.w = None
    
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        for transformer in self.transformers[:-1]:
            X_train = transformer.fit_transform(X_train)
        
        self.predictor.fit(X_train, y_train)
        self.w = self.predictor.w
    
    def predict(self, x: np.array) -> np.array:
        for transformer in self.transformers[:-1]:
            x = transformer.transform(x)
        
        return self.predictor.predict(x)

    def score(self, X_test: np.array, y_test: np.array) -> float:
        """perform RMSE on estimator, requires that model has been fit
        """
        y_pred = self.predict(X_test)
        return RMSE(y_test, y_pred)
    
    
# ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ █▀
# ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ ▄█

class StandardScaler:
    def __init__(self) -> None:
        pass
    
    def fit(self, x) -> None:
        # applies function f to each dimension d in data X
        apply_per_dim = lambda f, X: np.array([f(X[:, d]) for d in range(X.shape[1])])
        self.μ = apply_per_dim(np.mean, x)
        self.σ = apply_per_dim(np.std, x)
        self.σ[self.σ == 0] = 1 # avoid 0^-1
    
    def transform(self, x) -> np.array:
        return (x - self.μ) / self.σ
    
    # REVIEW: when do we ever use this
    def inverse_transform(self, x) -> np.array:
        return x * self.σ + self.μ
    
    def fit_transform(self, x) -> np.array:
        self.fit(x)
        return self.transform(x)

class PolynomialFeatures:
    def __init__(self, degree: int) -> None:
        self.degree = degree
        self.exponents = None
        
    def fit(self, X: np.array) -> None:
        m = X.shape[1]
        idcs = [[[0 for _ in range(m)]]]
        for _ in range(self.degree):
            curr = list()
            for prev in idcs[-1]:
                # find last nonzero exp
                last = max(np.max(np.nonzero([1]+prev))-1, 0)

                for i in range(last, m):
                    new = prev.copy()
                    new[i] += 1
                    curr.append(new)
            idcs.append(curr)
        # flatten array
        self.exponents = reduce(lambda a, b: a+b, idcs)
    
    def transform(self, X: np.array) -> np.array:
        # theres probably libraries that make what i do better 
        # but i cant find them :)
        # also i am personally olbigated to include at least 1 oneliner in every project i make
        return np.array([
            np.array([
                reduce(
                    lambda a,b: a*b, 
                    [var**exp for var, exp in zip(row, term)]
                ) 
                for term in self.exponents
            ]) 
            for row in X
        ])
    
    def fit_transform(self, x: np.array) -> np.array:
        self.fit(x)
        return self.transform(x)
    
    
# █▀ █▀▀ █▀█ █▀█ █ █▄░█ █▀▀
# ▄█ █▄▄ █▄█ █▀▄ █ █░▀█ █▄█

def RMSE(y_true: np.array, y_pred: np.array) -> float:
    # using this bc its equiv to RMSE formula
    # but i like the code for this one better
    n = len(y_true)
    rmse = np.linalg.norm(y_true - y_pred) * (n**-0.5)
    return rmse

def cross_val_score(model: LinearRegressor, x: np.array, y: np.array, k: int):
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
        
        model.fit(X_train, y_train)
        score = model.score(X_valid, y_valid)
        scores.append(score)
    
    return np.mean(scores)