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
        

class LinearRegressor:
    def __init__(self):
        self.w = None

    # TODO: this
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        pseudo_inv = np.cross(
            np.linalg.inv(
                np.cross(X_train.T, X_train)
            ), 
            X_train.T
        )
        
    
    # TODO: this
    def predict(self, X_test) -> np.array:
        pass
    
    def score(self, X_test, y_test) -> float:
        mse = RMSE(y_test, self.predict(X_test))**2
        return mse
    
class RidgeRegressor(LinearRegressor):
    def __init__(self, λ: float) -> None:
        self.w = None
        self.λ = λ
        
    def score(self, X_test, y_test) -> float:
        n = len(X_test)
        penalty = self.λ/2 * np.linalg.norm(self.w)**2
        L2_reg_sqr_err = (n/2) * RMSE(y_test, self.predict(X_test))**2 + penalty

        return L2_reg_sqr_err

class StandardScaler:
    def __init__(self) -> None:
        pass
    
    def fit(self, x) -> None:
        self.μ = np.mean(x)
        self.σ = np.std(x)
    
    # DEBUG: this dont work right for matrices
    def transform(self, x) -> np.array:
        return (x - self.μ) / self.σ
    
    # REVIEW: when do we ever use this bro
    def untransform(self, x) -> np.array:
        return x * self.σ + self.μ
    
    def fit_transform(self, x) -> np.array:
        self.fit(x)
        return self.transform(x)

# TODO: this
# REVIEW: maybe dont do this
class PolynomialFeatures:
    def __init__(self, degree: int) -> None:
        self.degree = degree
        
    def fit(self, x: np.array) -> None:
        pass
    
    def transform(self, x: np.array) -> np.array:
        pass

# REVIEW: this
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