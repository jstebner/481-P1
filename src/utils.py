import numpy as np

def RMSE(hyp_f, data: np.array) -> float:
    # using this bc its equiv to classic RMSE formula
    # but i like the code for this one better
    n = len(data)
    y_test = data[:, 1]
    y_pred = hyp_f(data[:, 0])
    rmse = np.linalg.norm(y_test - y_pred) * n**-0.5
    return round(rmse, 5)

# NOTE: for poly reg: scale x in each dim, transform each dim with poly term, 
#       scale again, then fit (so scaler on transform matrix) to make weights 
#       comparable / interpretable

# following sklearn pattern
class LinearRegressor:
    # TODO: this
    def __init__(self,):
        # init params here or smthn
        pass

    # TODO: this
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """_summary_

        Args:
            X_train (np.array): _description_
            y_train (np.array): _description_
        """
        pass
    
    # TODO: this
    # REVIEW: if this is needed
    def transform(self, X_train) -> np.array:
        """_summary_

        Args:
            X_train (np.array): _description_
            y_train (np.array): _description_
        """
        pass
    
    # REVIEW: if this is needed
    def fit_transform(self, X_train, y_train) -> np.array:
        """_summary_

        Args:
            X_train (np.array): _description_
            y_train (np.array): _description_
        """
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    # TODO: this
    def predict(self, X_test) -> np.array:
        """_summary_

        Args:
            X_train (np.array): _description_
            y_train (np.array): _description_
        """
        pass
    
    # TODO: this
    # REVIEW: if this is the right place to put it
    def score(self, X_test, y_test) -> float:
        """_summary_

        Args:
            X_train (np.array): _description_
            y_train (np.array): _description_
        """
        pass
    
