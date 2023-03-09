# Import Libraries
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

D_X, D_y = np.loadtxt("data/train.dat", usecols=(0,1), unpack=True)

X = D_X.reshape(-1,1)
y = D_y.reshape(-1,1)

d = 0
lambda_val = 0

poly = PolynomialFeatures(degree=d)
X_trans = poly.fit_transform(X)
# Create generalized linear regression object
regr = Ridge(alpha=lambda_val,fit_intercept=False,solver='cholesky')
# Train the model using the training sets
regr.fit(X_trans, y)

# Saving the weights in a text file
weights_file = open(f"out/w.dat","w")

for i in range(d+1):
    weights_file.write(str(regr.coef_[0][i]))
    weights_file.write("\n")

weights_file.close()

