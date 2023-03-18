from impls import *
from matplotlib import pyplot as plt
import numpy as np

def main():
    # load data
    X_train, y_train, X_test, y_test = load_proj_data()
    
    
    # 1
    # perform 6-fold cross validation on linear regressor for each degree d in [0,12]
    k = 6
    max_d = 12
    d_opt = None
    
    degree = list(range(max_d+1))
    cv_err = list()
    for d in degree:
        # create model
        model = OutputScalingWrapper(
            Pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=d),
                StandardScaler(),
                LinearRegressor()
            )
        )
        
        # score model
        cv_err.append(
            cross_val_score(model, X_train, y_train, k)
        )
    
    
    # output results of cross validation
    with open('../out/cv_errors_data.dat', 'w') as file:
        file.write('\n'.join(f'{d} {e}' for d, e in zip(degree, cv_err)))

    d_opt = plot_line(degree, cv_err, 'cv err', 'red')

    plt.xticks(degree)
    plt.xlabel('degree')
    plt.ylabel('RMSE')
    plt.title(f'Average RMSE of 6-fold CV for Linear Regression')
    plt.grid()
    plt.show()
    
    
    # 2
    # fit with d*
    model = OutputScalingWrapper(
        Pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=d_opt),
            StandardScaler(),
            LinearRegressor()
        )
    )
    model.fit(X_train, y_train)

    # 3
    # snag w*
    with open('../out/w.dat', 'w') as file:
        file.write('\n'.join(map(str, model.w.T[0])))
    
    # 4
    # training and test rmse
    print(f'LinearRegressor with degree {d_opt} polynomial features:')
    train_loss = model.score(X_train, y_train)
    test_loss = model.score(X_test, y_test)
    print('\tTraining RMSE:',train_loss)
    print('\tTesting RMSE:', test_loss)
    print('\tRelative Error:', 100*abs(test_loss-train_loss)/train_loss,'%')
    
    
    # 5
    # plot d* curve over data
    plt.scatter(X_train, y_train, color='green', label='Train data')
    plt.scatter(X_test, y_test, color='red', label='Test data', marker='x')
    years = np.array(list(range(1968, 2024)))
    years_inp = years.reshape(-1,1)
    plt.plot(years, model.predict(years_inp), label='Fit Polynomial Curve')

    plt.xlabel('Year')
    plt.ylabel('Working-Age Population')
    plt.title('Polynomial Curve-Fitting Regression for Working-Age Data')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    import os
    # move to project root
    os.chdir(os.path.dirname(__file__))
    
    main()    