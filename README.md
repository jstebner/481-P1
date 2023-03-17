# 481-P1
Project 1 Report: Polynomial Curve-Fitting Regression for Working-Age Data

Full writeup is in docs/project1_report.pdf[# TODO make this link]

## Installation
All dependencies can be installed by running:
```
python3 -m pip install -r requirements.txt
```


## Overview
The aim of this project is to implement all components necessary to perform generalized linear regression on the data found in the ‘data/’ directory.

The implementations for all models and transformers can be found in the ‘src/impls.py’ file, and the usage of implementations is done in the ‘src/main.py’ file which can be run with:
```
$ python3 ./main.py
```
Note that the main file can be run from any location as it can locate the positions of supporting files.

All figures used in this document and data collected from testing the model are also stored in the ‘out/’ directory with names indicative of their content.

The model to be tested (in ‘src/main.py’) is given as follows:
```
model = OutputScalingWrapper(
    Pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=d),
        StandardScaler(),
        LinearRegressor()
    )
)
```
A rough outline of how the model seeks to perform polynomial curve fitting is given as follows:
1.	The target vector is scaled prior to fitting the model (for prediction accuracy)
2.	The input vector is scaled and transformed with Polynomial Features
3.	The transformed input vector is scaled to create interpretable weights
4.	A linear regressor is fit using the transformed input and target vectors
5.	When creating predictions, the model applies the weights found when predicting to the new transformed input (using the same transformations as the training data) and un-scales the resulting vector (using the same transformations as the training target vector).