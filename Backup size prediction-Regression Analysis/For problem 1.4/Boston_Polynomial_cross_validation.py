# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation           
from sklearn.cross_validation import cross_val_predict
import math
import numpy as np

#data geting
import csv
with open('housing_data.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:14] for row in reader]
    data = np.array(data_list)
    data = data.astype(np.float)

feature = data[:,0:13]
y =  data[:,13]
train_target = y
train_features = feature

rmse=[]
poly_degree=[1,2,3,4,5,6]
for i in poly_degree:
    
    '''Set polynominal feature'''
    polynomial_features = PolynomialFeatures(degree=i,include_bias=False)
    linear_regression = LinearRegression(fit_intercept = False)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])


    '''training and predict'''
    pipeline.fit(train_features, train_target)
    scores = cross_validation.cross_val_score(pipeline,train_features, 
                train_target, scoring="mean_squared_error", cv=10)
    predicted = cross_val_predict(pipeline,train_features, 
                              train_target,cv=10)

    mse_scores = -scores
    rmse.append(math.sqrt(mse_scores.mean()))

    #plot
    fig, ax = plt.subplots()
    ax.scatter(train_target, predicted)
    ax.plot([train_target.min(), train_target.max()], [train_target.min(), train_target.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    

#plot ploy_degree versus rmse
print "RMSE: ", rmse
fig2,bx=plt.subplots()
plt.plot(poly_degree,rmse)
bx.set_xlabel('Degree')
bx.set_ylabel('RMSE')
plt.show()
    
