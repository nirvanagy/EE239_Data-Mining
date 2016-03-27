# -*- coding: utf-8 -*-
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
import numpy as np
# loading data
import csv
with open('housing_data.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:14] for row in reader]
    data = np.array(data_list)
    data = data.astype(np.float)

feature = data[:,0:13]

y =  data[:,13]


for i in [1,0.1,0.01,0.001]:
    
# choose Ridge or Linear or Lasso
    
    #lr = linear_model.LinearRegression()
    #lr = linear_model.Ridge(alpha=i)
    lr = linear_model.Lasso(alpha=i)
    
#training and predict     
    lr.fit(feature,y)
    predicted = cross_val_predict(lr, feature, y, cv=10)
    scores = cross_val_score(lr, feature, y, cv=10,scoring='mean_squared_error')
    mse_scores = -scores
    rmse = math.sqrt(mse_scores.mean())

#plot figure

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    z = y-predicted
    fig2, bx=plt.subplots()
    bx.scatter(predicted,z)
    bx.set_xlabel('Measured')
    bx.set_ylabel('Residual')
    plt.show()

    print "rmse:", rmse