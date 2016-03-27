# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression         
from sklearn import datasets  
import math
from sklearn.metrics import mean_squared_error
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


#split datasets with sampleRatio
sampleRatio = 0.9
n_sample = len(y)
sampleBoundry = int(n_sample*sampleRatio)

#shuffle the whole data
shuffleidx = range(n_sample)
np.random.shuffle(shuffleidx)

train_features = feature[shuffleidx[:sampleBoundry]]
train_target = y[shuffleidx[:sampleBoundry]]

test_features = feature[shuffleidx[sampleBoundry:]]
test_target = y[shuffleidx[sampleBoundry:]]

#training
rmse=[]
rmsetrain=[]
rmsetest =[]
g = []
poly_degree=[1,2,3,4,5,6]
for i in poly_degree:
    
    polynomial_features = PolynomialFeatures(degree=i,include_bias=False)
    linear_regression = LinearRegression(fit_intercept = False)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
    
    
    '''train and predict'''
    pipeline.fit(train_features, train_target) 
    predicted_train=pipeline.predict(train_features)
    predicted =pipeline.predict(test_features)
    
    rmsetrain.append(math.sqrt(mean_squared_error(predicted_train,train_target)))
    rmsetest.append(math.sqrt(mean_squared_error(predicted,test_target)))
    g.append(-(math.sqrt(mean_squared_error(predicted_train,train_target))-math.sqrt(mean_squared_error(predicted,test_target))))
    
    

    #plot
    fig, ax = plt.subplots()
    ax.scatter(test_target, predicted)
    ax.plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

#plot ploy_degree versus rmse
print "RMSE: ", rmsetest
fig2,bx=plt.subplots()
#plt.plot(poly_degree,rmsetest)
plt.plot(poly_degree,g)
bx.set_xlabel('Degree')
bx.set_ylabel('G')
plt.show()
    

