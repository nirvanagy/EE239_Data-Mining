# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:00:26 2016

@author: masenfrank
"""
import neurolab as nl
from neurolab.error import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.cross_validation import KFold

#Load data
import csv
with open('housing_data.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:14] for row in reader]
    data = np.array(data_list)
    data = data.astype(np.float)

#Normalize input and output    
y = data[:,13]
length = len(y)
outp = y.reshape(length,1)
target = outp / outp.max(axis=0)

x = data[:,0:13]
inp = x / x.max(axis=0)

#10-fold cross validation
fold10 = KFold(len(inp), n_folds=10, shuffle=True, random_state=None)

#Trainding and Testing
test_target = np.array([])
test_predict = np.array([])
for train_index, test_index in fold10:
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    net = nl.net.newff([ [0,1] ]*13, [10,5,1])
    e = nl.train.train_rprop(net, X_train, y_train, epochs=1500, show=100, goal=0.0001)
    predicted = net.sim(X_test)
    
    test_target = np.append(test_target, y_test)
    test_predict = np.append(test_predict, predicted)
  
test_target = test_target.reshape(len(inp),1)
test_predict = test_predict.reshape(len(inp),1)

#Calculate MSE
f = MSE()
test_target = test_target * outp.max(axis=0)
test_predict = test_predict * outp.max(axis=0)
MeanSquaredError = f(test_target,test_predict)

#Plot
fig, ax = plt.subplots()
ax.scatter(test_target, test_predict)
ax.plot([test_target.min(), test_target.max()], [test_predict.min(), test_predict.max()], 'k--', lw=4)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
plt.show()

residual = abs(test_target - test_predict)
fig, ax = plt.subplots()
ax.scatter(test_predict, residual)
ax.plot([test_predict.min(), test_predict.max()], [0, 0], 'k--', lw=4)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
plt.show()
