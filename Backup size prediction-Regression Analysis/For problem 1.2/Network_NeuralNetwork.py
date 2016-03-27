# -*- coding: utf-8 -*-
#This script is for the network data using neural network regression.

import neurolab as nl
from neurolab.error import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.cross_validation import KFold

#Load data
import csv
with open('network_backup_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:7] for row in reader]
    data = np.array(data_list)
    
    data[data == 'Monday'] = 1
    data[data == 'Tuesday'] = 2
    data[data == 'Wednesday'] = 3
    data[data == 'Thursday'] = 4
    data[data == 'Friday'] = 5
    data[data == 'Saturday'] = 6
    data[data == 'Sunday'] = 7
    data[data == 'work_flow_0'] = 0
    data[data == 'work_flow_1'] = 1
    data[data == 'work_flow_2'] = 2
    data[data == 'work_flow_3'] = 3
    data[data == 'work_flow_4'] = 4
    data[data == 'File_0'] = 0
    data[data == 'File_1'] = 1
    data[data == 'File_2'] = 2
    data[data == 'File_3'] = 3
    data[data == 'File_4'] = 4
    data[data == 'File_5'] = 5
    data[data == 'File_6'] = 6
    data[data == 'File_7'] = 7
    data[data == 'File_8'] = 8
    data[data == 'File_9'] = 9
    data[data == 'File_10'] = 10
    data[data == 'File_11'] = 11
    data[data == 'File_12'] = 12
    data[data == 'File_13'] = 13
    data[data == 'File_14'] = 14
    data[data == 'File_15'] = 15
    data[data == 'File_16'] = 16
    data[data == 'File_17'] = 17
    data[data == 'File_18'] = 18
    data[data == 'File_19'] = 19
    data[data == 'File_20'] = 20
    data[data == 'File_21'] = 21
    data[data == 'File_22'] = 22
    data[data == 'File_23'] = 23
    data[data == 'File_24'] = 24
    data[data == 'File_25'] = 25
    data[data == 'File_26'] = 26
    data[data == 'File_27'] = 27
    data[data == 'File_28'] = 28
    data[data == 'File_29'] = 29

#Transform into categorical variables
redata = data[1:,:].astype(np.float)
line=len(redata)
row=15+7+6+5+30
feature = np.zeros([line,row])

for i in range(0,line):
    feature[i,-1+redata[i,0]]=1
    feature[i,14+redata[i,1]]=1
    feature[i,22+int(redata[i,2]/4)]=1
    feature[i,28+redata[i,3]]=1
    feature[i,33+redata[i,4]]=1    
inp = feature   

backupSize = redata[:,5]
backupSize = backupSize.astype(np.double)
backupSize = backupSize.reshape(len(backupSize),1)
target = backupSize

#10-fold cross validation
fold10 = KFold(len(inp), n_folds=10, shuffle=True, random_state=None)

#Training and testing
test_target = np.array([])
test_predict = np.array([])
for train_index, test_index in fold10:
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    net = nl.net.newff([ [0,1] ]*63, [10,5,1])
    e = nl.train.train_rprop(net, X_train, y_train, epochs=500, show=1, goal=0.0001)
    predicted = net.sim(X_test)
    
    test_target = np.append(test_target, y_test)
    test_predict = np.append(test_predict, predicted)
  
test_target = test_target.reshape(len(inp),1)
test_predict = test_predict.reshape(len(inp),1)

#Mean Squared Error
f = MSE()
MeanSquaredError = f(test_target,test_predict)
RMSE = np.sqrt(MeanSquaredError)
print('RMSE=',RMSE)

#Plot
fig, ax = plt.subplots()
ax.scatter(test_target, test_predict)
ax.plot([backupSize.min(), backupSize.max()], [backupSize.min(), backupSize.max()], 'k--', lw=4)
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

