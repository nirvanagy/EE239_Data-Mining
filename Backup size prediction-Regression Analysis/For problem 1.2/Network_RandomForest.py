# -*- coding: utf-8 -*-
#This script is for the network data using random forest.

from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import csv

#Load data
with open('network_backup_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:7] for row in reader]
    # all the data in np.array format
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
feature=feature[:,15:]


backupSize = redata[:,5]
backupSize = backupSize.astype(np.double)


#random forest
rf = RandomForestRegressor(n_estimators=200,max_depth=40)
rf.fit(feature,backupSize)
predicted = cross_val_predict(rf, feature, backupSize, cv=10)

mse = mean_squared_error(predicted,backupSize)
rmse = np.sqrt(mse)
print('RMSE=',rmse)
residuals = abs(backupSize-predicted)

#Plot
fig,ax = plt.subplots()
ax.scatter(backupSize, predicted)
ax.plot([0,1],[0,1],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()


fig2 = plt.subplot()
plt.scatter(predicted,residuals)
plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()





