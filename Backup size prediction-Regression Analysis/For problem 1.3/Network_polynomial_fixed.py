## This script is for the network data using polynomial fitting with a fixed training and test set

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression          
import csv
from sklearn.metrics import mean_squared_error
import math

with open('network_backup_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:7] for row in reader]
    # all the data in np.array format
    data = np.array(data_list)
    
    #map data
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
 #crop data   
feature=feature[:,22:33]
feature=feature.astype(np,int)
backupSize = redata[:,5]
backupSize = backupSize.astype(np.double)

#split datasets with sampleRatio
sampleRatio = 0.7
n_sample = len(backupSize)
sampleBoundry = int(n_sample*sampleRatio)
#shuffle the whole data
shuffleidx = range(n_sample)
np.random.shuffle(shuffleidx)
#set trainset and testset
train_features = feature[shuffleidx[:sampleBoundry]]
train_target = backupSize[shuffleidx[:sampleBoundry]]
test_features = feature[shuffleidx[sampleBoundry:]]
test_target = backupSize[shuffleidx[sampleBoundry:]]

rmse=[]
rmsetrain=[]
rmsetest =[]
g = []
#get rmse and generalization error of fitted polynomial with different degrees
poly_degree=[1,2,3,4,5]
for i in poly_degree:
    
    polynomial_features = PolynomialFeatures(degree=i,include_bias=False)
    linear_regression = LinearRegression(fit_intercept = False)

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])

    pipeline.fit(train_features, train_target) 
    predicted_train=pipeline.predict(train_features)
    predicted =pipeline.predict(test_features)
    
    rmsetrain.append(math.sqrt(mean_squared_error(predicted_train,train_target)))
    rmsetest.append(math.sqrt(mean_squared_error(predicted,test_target)))
    g.append(-(math.sqrt(mean_squared_error(predicted_train,train_target))-math.sqrt(mean_squared_error(predicted,test_target))))
    
#plot prediction results
    fig, ax = plt.subplots()
    ax.scatter(test_target, predicted)
    ax.plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

#plot degree versus rmse
fig2,bx=plt.subplots()
plt.plot(poly_degree,rmsetest)
bx.set_xlabel('Degree')
bx.set_ylabel('RMSE')
plt.show()

#plot degree vs generalization errir 
fig3,bx=plt.subplots()
plt.plot(poly_degree,g)
bx.set_xlabel('Degree')
bx.set_ylabel('G')
plt.show()