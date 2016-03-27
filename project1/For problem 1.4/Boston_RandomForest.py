# -*- coding: utf-8 -*-
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_predict, cross_val_score
import math
import numpy as np

'''getting data'''
import csv
with open('housing_data.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:14] for row in reader]
    data = np.array(data_list)
    data = data.astype(np.float)

feature = data[:,0:13]
y =  data[:,13]

train_features = feature
train_target = y


'''build random forest regression model'''
rf = RandomForestRegressor(n_estimators=13,max_depth=4)
rf.fit(train_features,train_target)

scores = cross_val_score(rf, train_features, train_target, cv=10,scoring='mean_squared_error')
predicted = cross_val_predict(rf, train_features, train_target, cv=10)
mse_scores = -scores
rmse = math.sqrt(mse_scores.mean())

'''plot figure'''
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

z = y-predicted
fig2=plt.subplot()
plt.scatter(predicted,z)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print 'RMSE= ',rmse