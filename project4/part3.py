# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 02:01:22 2016

@author: ningwang
"""
import train_featuregeneration as prj4
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm


train_feature,train_target=prj4.feature_seclection(file_str='./tweet_data/tweets_#superbowl.txt',start_year=2015,start_month=1,start_day=17,start_hour=16,
                        end_year=2015, end_month=2, end_day=8, end_hour=16)
                    
#Fvalue Pvalue tables
[F,p_value]=f_regression(train_feature, train_target)
   

'''linear'''
clf = linear_model.LinearRegression()
clf.fit(train_feature,train_target)
predicted = clf.predict(train_feature)
#r2
r_squared=clf.score(train_feature,train_target)



fig,ax = plt.subplots()
ax.scatter(train_target, predicted)
ax.plot([min(predicted),max(predicted)],[min(predicted),max(predicted)],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()


fig2 = plt.subplot()
residuals=abs(train_target-predicted)
plt.scatter(predicted,residuals)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

new_train_feature_plot= SelectKBest(f_regression, k=3).fit_transform(train_feature, train_target)
[a,b]=np.shape(new_train_feature_plot)
[m,n]=np.shape(train_feature)
index=[]
same=[]
for i in range(b):
    for j in range(n):
        same=list(set(new_train_feature_plot[:,i]==train_feature[:,j]))
#which features
        if same[0]==True:
            index.append(j)
            
#Plots of top features vs. predictant
fig_1 = plt.subplot()
plt.scatter(new_train_feature_plot[:,0],predicted)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Top feature 1')
plt.ylabel('Predictant')
plt.show()


fig_2 = plt.subplot()
plt.scatter(new_train_feature_plot[:,1],predicted)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Top feature 2')
plt.ylabel('Predictant')
plt.show()


fig_3 = plt.subplot()
plt.scatter(new_train_feature_plot[:,2],predicted)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Top feature 3')
plt.ylabel('Predictant')
plt.show()

#find common top features
new_train_feature_common= SelectKBest(f_regression, k=15).fit_transform(train_feature, train_target)
[a,b]=np.shape(new_train_feature_common)
[m,n]=np.shape(train_feature)
index_common=[]
same=[]
for i in range(b):
    for j in range(n):
        same=list(set(new_train_feature_common[:,i]==train_feature[:,j]))
 #which features
        if same[0]==True:
            index_common.append(j)
            
error=np.mean(residuals)

print r_squared
print error
print F, p_value, index, index_common 