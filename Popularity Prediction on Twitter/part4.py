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


train_feature,train_target=prj4.feature_seclection(file_str='./tweet_data/tweets_#patriots.txt',start_year=2015,start_month=1,start_day=17,start_hour=16,
                        end_year=2015, end_month=2, end_day=1, end_hour=16)
                        
                   
test_feature,test_target,startmon,startday,starthour=prj4.sample_seclection (file_str='./test_data/sample5_period1.txt')


new_train_feature= SelectKBest(f_regression, k=10).fit_transform(train_feature, train_target)
[F,p_value]=f_regression(train_feature, train_target)
   
[a,b]=np.shape(new_train_feature)
[m,n]=np.shape(train_feature)
index=[]

for i in range(b):
    for j in range(n):
        same=list(set(new_train_feature[:,i]==train_feature[:,j]))

        if same[0]==True:
            index.append(j)
new_test_feature=test_feature[:,0]
for i in index[1:]:
    new_test_feature=np.column_stack((new_test_feature,test_feature[:,i]))
    
'''gradient boosting'''   
#params = {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 1,
#          'learning_rate': 0.01, 'loss': 'ls'}          
#clf = ensemble.GradientBoostingRegressor(**params)
#clf.fit(new_train_feature, train_target)

'''linear'''
clf = linear_model.LinearRegression()
clf.fit(new_train_feature,train_target)
coeff=clf.coef_

'''logistic'''
#clf=linear_model.LogisticRegression()
#clf.fit(new_train_feature,train_target)

'''svr'''
#clf = svm.SVR()
#clf.fit(new_train_feature,train_target)

#predicted = clf.predict(new_test_feature)
new_test_feature=new_train_feature
test_target=train_target
predicted=cross_val_predict(clf, new_test_feature, test_target, cv=10)
scores = cross_val_score(clf,new_test_feature,test_target,cv=10,scoring='mean_absolute_error')
mse=mean_squared_error(predicted,test_target)
residuals=abs(test_target-predicted)
error=np.mean(residuals)

fig,ax = plt.subplots()
ax.scatter(test_target, predicted)
ax.plot([min(predicted),max(predicted)],[min(predicted),max(predicted)],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()


fig2 = plt.subplot()
plt.scatter(predicted,residuals)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

print error
print scores