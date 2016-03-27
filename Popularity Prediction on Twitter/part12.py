# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:20:40 2016

@author: YangG
"""
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm

time_h=[]
time_=[]
author=[]
follower=[]
retweet=[]
uniauthor_follower=[]
allhashtags=[]
user_mentions=[]
media_url=[]

data = []
f = open('./tweet_data/tweets_#gopatriots.txt','r')
for line in f:
    ff = json.loads(line)
    data.append(ff)
    t=ff['tweet']['created_at']
    t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
    if t.tm_year==2015:
        t=(t.tm_mon-1)*31*24+(t.tm_mday-18)*24+t.tm_hour+t.tm_min/60.0
        if t>=0:
            time_h.append(int(t))
            time_.append(t)
            author.append(ff['author']['nick'])
            follower.append(ff['author']['followers'])
            retweet.append(ff['metrics']['citations']['total'])
            allhashtags.append(len(ff['tweet']['entities']['hashtags']))                        
            user_mentions.append(len(ff['tweet']['entities']['user_mentions']))                
            if ('media' in ff ['tweet']['entities'].keys()) or len(ff['tweet']['entities']['urls'])>0 :           
                media_url.append(1)
            else:
                media_url.append(0)    

uniauthor_follower={}

for i in range(len(author)):
    uniauthor_follower[author[i]]=follower[i]

'''followers'''   
sum_follower=sum(uniauthor_follower.values())
avg_follower=sum_follower/len(uniauthor_follower)
max_follower=max(uniauthor_follower.values())            
            
tweet_h=[]
retweet_h=[]
follower_h=[]
mfollower_h=[]
hour=[]
allhashtags_h=[]
user_mentions_h=[]
media_url_h=[]

m=0
n=0
timemax=504
for i in range(0,timemax):
    m=time_h.count(i)
    
    if m!=0:
        tweet_h.append(m)    
        retweet_h.append(sum(retweet[n:n+m]))
        hour.append(i%24)
        allhashtags_h.append(sum(allhashtags[n:n+m]))
        user_mentions_h.append(sum(user_mentions[n:n+m]))       
        media_url_h.append(sum(media_url[n:n+m]))
        
        uniauthor_h=set(author[n:n+m])
        afollower=[]
        for j in uniauthor_h:    
            afollower.append(uniauthor_follower[j])
        follower_h.append(sum(afollower) ) 
        mfollower_h.append(max(afollower))
        n=n+m
    else:
        tweet_h.append(0)
        retweet_h.append(0)
        hour.append(i%24)
        follower_h.append(0)
        mfollower_h.append(0)
        allhashtags_h.append(0)
        user_mentions_h.append(0)       
        media_url_h.append(0)
''' '''

       
tweet_h=np.array(tweet_h)
retweet_h=np.array(retweet_h)
follower_h=np.array(follower_h)
mfollower_h=np.array(mfollower_h)
hour=np.array(hour)
allhashtags_h=np.array(allhashtags_h)
user_mentions_h=np.array(user_mentions_h)    
media_url_h=np.array(media_url_h) 

line=len(hour)
row=24
newhour = np.zeros([line,row])
for i in range(line):
    newhour[i,hour[i]]=1


l=len(tweet_h)
#feature=np.column_stack((hour[0:l-1],tweet_h[0:l-1],retweet_h[0:l-1],follower_h[0:l-1],mfollower_h[0:l-1]))
feature=np.column_stack((newhour[0:l-1],tweet_h[0:l-1],retweet_h[0:l-1],follower_h[0:l-1],mfollower_h[0:l-1]))
target=tweet_h[1:l]


'''average tweet retweet'''
plt.bar(range(len(tweet_h)),tweet_h)
avg_tweet=sum(tweet_h)*1.0/len(tweet_h)
n_retweet=sum(retweet_h)
avg_retweet=n_retweet*1.0/sum(tweet_h)

'''linear regression'''
lr = linear_model.LinearRegression()
lr.fit(feature,target)
coeff=lr.coef_
lr.score(feature,target)

#predicted = lr.predict(feature)
predicted = cross_val_predict(lr, feature, target, cv=10)
scores = cross_val_score(lr, feature, target, cv=10,scoring='mean_absolute_error')
error=np.mean(scores)
#mse=mean_squared_error(predicted,target)
#ttest=stats.ttest_rel(predicted,target)


'''OLS'''
model = sm.OLS(target, feature)
results = model.fit()
print(results.summary())


fig,ax = plt.subplots()
ax.scatter(target, predicted)
ax.plot([min(predicted),max(predicted)],[min(predicted),max(predicted)],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()

residuals=abs(target-predicted)
fig2 = plt.subplot()
plt.scatter(predicted,residuals)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

















  

