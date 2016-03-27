import json
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn import linear_model
import math
from sklearn.ensemble import RandomForestRegressor

def open_file (file_str,hashtags, time_h, time_,follower,
               typ,allhashtags,user_mentions,favorite_count,
               retweet_count,original_author_follower,media_url):
    string1 = './tweet_data/tweets_#'
    string2 = '.txt'
    path = string1+file_str+string2
    f = open(path,'r')
    for line in f:
        ff= json.loads(line)
        hashtags.append(ff)
        t=ff['tweet']['created_at']
        t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
        if t.tm_year==2015:
            t2=(t.tm_mon-1)*24*31+(t.tm_mday-18)*24+t.tm_hour+t.tm_min/60.0
            if t2>=0:
                time_h.append(int(t2))
                time_.append(t2)
                """ feature seclection"""
#                author.append(ff['author']['nick'])
                follower.append(ff['author']['followers'])
                if ff['type']=='tweet':
                    typ.append(0)
                else:
                    typ.append(1)                                   
                favorite_count.append(ff['tweet']['favorite_count'])
                retweet_count.append(ff['tweet']['retweet_count'])
                original_author_follower.append(ff['original_author']['followers']) 
#                for i in range(len(ff['tweet']['entities']['hashtags'])):
#                    allhashtags.append(ff['tweet']['entities']['hashtags'][i]['text'])
                allhashtags.append(len(ff['tweet']['entities']['hashtags']))
#                for i in range(len(ff['tweet']['entities']['user_mentions'])):
#                    user_mentions.append(ff['tweet']['entities']['user_mentions'][i]['id'])                             
                user_mentions.append(len(ff['tweet']['entities']['user_mentions']))                
                if ('media' in ff ['tweet']['entities'].keys()) or len(ff['tweet']['entities']['urls'])>0 :           
                    media_url.append(1)
                else:
                    media_url.append(0)                
        if t.tm_year==2015 and t.tm_mon==2 and t.tm_mday ==11:
            break

    return (hashtags, time_h, time_,follower,typ,allhashtags,user_mentions,
            favorite_count,retweet_count,original_author_follower,media_url)

gopatriots = []
time_h = []
time_ = []  
author = []
follower = []
typ = [] 
allhashtags = [] 
user_mentions = []
favorite_count = [] 
retweet_count = [] 
original_author_follower = []
media_url = []
open_file(file_str = 'nfl',hashtags = gopatriots,time_h=time_h,
          time_ = time_, follower = follower, typ = typ
          ,allhashtags= allhashtags,user_mentions=user_mentions,
          favorite_count =favorite_count,retweet_count =retweet_count
          ,original_author_follower = original_author_follower,
          media_url = media_url)


n_tweet = []
for i in range(0,504):
    n_tweet.append(time_h.count(i))
plt.bar(range(504),n_tweet)
 
feature = np.zeros([len(allhashtags),7],dtype =int) 
feature[:,0] = retweet_count
feature[:,1] = follower
feature[:,2] = media_url
feature[:,3] = allhashtags
feature[:,4] = user_mentions
feature[:,5] = original_author_follower
feature[:,6] = typ

target = np.zeros([len(allhashtags)])
target[:] = favorite_count


# build linear regression
lr = linear_model.LinearRegression()
lr.fit(feature,target)

#predict 
scores = cross_val_score(lr, feature, target, cv=10,scoring='mean_squared_error')
predicted = cross_val_predict(lr, feature, target, cv=10)
mse_scores = -scores
rmse = math.sqrt(mse_scores.mean())

#figre plot
fig, ax = plt.subplots()
ax.scatter(target, predicted)
ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
ax.axis([-10,100,-10,100])
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

'''build random forest regression model'''
rf = RandomForestRegressor(n_estimators=13,max_depth=4)
rf.fit(feature,target)

scores = cross_val_score(rf, feature, target, cv=10,scoring='mean_squared_error')
predicted = cross_val_predict(rf, feature, target, cv=10)
mse_scores = -scores
rmse = math.sqrt(mse_scores.mean())

'''plot figure'''
fig, ax = plt.subplots()
ax.scatter(target, predicted)
ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
ax.axis([-10,100,-10,100])
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
