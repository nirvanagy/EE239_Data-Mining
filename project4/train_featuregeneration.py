import json
import time
import numpy as np


def feature_seclection (file_str,start_year=2015,start_month=1, start_day=17 ,start_hour=0,
                        end_year=2015, end_month=2, end_day=8, end_hour=0):
                            
    path = file_str
    hashtags = []
    author = []
    follower = []
    typ =[]
    allhashtags=[]
    user_mentions=[]
    media_url = []
    time_h=[]
    time_ =[]
    f = open(path,'r')
    for line in f:
        ff= json.loads(line)
        hashtags.append(ff)
        t=ff['tweet']['created_at']
        t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
        if t.tm_year==start_year:
            t2=((t.tm_mon-start_month)*24*31+(t.tm_mday-start_day)*24+t.tm_hour-start_hour+t.tm_min/60.0)*3
            if t2>=0:
                time_h.append(int(t2))
                time_.append(t2)
                
                """ feature seclection"""
                author.append(ff['author']['nick'])
                follower.append(ff['author']['followers'])
                typ.append(ff['type'])
                allhashtags.append(len(ff['tweet']['entities']['hashtags']))                        
                user_mentions.append(len(ff['tweet']['entities']['user_mentions']))                
                if ('media' in ff ['tweet']['entities'].keys()) or len(ff['tweet']['entities']['urls'])>0 :           
                    media_url.append(1)
                else:
                    media_url.append(0) 
                                                           
        if t.tm_year==end_year and t.tm_mon==end_month and t.tm_mday ==end_day and t.tm_hour == end_hour:
            break
    
    uniauthor_follower={}
    
    for i in range(len(author)):
        uniauthor_follower[author[i]]=follower[i]
    
    '''followers'''   
    #sum_follower=sum(uniauthor_follower.values())
    #avg_follower=sum_follower/len(uniauthor_follower)
    #max_follower=max(uniauthor_follower.values())            
                
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
    timemax=max(time_h)
    for i in range(0,timemax+1):
        m=time_h.count(i)
        
        if m!=0:
            tweet_h.append(m)    
            retweet_h.append(typ[n:n+m].count('retweet:native')+typ[n:n+m].count('retweet:reply'))
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

    tweet_h=np.array(tweet_h)
    retweet_h=np.array(retweet_h)
    follower_h=np.array(follower_h)
    mfollower_h=np.array(mfollower_h)
    hour=np.array(hour)
    allhashtags_h=np.array(allhashtags_h)
    user_mentions_h=np.array(user_mentions_h)    
    media_url_h=np.array(media_url_h) 
    
    """ Build time series window"""
    tweet_TimeMean = []
    tweet_TimeStd = []
    tweet_TimeDiff = []
    
    retweet_TimeMean = []
    retweet_TimeStd = []
    retweet_TimeDiff =[]
    
    follower_TimeMean = []
    follower_TimeStd = []
    follower_TimeDiff =[]
    
    mfollower_TimeMean = []
    mfollower_TimeStd = []
    mfollower_TimeDiff =[]
    
    allhashtags_TimeMean = []
    allhashtags_TimeStd = []
    allhashtags_TimeDiff =[]
    
    user_mentions_TimeMean = []
    user_mentions_TimeStd = []
    user_mentions_TimeDiff =[]
    
    media_url_TimeMean = []
    media_url_TimeStd = []
    media_url_TimeDiff = []
    
    for i in range(len(tweet_h)-11):
        tweet_TimeMean.append(np.mean(tweet_h[i:i+9]))
        tweet_TimeStd.append(np.std(tweet_h[i:i+9]))
        tweet_TimeDiff.append(np.mean(abs(np.diff(tweet_h[i:i+9]))))
        
        retweet_TimeMean.append(np.mean(retweet_h[i:i+9]))
        retweet_TimeStd.append(np.std(retweet_h[i:i+9]))
        retweet_TimeDiff.append(np.mean(abs(np.diff(retweet_h[i:i+9]))))
        
        follower_TimeMean.append(np.mean(follower_h[i:i+9]))
        follower_TimeStd.append(np.std(follower_h[i:i+9]))
        follower_TimeDiff.append(np.mean(abs(np.diff(follower_h[i:i+9]))))
        
        mfollower_TimeMean.append(np.mean(mfollower_h[i:i+9]))
        mfollower_TimeStd.append(np.std(mfollower_h[i:i+9]))
        mfollower_TimeDiff.append(np.mean(abs(np.diff(mfollower_h[i:i+9]))))
        
        allhashtags_TimeMean.append(np.mean(allhashtags_h[i:i+9]))
        allhashtags_TimeStd.append(np.std(allhashtags_h[i:i+9]))
        allhashtags_TimeDiff.append(np.mean(abs(np.diff(allhashtags_h[i:i+9]))))
        
        user_mentions_TimeMean.append(np.mean(user_mentions_h[i:i+9]))
        user_mentions_TimeStd.append(np.std(user_mentions_h[i:i+9]))
        user_mentions_TimeDiff.append(np.mean(abs(np.diff(user_mentions_h[i:i+9]))))
        
        media_url_TimeMean.append(np.mean(media_url_h[i:i+9]))
        media_url_TimeStd.append(np.std(media_url_h[i:i+9]))
        media_url_TimeDiff.append(np.mean(abs(np.diff(media_url_h[i:i+9]))))    
        

    feature = np.column_stack((tweet_TimeMean,tweet_TimeStd,tweet_TimeDiff,
                              retweet_TimeMean,retweet_TimeStd,retweet_TimeDiff,
                              follower_TimeMean,follower_TimeStd,follower_TimeDiff,
                              mfollower_TimeMean,mfollower_TimeStd,mfollower_TimeDiff,
#                              allhashtags_TimeMean,allhashtags_TimeStd,allhashtags_TimeDiff,
                              user_mentions_TimeMean,user_mentions_TimeStd,user_mentions_TimeDiff,
                              media_url_TimeMean,media_url_TimeStd,media_url_TimeDiff))
    
    target =  np.array(tweet_h[9:len(tweet_h)-2])+np.array(tweet_h[10:len(tweet_h)-1])+np.array(tweet_h[11:len(tweet_h)])
#    target.shape=(1,len(tweet_h)-11)
#    target = np.transpose(target)
    return (feature,target)
 
 
def sample_seclection (file_str):
    path = file_str
    hashtags = []
    author = []
    follower = []
    retweet =[]
    allhashtags=[]
    user_mentions=[]
    media_url = []
    time_h=[]
    time_ =[]
    test =1
    f = open(path,'r')
    for line in f:
        if test ==1: 
            start_line = json.loads(line)
            start_date = start_line['tweet']['created_at']
            start_date = time.strptime(start_date, "%a %b %d  %H:%M:%S +0000 %Y")
            test=0
        ff= json.loads(line)
        hashtags.append(ff)
        t=ff['tweet']['created_at']
        t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
        if t.tm_year==start_date.tm_year:
            t2=((t.tm_mon-start_date.tm_mon)*24*31+(t.tm_mday-start_date.tm_mday)*24+
                            t.tm_hour-start_date.tm_hour+t.tm_min/60.0)*3
            if t2>=0:
                time_h.append(int(t2))
                time_.append(t2)
                
                """ feature seclection"""
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
    timemax=max(time_h)
    for i in range(0,timemax+1):
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

    tweet_h=np.array(tweet_h)
    retweet_h=np.array(retweet_h)
    follower_h=np.array(follower_h)
    mfollower_h=np.array(mfollower_h)
    hour=np.array(hour)
    allhashtags_h=np.array(allhashtags_h)
    user_mentions_h=np.array(user_mentions_h)    
    media_url_h=np.array(media_url_h) 
    
    """ Build time series window"""
    tweet_TimeMean = []
    tweet_TimeStd = []
    tweet_TimeDiff = []
    
    retweet_TimeMean = []
    retweet_TimeStd = []
    retweet_TimeDiff =[]
    
    follower_TimeMean = []
    follower_TimeStd = []
    follower_TimeDiff =[]
    
    mfollower_TimeMean = []
    mfollower_TimeStd = []
    mfollower_TimeDiff =[]
    
    allhashtags_TimeMean = []
    allhashtags_TimeStd = []
    allhashtags_TimeDiff =[]
    
    user_mentions_TimeMean = []
    user_mentions_TimeStd = []
    user_mentions_TimeDiff =[]
    
    media_url_TimeMean = []
    media_url_TimeStd = []
    media_url_TimeDiff = []
 
    for i in range(len(tweet_h)-11):
        tweet_TimeMean.append(np.mean(tweet_h[i:i+9]))
        tweet_TimeStd.append(np.std(tweet_h[i:i+9]))
        tweet_TimeDiff.append(np.mean(abs(np.diff(tweet_h[i:i+9]))))
        
        retweet_TimeMean.append(np.mean(retweet_h[i:i+9]))
        retweet_TimeStd.append(np.std(retweet_h[i:i+9]))
        retweet_TimeDiff.append(np.mean(abs(np.diff(retweet_h[i:i+9]))))
        
        follower_TimeMean.append(np.mean(follower_h[i:i+9]))
        follower_TimeStd.append(np.std(follower_h[i:i+9]))
        follower_TimeDiff.append(np.mean(abs(np.diff(follower_h[i:i+9]))))
        
        mfollower_TimeMean.append(np.mean(mfollower_h[i:i+9]))
        mfollower_TimeStd.append(np.std(mfollower_h[i:i+9]))
        mfollower_TimeDiff.append(np.mean(abs(np.diff(mfollower_h[i:i+9]))))
        
        allhashtags_TimeMean.append(np.mean(allhashtags_h[i:i+9]))
        allhashtags_TimeStd.append(np.std(allhashtags_h[i:i+9]))
        allhashtags_TimeDiff.append(np.mean(abs(np.diff(allhashtags_h[i:i+9]))))
        
        user_mentions_TimeMean.append(np.mean(user_mentions_h[i:i+9]))
        user_mentions_TimeStd.append(np.std(user_mentions_h[i:i+9]))
        user_mentions_TimeDiff.append(np.mean(abs(np.diff(user_mentions_h[i:i+9]))))
        
        media_url_TimeMean.append(np.mean(media_url_h[i:i+9]))
        media_url_TimeStd.append(np.std(media_url_h[i:i+9]))
        media_url_TimeDiff.append(np.mean(abs(np.diff(media_url_h[i:i+9]))))    
        

    feature = np.column_stack((tweet_TimeMean,tweet_TimeStd,tweet_TimeDiff,
                              retweet_TimeMean,retweet_TimeStd,retweet_TimeDiff,
                              follower_TimeMean,follower_TimeStd,follower_TimeDiff,
                              mfollower_TimeMean,mfollower_TimeStd,mfollower_TimeDiff,
#                              allhashtags_TimeMean,allhashtags_TimeStd,allhashtags_TimeDiff,
                              user_mentions_TimeMean,user_mentions_TimeStd,user_mentions_TimeDiff,
                              media_url_TimeMean,media_url_TimeStd,media_url_TimeDiff))
    
    target =  np.array(tweet_h[9:len(tweet_h)-2])+np.array(tweet_h[10:len(tweet_h)-1])+np.array(tweet_h[11:len(tweet_h)])
#    target.shape=(1,len(tweet_h)-11)
#    target = np.transpose(target)

    return(feature,target,start_date.tm_mon,start_date.tm_mday,start_date.tm_hour)
       
 