import re

import json
import time
import numpy as np
file_str='./test_data/sample10_period3.txt'

path = file_str
hashtags = []
allhashtags=[]
user_mentions=[]
time_h=[]
time_ =[]
test =1
highlight=[]
count =[]

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
                        t.tm_hour-start_date.tm_hour+t.tm_min/60.0)
        if t2>=0:
            time_h.append(int(t2))
            time_.append(t2)
            
            
            highlight.append(ff['highlight'])            
            match = re.search( r'#gohawks', ff['highlight'], re.M|re.I)
            count.append(len(re.findall("#",ff['highlight'])))
            if match:
                allhashtags.append(1)
            else:
                allhashtags.append(0) 
                
c=allhashtags.count(1)                
print 'total hashtags:', c  
                     
"""results"""

#superbowel 1,2,3,9
#hawks 4,5,6,7
#nfl 8,10
#



