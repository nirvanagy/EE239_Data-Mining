# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 21:47:04 2016

@author: YangG
"""


from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import string

import re
import nltk.stem
stemmer2 = nltk.stem.SnowballStemmer('english')

#load training and testing data 4 categories

remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(subset='train',remove=remove) 
#twenty_train = fetch_20newsgroups(subset='train',remove=remove) 

traindata=twenty_train.data
traintarget=twenty_train.target
traintarget=list(traintarget)
stop_words = text.ENGLISH_STOP_WORDS #stopwords

DATA=[];
for j in range(0,20):
    data=''
    for i in range(len(traindata)):
        if twenty_train.target[i]==j:
            data+=twenty_train.data[i]
    DATA.append(data)
    
clean_traindata = []
for i in range(len(DATA)):
    temp = DATA[i]
    temp = re.sub("[,.-:/()]"," ",temp)
    temp = temp.lower()
    words = temp.split()
    after_stop=[w for w in words if not w in stop_words]
    after_stem=[stemmer2.stem(plura1) for plura1 in after_stop]
    temp=" ".join(after_stem)
    clean_traindata.append(temp)

count_vect = CountVectorizer(min_df=1,stop_words=stop_words)

X_train_counts_icf = count_vect.fit_transform(clean_traindata)
#X_train_counts = count_vect.fit_transform(traindata)
count_train_icf=X_train_counts_icf.toarray() 

tficf_transformer = TfidfTransformer()
X_train_tficf = tficf_transformer.fit_transform(X_train_counts_icf)
tfxicf = X_train_tficf.toarray()
tfxicf = np.array(tfxicf)


filename1='comp.sys.ibm.pc.hardware'
filetarget1=twenty_train.target_names.index(filename1)
max10_1=sorted(tfxicf[filetarget1,:],reverse=True)[0:10]
for i in max10_1:
    (x,y)=np.where(tfxicf == i)
    for i in y:
        print count_vect.get_feature_names()[i]        
print '\n'


filename2='comp.sys.mac.hardware'
filetarget2=twenty_train.target_names.index(filename2)
max10_2=sorted(tfxicf[filetarget2,:],reverse=True)[0:10]
for i in max10_2:
    (x,y)=np.where(tfxicf == i)
    for i in y:
        print count_vect.get_feature_names()[i]
print '\n'
     

filename3='misc.forsale'
filetarget3=twenty_train.target_names.index(filename3)
max10_3=sorted(tfxicf[filetarget3,:],reverse=True)[0:10]
for i in max10_3:
    (x,y)=np.where(tfxicf == i)
    for i in y:
        print count_vect.get_feature_names()[i]
print '\n'
        
filename4='soc.religion.christian'
filetarget4=twenty_train.target_names.index(filename4)
max10_4=sorted(tfxicf[filetarget4,:],reverse=True)[0:10]
for i in max10_4:
    (x,y)=np.where(tfxicf == i)
    for i in y:
        print count_vect.get_feature_names()[i]
print '\n'
        