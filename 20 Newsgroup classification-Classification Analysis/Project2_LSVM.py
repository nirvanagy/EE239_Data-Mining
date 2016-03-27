# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:28:12 2016

@author: ningwang
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
categories = [ 'comp.sys.ibm.pc.hardware','comp.graphics', 
              'comp.sys.mac.hardware','comp.os.ms-windows.misc',
              'rec.autos','rec.motorcycles',
              'rec.sport.baseball','rec.sport.hockey']
remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(categories=categories,subset='train',remove=remove) 
#twenty_train = fetch_20newsgroups(subset='train',remove=remove) 
twenty_test = fetch_20newsgroups(categories=categories,subset='test',remove=remove)

traindata=twenty_train.data
traintarget=twenty_train.target
traintarget=list(traintarget)
stop_words = text.ENGLISH_STOP_WORDS #stopwords


testdata = twenty_test.data
testtarget = twenty_test.target
testtarget = list(testtarget)

data=''
for i in range(len(traindata)):
    if twenty_train.target[i]==0:
        data+=twenty_train.data[i]

clean_traindata = []
for i in range(len(traindata)):
    temp = traindata[i]
    temp = re.sub("[,.-:/()]"," ",temp)
    temp = temp.lower()
    words = temp.split()
    after_stop=[w for w in words if not w in stop_words]
    after_stem=[stemmer2.stem(plura1) for plura1 in after_stop]
    temp=" ".join(after_stem)
    clean_traindata.append(temp)

   
clean_testdata = []
for i in range(len(testdata)):
    temp = testdata[i]
    temp = re.sub("[,.-:/()]"," ",temp)
    temp = temp.lower()
    words = temp.split()
    after_stop=[w for w in words if not w in stop_words]
    after_stem=[stemmer2.stem(plura1) for plura1 in after_stop]
    temp=" ".join(after_stem)
    clean_testdata.append(temp)   




count_vect = CountVectorizer(min_df=6,stop_words=stop_words)

X_train_counts = count_vect.fit_transform(clean_traindata)
#X_train_counts = count_vect.fit_transform(traindata)
count_train=X_train_counts.toarray()   
#print count_vect.get_feature_names()

count_vect_test = CountVectorizer(min_df=6,stop_words=stop_words)
X_test_counts = count_vect_test.fit_transform(clean_testdata)
count_test = X_test_counts.toarray()
#print count_vect_test.get_feature_names()

"""tf*idf train_data"""
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tfxidf = X_train_tfidf.toarray()
tfxidf = np.array(tfxidf)
#print X_train_tfidf.toarray()[0:30,:10]

""" fi*idf test_data"""
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
tfxidf_test = X_test_tfidf.toarray()
tfxidf_test = np.array(tfxidf_test)



#"""just for problem C: find different topics """
#"""!!!tficf!!!"""
#
#
#
filename='comp.sys.ibm.pc.hardware'
filetarget=twenty_train.target_names.index(filename)
indices=[]
for idx, elem in enumerate(traintarget):
    if elem==filetarget:  
        indices.append(idx)

t1=tfxidf[indices,:]
(a,b)=t1.shape  
t1d=np.reshape(t1,a*b)  #change to 1D
max10=sorted(t1d,reverse=True)[0:20] #20 largest

# remove repeted values
m = []
for i in max10:
    if i not in m:
        m.append(i)

yy=[]
n=0   
# print 10 most significant words           
for i in m:
    if n !=10:
        (x,y)=np.where(t1 == i)
        for i in y:
            if i not in yy:
                yy.append(i)    
#                print count_vect.get_feature_names()[i]
                n=n+1
            
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.decomposition import TruncatedSVD

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf)
#vectors_new=svd.fit_transform(tfxidf)
svd_traindata=svd.transform(tfxidf)

svd_test = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd_test.fit(tfxidf_test)
svd_testdata = svd_test.fit_transform(tfxidf_test)


CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])

computer_train=0
recrea_train=0
computer_test=0
recrea_test=0

for i in range(len(twenty_train.target)):
    if twenty_train.target[i] in CompTech:
        twenty_train.target[i] = 0
        computer_train+=1
    elif twenty_train.target[i] in Rec:
        twenty_train.target[i] = 1
        recrea_train+=1

for i in range(len(twenty_test.target)):
    if twenty_test.target[i] in CompTech:
        twenty_test.target[i] = 0
        computer_test+=1

    elif twenty_test.target[i] in Rec:
        twenty_test.target[i] = 1
        recrea_test+=1

from sklearn import svm
linear_clf = svm.LinearSVC()
linear_clf.fit(svd_traindata, twenty_train.target) 

scores = linear_clf.fit(svd_traindata, twenty_train.target).decision_function(svd_testdata)

predicted_test = linear_clf.predict(svd_testdata)
print predicted_test

from sklearn import metrics
print metrics.classification_report(twenty_test.target, predicted_test,
    target_names=['Computer Technology','Recreational Activity'])

print metrics.confusion_matrix(twenty_test.target, predicted_test)


# Compute ROC curve and area the curve

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(twenty_test.target, scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')


#mean_tpr += interp(mean_fpr, fpr, tpr)
#mean_tpr[0] = 0.0
#roc_auc = auc(fpr, tpr)
#plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))













