# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:07:18 2016

@author: masenfrank
"""

from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

import re
import nltk.stem
stemmer2 = nltk.stem.SnowballStemmer('english')

categories = [ 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(categories=categories,subset='train',remove=remove) 
twenty_test = fetch_20newsgroups(categories=categories,subset='test',remove=remove)

traindata=twenty_train.data
traintarget=twenty_train.target
traintarget=list(traintarget)

testdata=twenty_test.data
testtarget=twenty_test.target
testtarget=list(testtarget)

stop_words = text.ENGLISH_STOP_WORDS #stopword

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

count_vect_train = CountVectorizer(min_df=6,stop_words=stop_words)
X_train_counts = count_vect_train.fit_transform(clean_traindata)
count_train=X_train_counts.toarray()

count_vect_test = CountVectorizer(min_df=6,stop_words=stop_words)
X_test_counts = count_vect_test.fit_transform(clean_testdata)
count_test=X_test_counts.toarray()   

#tf*idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tfxidf_train = X_train_tfidf.toarray()
tfxidf_train = np.array(tfxidf_train)

#tf*idf
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
tfxidf_test = X_test_tfidf.toarray()
tfxidf_test = np.array(tfxidf_test)

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf_train)
train_features=svd.fit_transform(tfxidf_train)

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf_test)
test_features=svd.fit_transform(tfxidf_test)

train_target = twenty_train.target
test_target = twenty_test.target


from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
test_predict = OneVsOneClassifier(SVC(C=300,random_state=42)).fit(train_features, train_target).predict(test_features)

print(metrics.classification_report(test_target, test_predict,
    target_names=categories))

metrics.confusion_matrix(test_target, test_predict)
