# -*- coding: utf-8 -*-

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
categories = [ 'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
                  'misc.forsale','soc.religion.christian']
                  
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
count=X_train_counts.toarray()   
#print count_vect.get_feature_names()

count_vect_test = CountVectorizer(min_df=6,stop_words=stop_words)
X_test_counts = count_vect_test.fit_transform(clean_testdata)
count_test = X_test_counts.toarray
#print count_vect_test.get_feature_names()

"""tf*idf train_data"""
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tfxidf = X_train_tfidf.toarray()
tfxidf = np.array(tfxidf)
print X_train_tfidf.toarray()[0:30,:10]

""" fi*idf test_data"""
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
tfxidf_test = X_test_tfidf.toarray()
tfxidf_test = np.array(tfxidf_test)


from sklearn.decomposition import TruncatedSVD

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf)
#vectors_new=svd.fit_transform(tfxidf)
svd_traindata=svd.transform(tfxidf)

svd_test = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd_test.fit(tfxidf_test)
svd_testdata = svd_test.fit_transform(tfxidf_test)


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
predicted_test = OneVsRestClassifier(LinearSVC(C=10,random_state=0)).fit(svd_traindata,
                         twenty_train.target).predict(svd_testdata)

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted_test,
    target_names=categories))

metrics.confusion_matrix(twenty_test.target, predicted_test)










