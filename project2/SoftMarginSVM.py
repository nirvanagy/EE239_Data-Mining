# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:43:33 2016

@author: masenfrank
"""

from sklearn import svm
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

#load training and testing data 4 categories
categories = [ 'comp.graphics','comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', \
               'comp.os.ms-windows.misc','rec.autos','rec.motorcycles', \
               'rec.sport.baseball','rec.sport.hockey']
remove = ('headers', 'footers', 'quotes')
twenty = fetch_20newsgroups(categories=categories,subset = 'all',remove=remove) 
#twenty_train = fetch_20newsgroups(categories=categories,subset='train',remove=remove) 
#twenty_test = fetch_20newsgroups(categories=categories,subset='test',remove=remove)

data = twenty.data
target = twenty.target
#traindata=twenty_train.data
#traintarget=twenty_train.target
#traintarget=list(traintarget)

#testdata=twenty_test.data
#testtarget=twenty_test.target
#testtarget=list(testtarget)

stop_words = text.ENGLISH_STOP_WORDS #stopwords

"""
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
"""


clean_data = []
for i in range(len(data)):
    temp = data[i]
    temp = re.sub("[,.-:/()]"," ",temp)
    temp = temp.lower()
    words = temp.split()
    after_stop=[w for w in words if not w in stop_words]
    after_stem=[stemmer2.stem(plura1) for plura1 in after_stop]
    temp=" ".join(after_stem)
    clean_data.append(temp)
    
"""   
count_vect_train = CountVectorizer(min_df=6,stop_words=stop_words)
X_train_counts = count_vect_train.fit_transform(clean_traindata)
count_train=X_train_counts.toarray()

count_vect_test = CountVectorizer(min_df=6,stop_words=stop_words)
X_test_counts = count_vect_test.fit_transform(clean_testdata)
count_test=X_test_counts.toarray()   
"""

count_vect = CountVectorizer(min_df=3,stop_words=stop_words)
X_counts = count_vect.fit_transform(clean_data)
count_train=X_counts.toarray()

"""
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
"""

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
tfxidf = X_tfidf.toarray()
tfxidf = np.array(tfxidf)

"""
svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf_train)
train_features=svd.fit_transform(tfxidf_train)

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf_test)
test_features=svd.fit_transform(tfxidf_test)
"""

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(tfxidf)
features=svd.fit_transform(tfxidf)

CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])
for i in range(len(twenty.target)):
    if twenty.target[i] in CompTech:
        twenty.target[i] = 0
    elif twenty.target[i] in Rec:
        twenty.target[i] = 1
target = twenty.target


from sklearn.cross_validation import KFold
fold5 = KFold(len(features),n_folds=5, shuffle=True, random_state=None)
test_target = np.array([])
test_predict = np.array([])
for train_index, test_index in fold5:
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    clf = svm.SVC(kernel='linear',C=10)
    clf.fit(X_train, y_train) 
    predicted = clf.predict(X_test)
    test_target = np.append(test_target, y_test)
    test_predict = np.append(test_predict, predicted)

    
"""
CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])
for i in range(len(twenty_train.target)):
    if twenty_train.target[i] in CompTech:
        twenty_train.target[i] = 0
    elif twenty_train.target[i] in Rec:
        twenty_train.target[i] = 1

tar = twenty_train.target

clf = svm.SVC(C=1.0)
clf.fit(train_features, tar) 
predicted = clf.predict(test_features)
print(predicted)

CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])
for i in range(len(twenty_test.target)):
    if twenty_test.target[i] in CompTech:
        twenty_test.target[i] = 0
    elif twenty_test.target[i] in Rec:
        twenty_test.target[i] = 1
tar = twenty_test.target
"""


print(metrics.classification_report(test_target, test_predict,
    target_names=['Computer Technology','Recreational Activity']))

metrics.confusion_matrix(test_target, test_predict)
   