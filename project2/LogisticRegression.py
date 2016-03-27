# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:24:12 2016

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

categories = [ 'comp.graphics','comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', \
               'comp.os.ms-windows.misc','rec.autos','rec.motorcycles', \
               'rec.sport.baseball','rec.sport.hockey']
remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(categories=categories,subset='train',remove=remove) 
twenty_test = fetch_20newsgroups(categories=categories,subset='test',remove=remove)

traindata=twenty_train.data
traintarget=twenty_train.target
traintarget=list(traintarget)

testdata=twenty_test.data
testtarget=twenty_test.target
testtarget=list(testtarget)

stop_words = text.ENGLISH_STOP_WORDS #stopwords

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

CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])
for i in range(len(twenty_train.target)):
    if twenty_train.target[i] in CompTech:
        twenty_train.target[i] = 0
    elif twenty_train.target[i] in Rec:
        twenty_train.target[i] = 1
train_target = twenty_train.target

CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])
for i in range(len(twenty_test.target)):
    if twenty_test.target[i] in CompTech:
        twenty_test.target[i] = 0
    elif twenty_test.target[i] in Rec:
        twenty_test.target[i] = 1
test_target = twenty_test.target

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=0.007)
logreg.fit(train_features, train_target)
test_predict = logreg.predict(test_features)


target_score = logreg.fit(train_features, train_target).decision_function(test_features)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(test_target, target_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], '--',color=(0.6, 0.6, 0.6), label='Luck')

print metrics.classification_report(twenty_test.target, test_predict,
    target_names=['Computer Technology','Recreational Activity'])

print metrics.confusion_matrix(twenty_test.target, test_predict)