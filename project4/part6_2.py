import re

import json
import time
import numpy as np
import random
import nltk.stem
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

file_str='./tweet_data/tweets_#gopatriots.txt'
file_str1='./tweet_data/tweets_#gohawks.txt'
file_str2='./tweet_data/tweets_#nfl.txt'

path = file_str
hashtags = []
time_h=[]
time_ =[]
highlight=[]
f = open(path,encoding='utf-8')
for line in f:
    ff= json.loads(line)
    hashtags.append(ff)
    t=ff['tweet']['created_at']
    t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
    if t.tm_year==2015:
        t2=((t.tm_mon-1)*24*31+(t.tm_mday-18)*24+
                        t.tm_hour-8+t.tm_min/60.0)
        if t2>=0:
            time_h.append(int(t2))
            time_.append(t2)
            highlight.append(ff['highlight'])            
    if t.tm_year==2015 and t.tm_mon==2 and t.tm_mday ==7 and t.tm_hour == 16:
        break

path1 = file_str1
hashtags = []
time_h=[]
time_ =[]
highlight1=[]
f = open(path1,encoding='utf-8')
for line in f:
    ff= json.loads(line)
    hashtags.append(ff)
    t=ff['tweet']['created_at']
    t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
    if t.tm_year==2015:
        t2=((t.tm_mon-1)*24*31+(t.tm_mday-18)*24+
                        t.tm_hour-8+t.tm_min/60.0)
        if t2>=0:
            time_h.append(int(t2))
            time_.append(t2)
            highlight1.append(ff['highlight'])            
    if t.tm_year==2015 and t.tm_mon==2 and t.tm_mday ==7 and t.tm_hour == 16:
        break

path2 = file_str2
hashtags = []
time_h=[]
time_ =[]
highlight2=[]
highlight3=[]
f = open(path2,encoding='utf-8')
for line in f:
    ff= json.loads(line)
    hashtags.append(ff)
    t=ff['tweet']['created_at']
    t=time.strptime(t, "%a %b %d  %H:%M:%S +0000 %Y")
    if t.tm_year==2015:
        t2=((t.tm_mon-1)*24*31+(t.tm_mday-18)*24+
                        t.tm_hour-8+t.tm_min/60.0)
        if t2>=0:
            time_h.append(int(t2))
            time_.append(t2)
            
            match1 = re.search( r'#gopatriots', ff['highlight'], re.M|re.I)
            if match1:
                highlight2.append(ff['highlight'])   
            match2 = re.search( r'#gohawks', ff['highlight'], re.M|re.I)
            if match2:
                highlight3.append(ff['highlight']) 
    if t.tm_year==2015 and t.tm_mon==2 and t.tm_mday ==7 and t.tm_hour == 16:
        break


train_hawks = random.sample(highlight1,2500)
train_patriots = random.sample(highlight,2500)
traindata = train_patriots+train_hawks

train_target_patriots = [0]*len(train_patriots)
train_target_hawks = [1]*len(train_hawks)
traintarget = np.array(train_target_patriots+train_target_hawks)

test_hawks = random.sample(highlight3,1000)
test_patriots = highlight2
testdata = test_patriots+test_hawks

test_target_patriots = [0]*len(test_patriots)
test_target_hawks = [1]*len(test_hawks)
testtarget = np.array(test_target_patriots+test_target_hawks)



stop_words = text.ENGLISH_STOP_WORDS #stopwords
stemmer2 = nltk.stem.SnowballStemmer('english')
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

count_vect_train = CountVectorizer(stop_words=stop_words)
X_train_counts = count_vect_train.fit_transform(clean_traindata)
count_train=X_train_counts.toarray()

count_vect_test = CountVectorizer(stop_words=stop_words)
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
test_features=svd.fit_transform(tfxidf_test)



from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words=stop_words)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(clean_traindata, traintarget)
multinomial_scores = text_clf.fit(clean_traindata, traintarget).predict_proba(clean_testdata)
predicted_multinomial = text_clf.predict(clean_testdata)

print (metrics.classification_report(testtarget, predicted_multinomial,
    target_names=['patriots','hawks']))

print(metrics.confusion_matrix(testtarget, predicted_multinomial))

fpr, tpr, thresholds = roc_curve(testtarget, multinomial_scores[:,1])
roc_auc = auc(fpr, tpr)
fig1 = plt.subplot()
plt.plot(fpr, tpr,label='ROC Curve of Linear SVM Classifier (area = {0:0.2f})'
            ''.format(roc_auc),lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.legend(loc="lower right",prop={'size':8})