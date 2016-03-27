from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



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



CompTech = np.array([0,1,2,3])
Rec = np.array([4,5,6,7])
for i in range(len(twenty_train.target)):
    if twenty_train.target[i] in CompTech:
        twenty_train.target[i] = 0
    elif twenty_train.target[i] in Rec:
        twenty_train.target[i] = 1

for i in range(len(twenty_test.target)):
    if twenty_test.target[i] in CompTech:
        twenty_test.target[i] = 0
    elif twenty_test.target[i] in Rec:
        twenty_test.target[i] = 1


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(clean_traindata, twenty_train.target)
multinomial_scores = text_clf.fit(clean_traindata, twenty_train.target).predict_proba(clean_testdata)
predicted_multinomial = text_clf.predict(clean_testdata)



import numpy as np

docs_test = clean_testdata
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target) 


print metrics.classification_report(twenty_test.target, predicted,
    target_names=['Computer Technology','Recreational Activity'])

metrics.confusion_matrix(twenty_test.target, predicted)

fig2 = plt.subplot()
fpr2, tpr2, thresholds2 = roc_curve(twenty_test.target, multinomial_scores[:,1])
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.show()

