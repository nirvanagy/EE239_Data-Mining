# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:09:27 2016

@author: masenfrank
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

train_target_names = newsgroups_train.target_names
train_target_names = np.array(train_target_names)

train_filenames = newsgroups_train.filenames
train_filenames = np.array(train_filenames)

train_target_index = newsgroups_train.target
train_target_index = np.array(train_target_index)

#Counter(train_target_index)


fig1,ax = plt.subplots()
plt.hist(train_target_index,20,normed=1,alpha=0.75)
ax.set_xlabel('Category Index')
ax.set_ylabel('Probability')


newsgroups_test = fetch_20newsgroups(subset='test')
test_target_names = newsgroups_test.target_names
test_target_names = np.array(test_target_names)

test_filenames = newsgroups_test.filenames
test_filenames = np.array(test_filenames)

test_target_index = newsgroups_test.target
test_target_index = np.array(test_target_index)

#Counter(test_target_index)


fig2, bx = plt.subplots()
plt.hist(test_target_index,20,normed=1,alpha=0.75)
bx.set_xlabel('Category Index')
bx.set_ylabel('Probability')