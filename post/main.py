import os
import sys
import tarfile
import time
import urllib.request
import pyprind
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gzip
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import fbeta_score, make_scorer

def readfile(gd):
    #read os.getcwd()
    num = 0
    colums = {"headline":[],
                "label":[]}
    all_data = pd.DataFrame.from_dict(colums)
    list = ['pos','eng']
    for j in list:
        local = gd+j
        FileList = os.listdir(local)
        for i in FileList:
            df = pd.read_csv(local+"/"+i)
            df = df.loc[:,['headline','label']]
            all_data = all_data.append(df)
            num+= len(df["headline"])
    all_data=all_data.sample(frac=1.0).reset_index(drop=True)
    return all_data

def preprocessor(text):
    #<[^>]*>
    text = re.sub('([a-zA-z0-9]+)|(\&+|\%+|\!+|\！＋)', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

def stopword():
    f = open("post_data/stop/stopword.txt", encoding='utf-8')
    f = f.read().splitlines()
    return f

def tokenizer(text):
    return text.split()

#data = readfile("post_data/")
data = pd.read_csv("ALL%8380.csv")
data['headline'] = data['headline'].apply(preprocessor)
#data.to_csv("ALL.csv", index=False, encoding='utf-8')
#print(data)

x_train = data.loc[:335,"headline"].values
y_train = data.loc[:335,"label"].values

x_test = data.loc[335:,"headline"].values
y_test = data.loc[335:,"label"].values
pca = PCA(n_components = 2)
stop = stopword()

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        token_pattern='\\b\\w+\\b')

#1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0,10.0,100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0,10.0,100.0]},
              ]
#('pca',pca),
lr_tfidf1 = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])
scoring = {'Accuracy':make_scorer(accuracy_score),
        'Precision': make_scorer(precision_score, average='macro'),
        'Recall': make_scorer(recall_score, average='macro'),
        'F1': make_scorer(f1_score, average='macro')}

#refit='Accuracy',
#return_train_score=True
gs_lr_tfidf_acc = GridSearchCV(lr_tfidf1, param_grid,
                        scoring=scoring,
                        cv=5,
                        verbose=2,
                        n_jobs=-1,
                        refit='Accuracy',
                        return_train_score=True
                        )

gs_lr_tfidf_acc.fit(x_train, y_train)
pd.DataFrame(gs_lr_tfidf_acc.cv_results_)
tfidf.fit_transform(x_train).toarray()
print('Best parameter set: %s ' % gs_lr_tfidf_acc.best_params_)
print('CV Accuracy: %.5f' % gs_lr_tfidf_acc.best_score_)


clf1 = gs_lr_tfidf_acc.best_estimator_
y_pred1 = clf1.predict(x_test)
y_true = y_test
print('Test Accuracy: %.5f' % clf1.score(x_test, y_test))
#print('Test Accuracy: %.5f' % clf1.score(y_true, y_pred1))

y_pred1 = clf1.predict(x_test)
y_true = y_test
print(confusion_matrix(y_true, y_pred1))
acc = accuracy_score(y_true,y_pred1)
pre = precision_score(y_true,y_pred1, average='macro')
recall = recall_score(y_true,y_pred1, average='macro')
F1 = f1_score(y_true,y_pred1, average='macro')
print('Accuracy %.6f' %acc)
print('precision %.6f' %pre)
print('recall %.6f' %recall)
print('F1 %.6f' %F1)