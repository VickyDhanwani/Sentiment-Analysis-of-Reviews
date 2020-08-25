# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:26:12 2020

@author: 91735
"""


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
def pre_process(text) :
    text = text.lower()
    text = re.sub("(\\d|\\W)+", " ", text)
    return text

data = pd.read_csv("AllProductReviews.csv", sep = ",")

stopwords = ["THE", "FOR", "A", "AN", "IN", "OURSELVES", "OUR", "OURSELF", 
            "YOU", "YOUR", "YOURS" , "YOURSELF", "YOURSELVES","ME", "MY", "MINE", "BUT", "AGAIN",
            "IT", "ITS", "THEY", "THEIR", "THEIRS", "AS", "I", "AM", "IS", "ARE", "THEM", "DO",
            "DOING", "OF", "HE", "SHE", "HIM", "HER", "HERSELF", "HIMSELF", "THROUGH", "AND", "BEEN",
            "HAVE", "HAVING", "HAS", "WILL", "AND", "DID", "WITH", "TO", "UP" ]

data["CompleteReview"] = data["ReviewTitle"].str.cat(data["ReviewBody"], sep = " ")
print(data["CompleteReview"])
cols = ["ReviewTitle", "ReviewBody"]
data = data[data["ReviewTitle"].isnull() == False]
data = data[data["ReviewBody"].isnull() == False]
print(data.ReviewTitle.shape)
X = data["CompleteReview"]
Y = data["ReviewStar"]
print(X.shape, Y.shape)

X_train = X
Y_train = Y
trainf = X_train.tolist()
for i in range(0, len(trainf)):
    trainf[i] = pre_process(trainf[i]) 

X_train = pd.Series(trainf)
"""testf = X_test.tolist()
for i in range(0, len(testf)):
    testf[i] = pre_process(testf[i])

X_test = pd.Series(testf)
"""
nb = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4), stop_words = stopwords)),
               ('tfidf', TfidfTransformer()),
               ('clf', LinearSVC()),
    ])
nb.fit(X_train, Y_train)
#joblib.dump(nb, "SVM.pkl")
X_test = ["Sound is good , these earphone are good except one thing: the earphones are really very heavy and you cannot put it into your ears as it will fall down. That is the main reason that I have to return this product."]
print(nb.predict(X_test))
