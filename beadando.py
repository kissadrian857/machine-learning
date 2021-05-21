# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:02:00 2021

@author: Adrián
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('https://raw.githubusercontent.com/kissadrian857/machine-learning/master/divorce_data.csv', delimiter=';')

y = data['Divorce'].copy() #the actual results
X = data.drop('Divorce', axis=1).copy() #data without the results
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=2021)

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);
score_test_logreg = logreg_classifier.score(X_test,y_test);  #  goodness of fit
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities

#print("Test Accuracy: {:.2f}%".format(score_test_logreg * 100))

# Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
score_train_naive_bayes = naive_bayes_classifier.score(X_train,y_train);  #  goodness of fit
score_test_naive_bayes = naive_bayes_classifier.score(X_test,y_test);  #  goodness of fit
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities