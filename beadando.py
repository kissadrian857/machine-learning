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
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve #  importing performance metrics

data = pd.read_csv('https://raw.githubusercontent.com/kissadrian857/machine-learning/master/divorce_data.csv', delimiter=';')

y = data['Divorce'].copy() #the actual results
X = data.drop('Divorce', axis=1).copy() #data without the results
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=2021)
target_names = ["divorced","married"]

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);   # spam prediction for train
accuracy_logreg_train = logreg_classifier.score(X_train,y_train);
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction for test
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities
accuracy_logreg_test = logreg_classifier.score(X_test,y_test);

# Plotting non-normalized confusion matrix
plot_confusion_matrix(logreg_classifier, X_train, y_train, display_labels = target_names);
plot_confusion_matrix(logreg_classifier, X_test, y_test, display_labels = target_names);

# Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
score_train_naive_bayes = naive_bayes_classifier.score(X_train,y_train);  #  goodness of fit
score_test_naive_bayes = naive_bayes_classifier.score(X_test,y_test);  #  goodness of fit
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities









