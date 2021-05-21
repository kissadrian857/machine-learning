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
from sklearn.cluster import KMeans;  # importing clustering algorithms
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib

data = pd.read_csv('https://raw.githubusercontent.com/kissadrian857/machine-learning/master/divorce_data.csv', delimiter=';')

y = data['Divorce'].copy() #the actual results
X = data.drop('Divorce', axis=1).copy()
X = X.to_numpy() #data without the results
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=2021)
target_names = ["married","divorced"]
colors = ['blue','red']

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
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);  # spam prediction for train
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities

# Plotting non-normalized confusion matrix
plot_confusion_matrix(naive_bayes_classifier, X_train, y_train, display_labels = target_names);
plot_confusion_matrix(naive_bayes_classifier, X_test, y_test, display_labels = target_names); 

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, yprobab_naive_bayes[:,1]);
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(5);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate'); 
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

# Full PCA using scikit-learn
pca = PCA();
pca.fit(X);

# Visualizing the variance ratio which measures the importance of PCs
fig = plt.figure(6);
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 

# Visualizing the cumulative ratio which measures the impact of first n PCs
fig = plt.figure(7);
plt.title('Cumulative explained variance ratio plot');
cum_var_ratio = np.cumsum(var_ratio);
x_pos = np.arange(len(cum_var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,cum_var_ratio, align='center', alpha=0.5);
plt.show(); 

# PCA with limited components
pca = PCA(n_components=2);
pca.fit(X);
X_pc = pca.transform(X);
class_mean = np.zeros((2,54));
for i in range(2):
    class_ind = [y==i][0].astype(int);
    class_mean[i,:] = np.average(X, axis=0, weights=class_ind);
PC_class_mean = pca.transform(class_mean);    
full_mean = np.reshape(pca.mean_,(1,54));
PC_mean = pca.transform(full_mean);

fig = plt.figure(8);
plt.title('Dimension reduction of the Iris data by PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(X_pc[:,0], X_pc[:,1],s=50,c=y,
            cmap=col.ListedColormap(colors),label='Datapoints');
plt.scatter(PC_class_mean[:,0],PC_class_mean[:,1],s=50,marker='P',
            c=np.arange(2),cmap=col.ListedColormap(colors),label='Class means');
plt.scatter(PC_mean[:,0],PC_mean[:,1],s=50,c='black',marker='X',label='Overall mean');
plt.legend();
plt.show();

# Default parameters
K = 2;

# # K-means clustering with fix K
kmeans_cluster = KMeans(n_clusters=K, random_state=2021);
kmeans_cluster.fit(X);   #  fiting cluster model for X
y_pred = kmeans_cluster.predict(X);   #  predicting cluster label
sse = kmeans_cluster.inertia_;   # sum of squares of error (within sum of squares)
centers = kmeans_cluster.cluster_centers_;  # centroid of clusters

# # Davies-Bouldin goodness-of-fit
# DB = davies_bouldin_score(X,y_pred);  

# # Printing the results
# print(f'Number of cluster: {K}');
# print(f'Within SSE: {sse}');
# print(f'Davies-Bouldin index: {DB}');

# # Visualizing of datapoints with cluster labels and centroids
# fig = plt.figure(10);
# plt.title('Scatterplot of datapoints with clusters');
# plt.xlabel('X1');
# plt.ylabel('X2');
# plt.scatter(X[:,0],X[:,1],s=50,c=y_pred);   #  dataponts with cluster label
# plt.scatter(centers[:,0],centers[:,1],s=50,c='red');  #  centroids
# plt.show();






