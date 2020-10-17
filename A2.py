#!/usr/bin/env python
# coding: utf-8

# #**SIT 720 - Machine Learning**
# 
# Lecturer: Chandan Karmakar | karmakar@deakin.edu.au
# 
# School of Information Technology,
# <br/>Deakin University, VIC 3125, Australia.

# #**Assessment Task 2 (30 marks)**
# 
# ##**Submission Instruction**
# 1.  Student should insert Python code or text responses into the cell followed by the question.
# 
# 2.  For answers regarding discussion or explanation, **maximum five sentences are suggested**.
# 
# 3.  Rename this notebook file appending your student ID. For example, for student ID 1234, the submitted file name should be A2_1234.ipynb.
# 
# 4.  Insert your student ID and name in the following cell.

# In[ ]:


# Student ID: 218599279

# Student name: Edwin John Nadarajan


# ## Part 1: Clustering *(15 marks)*
# 
# Let's assume you want to design an environment to predict a class/category from a dataset based on specific features of that class. However, all the features are not strong enough or in other words features not that much variance/uniqueness across the classes. So, you have to design a clustering model by answering the following questions:
# 
# 1. Download the attached clustering.csv file. Read the file and separate the class and feature matrix. __(2 marks)__

# In[23]:


# INSERT your code (or comment) here
import pandas as pd
import math


df = pd.read_csv('clustering.csv')

#In this datatset, all the columns which aren't the Class are the features.
featuresMatrix = df[['height','length','width','std','min','max','kurtosis']]
classMatrix = df[['Class']]

print("Features Matrix:\n")
print(featuresMatrix)
print("Class Matrix:\n")
print(classMatrix)


# 2. Determine the number of clusters from the dataset. Is this same as the actual number of classes in the dataset? __(1 marks)__
# 
# <!-- Choose the best three features using different selection criteria (ANOVA, Chi-squared) based on the purity score for the k-mean cluster (Euclidean distance matrix). Which one is good and why? __(5 marks)__ -->

# In[25]:


# INSERT your code (or comment) here
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np

X = featuresMatrix  

plt.plot()

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#As seen in the elbow graph, the graph forms an elbow at k=3. 
#Therefore, the optimal number of clusters for this dataset is 3 which is also the number of classes in this dataset.


# 3. Perform K-Means clustering on the complete dataset and report purity score. __(2 marks)__ 

# In[129]:


# INSERT your code (or comment) here
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

#Standardizing the data
X = StandardScaler().fit_transform(df)

#KMeans with k=3 as derived from previous question
kmeans = KMeans(n_clusters=3).fit(X)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["g.","r.","c."]

for i in range(len(X)):
    #print(labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

# Visualize the centroids
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 50, zorder = 10)
plt.show()

#taking classMatrix ('Class' column) as true values and labels from kmeans as predicted values
purity = purity_score(classMatrix, labels);
print("Purity score of kmeans clustering on the given dataset is: "+str(purity))


# 4. There are several distance metrics for  K-Means such as euclidean, squared euclidean, Manhattan, Chebyshev, Minkowski. [ __Hints:__ See the pyclustering library for python.]
#     - Your job is to compare the purity score of k-means clustering for different distance metrics. __(5 marks)__ 
#     - Select the best distance metric and explain why this distance metric is best for the given dataset. __(2 marks)__ 

# In[71]:


# INSERT your code (or comment) here
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from scipy.spatial import distance

k = 3
data = X

def squared_euclidean_distance(u, v):
    diff = u - v
    return np.dot(diff, diff)
def manhattan_distance(u, v):
    diff = u - v
    return np.sum(np.abs(diff))
def minkowski3_distance(u, v):
    #order set to 3.0 in this example
    return distance.minkowski(u, v, 3.0)

#Storing the best distances along with their purity results
best = ''
highest = 0

kclusterer = KMeansClusterer(k, distance=nltk.cluster.util.euclidean_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
purity = purity_score(classMatrix, assigned_clusters);
if highest<purity:
    highest = purity
    best = 'euclidean distance'
print("Purity score of kmeans clustering based on euclidean distance on the given dataset is: "+str(purity))

kclusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
purity = purity_score(classMatrix, assigned_clusters);
if highest<purity:
    highest = purity
    best = 'cosine distance'
print("Purity score of kmeans clustering based on cosine distance on the given dataset is: "+str(purity))

kclusterer = KMeansClusterer(k, distance=squared_euclidean_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
purity = purity_score(classMatrix, assigned_clusters);
if highest<purity:
    highest = purity
    best = 'squared euclidean distance'
print("Purity score of kmeans clustering based on squared euclidean distance on the given dataset is: "+str(purity))

kclusterer = KMeansClusterer(k, distance=manhattan_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
purity = purity_score(classMatrix, assigned_clusters);
if highest<purity:
    highest = purity
    best = 'manhattan distance'
print("Purity score of kmeans clustering based on manhattan distance on the given dataset is: "+str(purity))

kclusterer = KMeansClusterer(k, distance=distance.chebyshev, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
purity = purity_score(classMatrix, assigned_clusters);
if highest<purity:
    highest = purity
    best = 'chebyshev distance'
print("Purity score of kmeans clustering based on chebyshev distance on the given dataset is: "+str(purity))

kclusterer = KMeansClusterer(k, distance=minkowski3_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
purity = purity_score(classMatrix, assigned_clusters);
if highest<purity:
    highest = purity
    best = 'minkowski distance of order 3.0'
print("Purity score of kmeans clustering based on minkowski distance of order 3.0 on the given dataset is: "+str(purity))
print('\n')
print('Highest purity in kmeans is achieved using '+best+' with purity of '+str(highest)+' which makes it the best distance metric for the given dataset')


# 5. Use selection criteria (ANOVA, Chi-squared) to select best three features and use them for K-Means clustering. Based on the purity score which feature set are you going to recommend and why? __(3 marks)__

# In[130]:


# INSERT your code (or comment) here

#I'm using ANOVA to select the best three features
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import column_or_1d
from matplotlib import pyplot

# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
X = featuresMatrix
y = column_or_1d(classMatrix, warn=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

featureScores = {}
#scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    featureScores[fs.scores_[i]] = i 
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#selecting the best 3 features
fs.scores_ = np.sort(fs.scores_)[::-1]
best = []
for i in range(3):
    best.append(featureScores[fs.scores_[i]])
features = []
for col_name in X.columns: 
    features.append(col_name)
bestX = featuresMatrix[[features[best[0]],features[best[1]],features[best[2]]]]

#kmeans with the best 3 features
kmeans = KMeans(n_clusters=3).fit(bestX)
kmeans.fit(bestX)
labels = kmeans.labels_
purity = purity_score(classMatrix, labels);
print('Purity score of the best 3 features based on ANOVA is: '+str(purity))

# Since the purity score of even the best 3 feature set is lower than when using all features, 
# I'd recommend using all features unless we really need to reduce it to 3.


# # Part-2 (Dimensionality Reduction using PCA/SVD) *(15 marks)*
# 
# 1. For the dataset (clustering.csv), perform PCA.
#     - plot the captured variance with respect to increasing latent dimensionality. __(2.5 marks)__
#   
#   What is the minimum dimension that captures:
#     - at least 89% variance? __(1.5 marks)__
#     - at least 99% variance? __(1 marks)__

# In[112]:


# INSERT your code (or comment) here

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

Xnorm = scale(featuresMatrix)
pca = PCA(n_components=featuresMatrix.columns.size)
pca.fit(Xnorm)

var= pca.explained_variance_ratio_
print(var)

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)
plt.xlabel("Principal components")
plt.ylabel("Variance captured")

#As seen from the graph, the minimum dimension that captures at least 
#      89% variance is 2     (rounding off 88.98 to 89)
# and  99% variance is 4


# 2. Determine the purity of clusters formed by the number of principal components which captured 89% and 99% variances respectively. Plot a line graph of the purity scores against the captured variances. Discuss your findings. __(7 marks)__

# In[128]:


# INSERT your code (or comment) here

from sklearn.decomposition import PCA as sklearnPCA

#For PCA capturing 89% variance
pca = sklearnPCA(n_components=2) #2-dimensional PCA
pca89 = pd.DataFrame(pca.fit_transform(X_norm))

kmeans = KMeans(n_clusters=3).fit(pca89)
kmeans.fit(pca89)
labels = kmeans.labels_
purity = purity_score(classMatrix, labels);
print('Purity of clusters formed by the number of principal components which captured 89% variance is '+str(purity))

#For PCA capturing 99% variance
pca = sklearnPCA(n_components=4) #4-dimensional PCA
pca99 = pd.DataFrame(pca.fit_transform(X_norm))

kmeans = KMeans(n_clusters=3).fit(pca99)
kmeans.fit(pca99)
labels = kmeans.labels_
purity = purity_score(classMatrix, labels);
print('Purity of clusters formed by the number of principal components which captured 99% variance is '+str(purity))

#For plotting purity vs captured variance graph
pca = PCA(n_components=4)
pca.fit(Xnorm)
variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
purities = []
for i in range(1,5):
    pca = sklearnPCA(n_components=i) #i-dimensional PCA
    pca = pd.DataFrame(pca.fit_transform(X_norm))
    kmeans = KMeans(n_clusters=3).fit(pca)
    kmeans.fit(pca)
    labels = kmeans.labels_
    purity = purity_score(classMatrix, labels);
    purities.append(purity)
plt.plot(variance,purities)
plt.title('Purity vs Captured Variance Graph')
plt.xlabel('Captured Variance')
plt.ylabel('Purity')
plt.show()

#The purity vs captured variance graph shows that purity increases with the increase in captured variance.


# 3. Let's assume you have two datasets one is linear and another is curved structural data.
#     - Can we apply PCA on these datasets? Justify your answer. __(3 marks)__ 

# In[ ]:


# INSERT your code (or comment) here
"""
    Yes, we can apply PCA on both datasets however, the results of PCA on curved structural data would be meaningless as
    PCA assumes linear correlations. PCA simply rotates the given coordinate axes which is a linear operation and it 
    optimizes the variance of the given data in the least number of dimensions. Its sensitivity is limited to variance and 
    therefore it doesn't pick out any structure.
"""

