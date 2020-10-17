#!/usr/bin/env python
# coding: utf-8

# # SIT 720 - Machine Learning
# 
# Lecturer: Chandan Karmakar | karmakar@deakin.edu.au
# 
# School of Information Technology,
# <br/>Deakin University, VIC 3125, Australia.

# ## Assessment Task 5 (35 marks)
# 
# In this assignment, you will use a lot of concepts learnt in this course to come up with a good solution for a given chronic kidney disease prediction problem.
# 
# ## Submission Instruction
# 1.  Student should insert Python code or text responses into the cell followed by the question.
# 
# 2.  For answers regarding discussion or explanation, **maximum ten sentences are suggested**.
# 
# 3.  Rename this notebook file appending your student ID. For example, for student ID 1234, the submitted file name should be A5_1234.ipynb.
# 
# 4.  Insert your student ID and name in the following cell.

# In[ ]:


# Student ID: 218599279

# Student name: Edwin John Nadarajan


# ##The dataset
# 
# **Dataset file name:** chronic_kidney_disease.csv
# 
# **Attribute Information:** 
# 
# There are 24 features + class = 25 attributes
# 1.  Age(numerical): age in years
# 2.Blood Pressure (numerical): bp in mm/Hg
# 3.Specific Gravity (categorical): sg - (1.005,1.010,1.015,1.020,1.025)
# 4.Albumin (categorical): al - (0,1,2,3,4,5)
# 5.Sugar (categorical): su - (0,1,2,3,4,5)
# 6.Red Blood Cells (categorial): rbc - (0, 1)
# 7.Pus Cell (categorical): pc - (0, 1)
# 8.Pus Cell clumps (categorical): pcc - (0, 1)
# 9.Bacteria (categorical): ba - (0, 1)
# 10.Blood Glucose Random (numerical): bgr in mgs/dl
# 11.Blood Urea (numerical): bu in mgs/dl
# 12.Serum Creatinine (numerical): sc in mgs/dl
# 13.Sodium (numerical): sod in mEq/L
# 14.Potassium (numerical): pot in mEq/L
# 15.Hemoglobin (numerical): hemo in gms
# 16.Packed Cell Volume (numerical)
# 17.White Blood Cell Count (numerical): wc in cells/cumm
# 18.Red Blood Cell Count (numerical): rc in millions/cmm
# 19.Hypertension (categorical): htn - (0, 1)
# 20.Diabetes Mellitus (categorical): dm - (0, 1)
# 21.Coronary Artery Disease (categorical): cad - (0, 1)
# 22.Appetite (categorical): appet - (0, 1)
# 23.Pedal Edema (categorical): pe - (0, 1)
# 24.Anemia (categorical): ane - (0, 1)
# 25.Class (categorical): class - (ckd, notckd)
# 

# ## Part 1: Short questions: **(6 marks)**
# 
# 
# 

# ![image.png](data:image/png;base64,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)

# 
# 1.  For the above figure, what value of k in KNN method will give the best accuracy for leave-one-out cross-validation. Report accuracy and k value. **(3 marks)**

# In[ ]:


# CODE and/or COMMENT
# k = 2 should give the best accuracy fro leave-one-out cross-validation method.
# accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives) 
# accuracy = (14+14)/(14+4+14+4) = 28/36 = 0.777


# 2.  In classification, overfitting and underfitting is a big problem. Does it happen in Random Forest or not? Why? **(3 marks)**

# In[ ]:


# CODE and/or COMMENT
"""
A single decision tree can easily overfit due to noise in the data. Random Forest is an ensemble of decision tree.
Therefore, a Random Forest with only one tree will overfit. But as the number of trees increases, overfitting decreases.
The more trees in Random Forest, the better. Overfitting can be prevented by increasing min_samples_leaf parameter (but not too much).
As for underfitting, yes, that too can happen depending on how the hyperparameters are set. For eg: increasing min_samples_split
hyperparameter too much can cause underfitting. Also, if the value of the max_leaf_nodes is very small, the random forest
is likely to underfit. 
"""


# ## Part 2: **(24 marks = 4 methods x 6)**
# 
# Using the following four supervised machine learning methods, answer questions(A-D).
# 1.   Support vector machine
# 2.   K-Nearest Neighbour
# 3.   Decision tree, and
# 4.   Random forest
# 
# **A.**  Build optimised classification model to predict the chronic kidney disease from the dataset. **(1 marks)**
# 
# **B.** For each optimised model, answer the followings -  **(3 marks)**
# 
# *  which hyperparameters were optimised? [Hint: For SVM, kernel can be considered as one of the hyperparameters.]
# 
# *  what set or range of values were used for each hyperparameter?
# 
# *  which metric was used to measure the performance?
# 
# *  justify your design decisions.
# 
# **C.**  Plot the prediction performance against hyperparameter values to visualise the optimisation process and mark the optimal value. **(1 marks)**
# 
# **D.** Evaluate the model (obtained from A) performance on the test set. Report the confusion matrix, F1-score and accuracy. **(1 marks)**

# In[45]:


# CODE and/or COMMENT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('chronic_kidney_disease.csv')
#cleaning
df = df[df!='?']
df = df.dropna()

#preprocessing
y = df['class']
X = df.drop(columns=['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

#standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(y.value_counts())
#I'm using accuracy as the main performance metric as this dataset is relatively balanced 
#(the number of not_ckd is close to ckd) as seen in y.value_counts()
#if it was imbalanced, I would have used F1 score or  sensitivity/recall

#Using SVM for prediction
print('SVM:\n')
#optimising SVM model based on kernel. Using these kernels:
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best = kernels[0]
accs = []
highest=0
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = kernel
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph
plt.plot(kernels,accs)
plt.xlabel('kernel')
plt.ylabel('Performance')
plt.show()
print('\nBest kernel hyperparameter in SVM is: '+best+'\n\n')
#optimising SVM model based on C. Using this range:
Cs = np.arange(1,10)
best = 1
accs = []
highest=0
for i in Cs:
    clf = svm.SVC(C=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = i
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph
plt.plot(Cs,accs)
plt.xlabel('C value')
plt.ylabel('Performance')
plt.show()
print('\nBest C value hyperparameter in SVM is: '+str(best)+'\n\n')

#Using KNN for classification
print('KNN:\n')
#Optimizing the model based on k. Using the range 1 to 10
best = 1
highest=0
accs = []
ks = np.arange(1,10)
for k in ks:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>=highest):
        best = k
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph
plt.plot(ks,accs)
plt.xlabel('k value')
plt.ylabel('Performance')
plt.show()
print('\nBest k value hyperparameter in KNN is: '+str(best)+'\n\n')

#Optimizing the model based on algorithm. Using these:
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
best = algorithms[0]
highest=0
accs = []
for algo in algorithms:
    classifier = KNeighborsClassifier(algorithm=algo)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = algo
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph    
plt.plot(algorithms,accs)
plt.xlabel('Algorithm')
plt.ylabel('Performance')
plt.show()
print('\nBest algorithm hyperparameter in KNN is: '+str(best)+'\n\n')

#Using Decision Tree for classification
print('Decision Tree:\n')
#Using min_samples_split hyperparameter for optimization with a range of 2 to 11 because minimum value must be 2
mss = np.arange(2,11)
highest = 0
accs = []
for j in mss:
    clf = DecisionTreeClassifier(min_samples_split=j)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = j
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph    
plt.plot(mss,accs)
plt.xlabel('min_samples_split')
plt.ylabel('Performance')
plt.show()
print('\nBest value of min_samples_split hyperparameters in Decision Tree is: '+str(best)+'\n\n')

#Using min_samples_leaf hyperparameter for optimization with a range of 1 to 10
msl = np.arange(1,10)
highest = 0
accs = []
for j in mss:
    clf = DecisionTreeClassifier(min_samples_leaf=j)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = j
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph    
plt.plot(msl,accs)
plt.xlabel('min_samples_leaf')
plt.ylabel('Performance')
plt.show()
print('\nBest value of min_samples_leaf hyperparameters in Decision Tree is: '+str(best)+'\n\n')

#Using Random Forest for classification
print('Random Forest:\n')
#n_estimators ie the number of trees in the forest is selected as the hyperparameter for optimization ranging from 1 to 201
estimators = np.arange(1,201,20)
highest = 0
accs = []
for i in estimators:
    clf=RandomForestClassifier(n_estimators=i)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    cc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = i
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph    
plt.plot(estimators,accs)
plt.xlabel('Estimators')
plt.ylabel('Performance')
plt.show()
print('\nBest value of n_estimators hyperparameters in Decision Tree is: '+str(best)+'\n\n')

#Using min_samples_split hyperparameter for optimization in range 2 to 11 because minimum value cannot be less than 2
mss = np.arange(2,11)
highest = 0
accs = []
for i in mss:
    clf=RandomForestClassifier(min_samples_split=i)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #performance evaluation using accuracy because the dataset is balanced
    cc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    #saving the top accuracy score
    if(acc>highest):
        best = i
        highest = acc
    print('Accuracy: '+str(acc))
#plotting on graph    
plt.plot(mss,accs)
plt.xlabel('min_samples_split')
plt.ylabel('Performance')
plt.show()
print('\nBest value of min_samples_split hyperparameters in Decision Tree is: '+str(best)+'\n\n')


# In[56]:


#Using linear kernel and C value 1 for optimizing SVM model
print('\nSVM\n')
clf = svm.SVC(kernel='linear',C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#performance evaluation
acc = accuracy_score(y_test, y_pred)
print('Accuracy: '+str(acc))
print('Confusion Matrix:\n'+str(confusion_matrix(y_test, y_pred)))
print('Classification Report:\n'+str(classification_report(y_test, y_pred)))

#Using k=4 neighbours and algorith='auto' for optimizing KNN model
print('\nKNN\n')
clf = KNeighborsClassifier(n_neighbors=4,algorithm='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#performance evaluation
acc = accuracy_score(y_test, y_pred)
print('Accuracy: '+str(acc))
print('Confusion Matrix:\n'+str(confusion_matrix(y_test, y_pred)))
print('Classification Report:\n'+str(classification_report(y_test, y_pred)))

#Using min_samples_split = 2 and min_samples_leaf = 3 hyperparameter for optimization of Decision Tree model
print('\nDecision Tree\n')
clf = DecisionTreeClassifier(min_samples_split = 2,min_samples_leaf=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#performance evaluation
acc = accuracy_score(y_test, y_pred)
print('Accuracy: '+str(acc))
print('Confusion Matrix:\n'+str(confusion_matrix(y_test, y_pred)))
print('Classification Report:\n'+str(classification_report(y_test, y_pred)))

#Using n_estimators=1 and min_samples_split = 2 hyperparameter for optimization of Random Forest model
print('\nRandom Forest\n')
clf = RandomForestClassifier(n_estimators=1,min_samples_split=2)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#performance evaluation
acc = accuracy_score(y_test, y_pred)
print('Accuracy: '+str(acc))
print('Confusion Matrix:\n'+str(confusion_matrix(y_test, y_pred)))
print('Classification Report:\n'+str(classification_report(y_test, y_pred)))


# ## Part 3: Discussion **(5 marks)**
# 
# Based on the results obtained in Part-2, which classification method showed the best performance and why? Do you have any suggestions to further improve the model performances? **(5 marks)**

# In[ ]:


# CODE and/or COMMENT
"""
Based on the results obtained from Part-2, decision tree with min_samples_split of 2 and min_samples_leaf of 3 showed
the best performance. 
By increasing the value of the min_sample_split, we can reduce the number of splits that happen in the decision tree 
and therefore prevent the model from overfitting.
According to the graph of min_samples_leaf vs performance, we can clearly see that after a particular threshold, the
performance drops with the increase of min_samples_leaf. Apart from that, keeping this parameter value too low may result in
overfitting sometimes. Therefore the shown value of 3 is optimal.

One way to improve the performance of the model would be to optimize max_depth parameter. Using the max_depth parameter, 
what depth is wanted for the decision tree's growth can be set. Increasing max_depth might increase performance but we need
to be careful because after a certain threshold, overfitting will begin and test performance will drop.
"""

