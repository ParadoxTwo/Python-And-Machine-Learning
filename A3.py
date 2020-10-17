#!/usr/bin/env python
# coding: utf-8

# #**SIT 720 - Machine Learning**
# 
# Lecturer: Chandan Karmakar | karmakar@deakin.edu.au
# 
# School of Information Technology,
# <br/>Deakin University, VIC 3125, Australia.

# #**Assessment Task 3 (40 marks)**
# 
# ##Submission Instruction
# 1.  Student should insert Python code or text responses into the cell followed by the question.
# 
# 2.  For answers regarding discussion or explanation, **maximum five sentences are suggested**.
# 
# 3.  Rename this notebook file appending your student ID. For example, for student ID 1234, the submitted file name should be A3_1234.ipynb.
# 
# 4.  Insert your student ID and name in the following cell.

# In[ ]:


# Student ID: 218599279

# Student name: Edwin John Nadarajan


# ## Background
# 
# Environment and its changes are the most complex system. It is unarguably accepted that the temperature changes are greately affected by various environmental factors. Many of them are positively related to the  change, whereas, some have negative correlation. In this assesment task, you will analyse relationship among various environmental factors, which affect temperature.
# 
# ##The dataset
# 
# **Dataset file name:** weather_dataset.csv
# 
# **Dataset description:** The dataset contains total 10 features. Each row contains an hourly record of weather status and the data was recorded for the time period between 2006 and 2016.
# 
# **Features and labels:** 
# 
# 1.   recording_date_time (date_time): Date and time the data was recorded
# 2.   precip_type (string): Precipitation status, blank (no value) indicates unknown status
# 3.   temperature (float): Temperature in degree Celsius
# 4.   apparent_temperature (float): Feel like temperature in degree Celsius
# 5.   humidity (float): Percentage amount of water vapour in the air 
# 6.   wind_speed (float): Speed of the wind in km per hour
# 7.   wind_bearing (int): The direction of wind in degree in geo-polar co-ordinate. Value 0 means perfect east, 90 means perfect north, 180 and 270 means west and south respectively.
# 8.   visibility (float): Distance in km that is visible in naked eyes.
# 9.   cloud_cover (float): The fraction of the sky obscured by clouds. The value is 1 if the observed area is fully cloudy, 0 if no clouds and other fractional value indicates the portion of the area covered by clouds.
# 10.   pressure (float): Air pressure or atmospheric in milibars
# 

# ##**Part 1: Linear Regression:**  **(25 marks)**
# 
# 
# 1.   Load the dataset and split the data for training and testing - consider the data of last 2 years (2015 and 2016) for testing. Now exclude recording_date_time column from both training and test sets. Display the shape of training and test sets. **(3 marks)**

# In[38]:


# INSERT your code (or comment) here
import pandas as pd
import math
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("weather_dataset.csv")
df.fillna(-99999, inplace=True)
df.dropna(inplace=True)
test = len(df[df['recording_date_time'] >= '2015'])
train = len(df[df['recording_date_time'] < '2015'])
test_size = test*1.0/(test+train)

# replacing string with integer values for better fitting. snow is 0 and rain is 1.
df = df.replace(['snow','rain'],[0,1])
X = df[['recording_date_time','precip_type','apparent_temperature','humidity','wind_speed','wind_bearing','visibility','cloud_cover','pressure']].to_numpy()
y = df['temperature'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)
print("Training data holds records until 2014 end")
print(X_train)
print("Training data holds records from 2015 onwards")
print(X_test)

#removing date from the dataset
X_train = np.delete(X_train,0,axis=1) 
X_test = np.delete(X_test,0,axis=1)

print("Shape of training data: "+ str(X_train.shape))
print(X_train)
print("Shape of testing data: "+ str(X_test.shape))
print(X_test)


# 2.  Consider the 'temperature' as the target. List the insignificant features for predicting temperature, if any. Explain your findings. **(5 marks)**
# <br/><font color='green'>**[Hint for students: See the "7.3 Relevance and Covariance among features or variables" for more information.]** <font/>

# In[39]:


# INSERT your code (or comment) here

features = ['precip_type','apparent_temperature','humidity','wind_speed','wind_bearing','visibility','cloud_cover','pressure']
for i in range(0,len(features)):
    #printing the covariance matrix of each feature set with respect to the target: temperature
    print("Covariance matrix of "+features[i]+" to temperature is:")
    covMatrix = np.cov(X_train[:,i].astype(float),y_train.astype(float))
    print(covMatrix)
#when covariance of x, y
#   is greater than zero, it shows that x and y are positively correlated; if x is increasing, y is increasing.
#   is less than zero, it shows that x and y are inversely correlated; if x is increasing, y is decreasing.
#   is equal to zero, it shows that x and y are independent.
#results show either positive covariance or negative covariance for every feature set except cloud_cover. 
#this means that cloud_cover is an insignificant feature for predicting temperature and therefore we can safely remove it 
#in our future considerations.


# 3.  Now create a linear model considering the 'temperature' as the target variable and other columns as features (you can optionally remove non-contributing features). Show the test performance (as Mean Absolute Error, MAE) of the model. **(5 marks)**

# In[40]:


# INSERT your code (or comment) here
from sklearn.metrics import mean_absolute_error

#removing cloud_cover from our training and testing set.
X_train = np.delete(X_train,features.index('cloud_cover'),axis=1) 
X_test = np.delete(X_test,features.index('cloud_cover'),axis=1)
features.remove('cloud_cover')
# print(X_train)
#Creating a linear model: Linear Regression
classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

y_pred = classifier.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
print("The test performance (as Mean Absolute Error, MAE) of the model is "+str(MAE))


# 4.   Find the feature which shows maximum correlation with "pressure". Create a linear regression model to predict temperature using these two features ('pressure' and the one which shows maximum correlation). Compare the performance of this simplified model with the model developed in the previous question (Q-3). Explain the performance variation, if any. **(6 marks)**

# In[41]:


# INSERT your code (or comment) here

df = df[['precip_type','temperature','apparent_temperature','humidity','wind_speed','wind_bearing','visibility','cloud_cover','pressure']]
#shows correlation of each feature to the other
print(df.corr())
#as seen in the correlation matrix, the feature with the highest correlation to pressure is visibility
#now creating a linear regression model:

X_train,X_test,y_train,y_test = train_test_split(df[['visibility','pressure']],df[['temperature']],test_size=test_size, shuffle=False)
classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

y_pred = classifier.predict(X_test)
MAE_new = mean_absolute_error(y_test, y_pred)

print("The test performance of the new model is "+str(MAE_new)+ " and the old one is "+str(MAE))

# We know that mean absolute error basically is the absolute difference between the actual or true values and the values that are predicted 
# while the sign is removed.
# this means that lower the MAE is, better is the result.
# when we used visibility and pressure as our features for the new model, the MAE was much higher than the MAE of the previous model.
# this means that the new model has a very high error and isn't good for prediction.


# 5. Apportion the complete dataset into training and test sets, with an 40-60 split. **(6 marks)**
# 
#   (a)  Train a linear regression model without considering overfitting scenario and report the test performance. 
#   
#   (b) Create an optimal regularised linear regression model and report the test performance.
#   
#   (c) Explain the reason behind the performance variation, if any.
# 

# In[42]:


# INSERT your answer in maximum five sentences.
X = df[['precip_type','apparent_temperature','humidity','wind_speed','wind_bearing','visibility','cloud_cover','pressure']]
y = df[['temperature']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.6, shuffle=True)
classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

y_pred = classifier.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)

print("The test performance of the model is "+str(MAE))
# For a regularized linear regression model, we minimize the sum of RSS and a “penalty term” that penalizes coefficient size.
# Greater the alpha value is, higher the penalty and lesser the sum of RSS and the penalty
# LinearRegression doesn't support alpha, therefore we use RidgeRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn import metrics

#more the alpha, higher the regularization
classifier = Ridge(alpha=10, normalize=True)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

print("R-Square Value",r2_score(y_test,y_pred))
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))

#As seen in the results, for this dataset, their is no performance variation between the usual linear regression model and
# the optimal regularized linear regression model.
# Since regularization helps prevent overfitting and there's no difference in performance between regularized and non-regularized
# models, our usual linear regression is not overfitted.


# ##**Part 2: Logistic Regression:**  **(9 marks)**
# 
# 
# 1.  Can the same target (temperature, mentioned in Part-1) be used for logistic regression? Why? **(2 marks)**

# In[28]:


# INSERT your code (or comment) here

# Logistic Regression can take any real-valued number and map it into a value between 0 and 1 but never exactly at those limits.
# The value approaches but never reaches 0 or 1
# The value of temperature ranges in real numbers and therefore can't be used as target for logistic regression.
# precip_type would be the ideal target for logistic regression


# 2.  Split the dataset as 70-30% for training and testing. Create a logistic regression model to predict the 'precip_type'. Report the prediction accuracy of your model whether the "precip_type" is "rain" or not (use decision threshold of 0.45). **(5 marks)**
# 
# 

# In[76]:


# INSERT your code (or comment) here
from sklearn.linear_model import LogisticRegression
import pandas as pd
X = df[['temperature','apparent_temperature','humidity','wind_speed','wind_bearing','visibility','cloud_cover','pressure']].to_numpy()
X = preprocessing.scale(X)
y = df[['precip_type']].to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, shuffle=True)
logistic_regression= LogisticRegression(class_weight="balanced")
logistic_regression.fit(X_train,y_train)

# decision threshold set to 0.45
THRESHOLD = 0.45
y_preds = np.where(logistic_regression.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
# the default one provides better results actually:
# y_pred=logistic_regression.predict(X_test)

print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))


#  3.  Discuss the test performance using precision, recall and confusion matrix. **(2 marks)**

# In[75]:


# INSERT your code (or comment) here
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
sn.heatmap(cm, annot=True)
#The confusion matrix is a matrix which helps determine the accuracy of our prediction.
#a matrix is made with this format:
# [[True Positive,    False Negative]
#  [False Positive,   True Negative]]
# Accuracy is measured as (true positive + true negative)/(true positive + true negative + false positive + false negative)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
print("Accuracy: "+str(accuracy))

# Precision means on what proportion of all predictions that we made with our predictive model are actually true.
# Precision is measured as True Positive / True Positive + False Positive
precision = (cm[0][0])/(cm[0][0]+cm[1][0])
print("Precision: "+str(precision))
# ‘Recall’ is nothing but the measure that tells what proportion of records that actual show rain. 
# It shows how sensitive the classifier is in detecting positive instances.
# It is measured as True Positive / True Positive + False Negative
recall = (cm[0][0])/(cm[0][0]+cm[0][1])
print("Recall: "+str(recall))


# ##**Part 3: Objective function optimisation:**  **(6 marks)**

# Let’s consider the line graphs shown below and answer the following questions [Hint: See weekly content 7.4-7.10],

# <html>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b)</html>
# 
# ![image.png](data:image/png;base64,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)
# 
# 
# 

# 
# a.  Which of the above figures represents the convex objective function and why? (**1 marks**)
# 
# b.  Which hyper-parameter can help to reach the convergence point and the impact of value selection? (**2 marks**)
# 
# c.  How can we find the global minima for the objective function shown in Figure-b? _[N.B. Conceptual description will be accepted.]_ (**3 marks**)

# In[78]:


# a. When the local minima is also the global minima, the function would be a convex objective function. In the given diagrams,
# figure (a) has only one minima, which is both, the local and global minima. Therefore (a) represents convex objective function

# c. First we find all points on the function where the derivatives are 0. These points would either be minima or maxima
#    Next for each of those points, we find derivatives of values tending towards those points. If these derivatives are positive,
#    they represent a maxima and if negative, they represent minima. 
#    Finally, for all minima points, the least one would be the global minima.


# In[ ]:




