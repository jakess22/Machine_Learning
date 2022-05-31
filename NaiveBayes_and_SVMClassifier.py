import pandas as pd
import math
import sys
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import classification_report
from sklearn.naive_bayes import CategoricalNB, BernoulliNB, MultinomialNB, GaussianNB
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import svm


# Install a pip package if not already installed
!{sys.executable} -m pip install imbalanced-learn delayed


# If put the data(.csv) under the same folder, you could use
df = pd.read_csv('./garments_worker_productivity.csv')
df = df.drop(columns = ['date'])

print("All unique categorial attributes:")      
for column in df[['day','quarter', 'department', 'team']]:
    print(df[column].unique())
print("\nBased on unique values list, only Department values have typos.")

df['department'] = df['department'].replace(['sweing'], ['sewing'])
print("\nFixed typos:")
for column in df[['day','quarter', 'department', 'team']]:
    print(df[column].unique())

df.loc[df["actual_productivity"] >= df["targeted_productivity"], "satisfied"] = 1
df.loc[df["actual_productivity"] < df["targeted_productivity"], "satisfied"] = 0

    


df = df.drop(columns = ['actual_productivity', 'targeted_productivity'])

print("Checking for any empty values")
print(df.isna().any())
print("\nWip column is the only column with empty values, adjusting now.")

df['wip'] = df['wip'].fillna(0)
print(df.isna().any())

# Remember to continue the task with your processed data from Exercise 1
#['day','quarter', 'department', 'team']


for column in df[['day','quarter', 'department', 'team']]:
    print(df[column].unique())
encoder = OrdinalEncoder()

df2 = pd.DataFrame(encoder.fit_transform(df), columns = df.columns)
print("After encoding:")
for column in df[['day','quarter', 'department', 'team']]:
    print(df2[column].unique())

inputs = df2.drop('satisfied', axis = 'columns')
output = df2.satisfied
X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size = .2, random_state = 5)

# Naive Bayes Classifier - Categorical attributes
X_train_cat = X_train[['day','quarter', 'department', 'team']]
X_test_cat = X_test[['day','quarter', 'department', 'team']]

scaler = StandardScaler()
NaiveBayes = GaussianNB()
scaler.fit(X_train_cat)
NaiveBayes.fit(scaler.transform(X_train_cat), np.asarray(y_train))

print(classification_report(y_train, NaiveBayes.predict(scaler.transform(X_train_cat))))
print(classification_report(y_test, NaiveBayes.predict(scaler.transform(X_test_cat))))




# Naive Bayes Classifier - Numerical attributes
X_train_num = X_train.drop(columns = ['day','quarter', 'department', 'team'])
X_test_num = X_test.drop(columns = ['day','quarter', 'department', 'team'])

scaler = StandardScaler()
NaiveBayes = GaussianNB()
scaler.fit(X_train_num)
NaiveBayes.fit(scaler.transform(X_train_num), np.asarray(y_train))

print(classification_report(y_train, NaiveBayes.predict(scaler.transform(X_train_num))))
print(classification_report(y_test, NaiveBayes.predict(scaler.transform(X_test_num))))

# SVM Classifier - Linear kernel
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

svm = svm.SVC(kernel = 'linear')
svm.fit(scaler.transform(X_train), y_train)

print(classification_report(y_test, svm.predict(scaler.transform(X_test))))

# SVM Classifier - RBF kernel
svm = svm.SVC(kernel = 'rbf')
svm.fit(scaler.transform(X_train), y_train)

print(classification_report(y_test, svm.predict(scaler.transform(X_test))))

