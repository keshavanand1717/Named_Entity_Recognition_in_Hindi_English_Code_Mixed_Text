#Baseline Models
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
X = pd.read_csv('featureVectors.csv')
y = X['word.Tag']
X.drop('word.Tag', axis=1, inplace=True)

# Handle NaN and inf values
X = X.astype('float32')
y = y.astype('float32')
X = np.nan_to_num(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Classifiers
dtc = DecisionTreeClassifier(max_depth=32, class_weight={0:1, 1:1})
gnb = GaussianNB()
clf = RandomForestClassifier(max_depth=10)

# Train and predict with Decision Tree
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
target_names = ['I-Loc', 'B-Org', 'I-Per', 'Other', 'B-Per', 'I-Org', 'B-Loc']

print("Results for Decision Tree:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=6))
score = f1_score(y_pred, y_test, average='weighted')
print("Decision Tree F1 score: {:.6f}".format(score))

# Train and predict with Naive Bayes
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Results for Naive Bayes:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=6))
score = f1_score(y_pred, y_test, average='weighted')
print("Naive Bayes F1 score: {:.6f}".format(score))

# Train and predict with Random Forest
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Results for Random Forest:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=6))
score = f1_score(y_pred, y_test, average='weighted')
print("Random Forest F1 score: {:.6f}".format(score))
