# Hyperparameter tuned
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

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

# Parameter grids for each classifier
param_grid_dtc = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced', None]
}

param_grid_gnb = {
    'var_smoothing': np.logspace(-9, -1, 10)
}

# Random search parameters for Random Forest
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Hyperparameter tuning for Decision Tree
dtc = DecisionTreeClassifier()
grid_search_dtc = GridSearchCV(dtc, param_grid_dtc, scoring='f1_weighted', cv=5)
grid_search_dtc.fit(X_train, y_train)
best_dtc = grid_search_dtc.best_estimator_

# Print best parameters for Decision Tree
print("Best parameters for Decision Tree:", grid_search_dtc.best_params_)

# Hyperparameter tuning for Naive Bayes
gnb = GaussianNB()
grid_search_gnb = GridSearchCV(gnb, param_grid_gnb, scoring='f1_weighted', cv=5)
grid_search_gnb.fit(X_train, y_train)
best_gnb = grid_search_gnb.best_estimator_

# Print best parameters for Naive Bayes
print("Best parameters for Naive Bayes:", grid_search_gnb.best_params_)

# Hyperparameter tuning for Random Forest with RandomizedSearchCV
clf = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(clf, param_dist_rf, n_iter=2, scoring='f1_weighted', cv=5, random_state=0)
random_search_rf.fit(X_train, y_train)
best_rf = random_search_rf.best_estimator_

# Print best parameters for Random Forest
print("Best parameters for Random Forest:", random_search_rf.best_params_)

# Predict and print results
target_names = ['I-Loc', 'B-Org', 'I-Per', 'Other', 'B-Per', 'I-Org', 'B-Loc']

# Decision Tree
y_pred = best_dtc.predict(X_test)
print("Results for Decision Tree:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1, digits=6))
score = f1_score(y_pred, y_test, average='weighted')
print("Decision Tree F1 score: {:.6f}".format(score))

# Naive Bayes
y_pred = best_gnb.predict(X_test)
print("Results for Naive Bayes:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1, digits=6))
score = f1_score(y_pred, y_test, average='weighted')
print("Naive Bayes F1 score: {:.6f}".format(score))

# Random Forest
y_pred = best_rf.predict(X_test)
print("Results for Random Forest:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1, digits=6))
score = f1_score(y_pred, y_test, average='weighted')
print("Random Forest F1 score: {:.6f}".format(score))
