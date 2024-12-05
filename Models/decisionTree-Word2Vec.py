import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from gensim.models import Word2Vec

# Load your annotated dataset
data = pd.read_csv('annotatedData.csv')

# Replace NaN values in the 'Word' and 'Tag' columns
data['Word'] = data['Word'].fillna('').astype(str)
data['Tag'] = data['Tag'].fillna('Other')  # Replace NaN tags with a default category, e.g., 'Other'

# Train a Word2Vec model on the words in the dataset
sentences = data.groupby('Sent')['Word'].apply(list).tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Convert words to Word2Vec embeddings
def word_to_vec(word):
    if word in w2v_model.wv:
        return w2v_model.wv[word]
    else:
        return np.zeros(w2v_model.vector_size)

# Apply word embeddings to the dataset
X = np.array([word_to_vec(word) for word in data['Word']])

# Map tags to numerical values, and handle any NaN values in 'Tag'
y = data['Tag'].map({
    'I-Loc': 0, 'B-Org': 1, 'I-Per': 2, 'Other': 3, 'B-Per': 4, 'I-Org': 5, 'B-Loc': 6
}).fillna(3).values  # Default to 'Other' (3) if NaN remains

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Define models with best parameters for Decision Tree and Random Forest
dtc = DecisionTreeClassifier(max_depth=30, min_samples_leaf=5, min_samples_split=20, class_weight=None)
clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_leaf=2, min_samples_split=5, class_weight='balanced')

# Hyperparameter tuning for Naive Bayes
param_grid_gnb = {'var_smoothing': np.logspace(-10, -5, 10)}
gnb = GridSearchCV(GaussianNB(), param_grid_gnb, scoring='f1_weighted', cv=5)
gnb.fit(X_train, y_train)

# Print best parameters for Naive Bayes
print("Best parameters for Naive Bayes:", gnb.best_params_)

# Train and evaluate models
target_names = ['I-Loc', 'B-Org', 'I-Per', 'Other', 'B-Per', 'I-Org', 'B-Loc']

# Decision Tree
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Results for Decision Tree:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1, digits=6))
score = f1_score(y_test, y_pred, average='weighted')
print("Decision Tree F1 score: {:.6f}".format(score))

# Naive Bayes with tuned parameter
y_pred = gnb.predict(X_test)
print("Results for Naive Bayes:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1, digits=6))
score = f1_score(y_test, y_pred, average='weighted')
print("Naive Bayes F1 score: {:.6f}".format(score))

# Random Forest
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Results for Random Forest:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1, digits=6))
score = f1_score(y_test, y_pred, average='weighted')
print("Random Forest F1 score: {:.6f}".format(score))
