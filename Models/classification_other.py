import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sent").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None



data = pd.read_csv("annotatedData.csv", encoding="latin1")
data = data.ffill()

getter = SentenceGetter(data)
sentences = getter.sentences

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.startsWith#()': word.startswith("#"),
        'word.startsWith@()': word.startswith("@"),
        'word.1stUpper()': word[0].isupper(),
        'word.isAlpha()': word.isalpha(),
    }
    # # MORE FEATURES - Very less effect on final accuracy
    # for j in range(1, len(word) + 1):
    #     prefix = word[:j]
    #     features[f'prefix_{j}'] = prefix
    # for j in range(0, len(word) ):
    #     suffix = word[j:]
    #     features[f'suffix_{j}'] = suffix
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.startsWith#()': word1.startswith("#"),
            '-1:word.startsWith@()': word1.startswith("@"),
            '-1:word.1stUpper()': word1[0].isupper(),
            '-1:word.isAlpha()': word1.isalpha(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.startsWith#()': word1.startswith("#"),
            '+1:word.startsWith@()': word1.startswith("@"),
            '+1:word.1stUpper()': word1[0].isupper(),
            '+1:word.isAlpha()': word1.isalpha(),
        })
    else:
        features['EOS'] = True

    return features


def prepare_data(sentences):
    X = []  # List of feature dictionaries for each word in all sentences
    y = []  # List of tags for each word in all sentences
    
    for sent in sentences:
        X_sent = [word2features(sent, i) for i in range(len(sent))]
        y_sent = [tag for word, tag in sent]
        X.extend(X_sent)
        y.extend(y_sent)
    
    return X, y

# Prepare X and y using the function
X, y = prepare_data(sentences)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Use DictVectorizer to convert feature dictionaries into numerical features
vectorizer = DictVectorizer(sparse=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Initialize and train the DecisionTreeClassifier
classifiers = {
    "Naive Bayes (MultinomialNB)": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVC (Support Vector Classifier)": SVC(random_state=42),
    "K-Neighbors Classifier": KNeighborsClassifier()
}

# Step 4: Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    print(f"Training and evaluating {clf_name}...")
    
    # Train the classifier
    clf.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_vec)
    
    # Print classification report
    print(f"\n{clf_name} Classification Report:\n")
    print(classification_report(y_test, y_pred,digits=6))
    print("="*80)  # Divider between classifiers

