import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
import numpy as np


def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index
glove_file_path = 'glove.6B.300d.txt'  # Ensure this file is in the working directory
embeddings_index = load_glove_embeddings(glove_file_path)

# Step 3: Helper function to get the GloVe embedding for a word
def get_word_embedding(word):
    embedding = embeddings_index.get(word.lower())
    if embedding is None:
        embedding = np.zeros(300)  # Return zero vector if word not found
    return embedding

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
    word_embedding = get_word_embedding(word)
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
    for idx, value in enumerate(word_embedding):
        features[f'word.embedding_{idx}'] = value
    if i > 0:
        word1 = sent[i-1][0]
        word1_embedding = get_word_embedding(word1)
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
        for idx, value in enumerate(word1_embedding):
            features[f'-1:word.embedding_{idx}'] = value
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        word1_embedding = get_word_embedding(word1)
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
        for idx, value in enumerate(word1_embedding):
            features[f'+1:word.embedding_{idx}'] = value
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]


X = [sent2features(s) for s in sentences]

y = [sent2labels(s) for s in sentences]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=False,
)

# Train CRF on training set
crf.fit(X_train, y_train)

# Predict on the test set
y_pred = crf.predict(X_test)

# Generate Classification Report
report = flat_classification_report(y_true=y_test, y_pred=y_pred, digits=6)
print(report)
