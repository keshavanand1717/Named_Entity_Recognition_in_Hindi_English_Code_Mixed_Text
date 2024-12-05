# pip install spacy
# python -m spacy download en_core_web_sm


import pandas as pd
import spacy
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Define the SentenceGetter class to process data
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

# Load the annotated data (replace with your actual file path)
data = pd.read_csv("annotatedData.csv", encoding="latin1")
data = data.ffill()  # Forward fill any missing data

getter = SentenceGetter(data)
sentences = getter.sentences

# Feature extraction function that includes POS tags
def word2features(sent, i):
    word = sent[i][0]
    
    # Process sentence with spaCy for POS tagging
    doc = nlp(" ".join([w[0] for w in sent]))  # Parse the sentence with spaCy
    pos_tag = doc[i].pos_  # Get POS tag for the word
    
    # Build the feature dictionary
    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],  # Last 3 characters
        'word[-2:]': word[-2:],  # Last 2 characters
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.containsDigit()': any(c.isdigit() for c in word),
        'word.containsSpecialChar()': any(not c.isalnum() for c in word),
        'word.prefix(3)': word[:3],  # 3-character prefix
        'word.suffix(3)': word[-3:],  # 3-character suffix
        'word.pos': pos_tag,  # Add POS tag as a feature
    }
    
    # If it's not the first word, add features for the previous word
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.prefix(3)': word1[:3],  # Prefix of previous word
            '-1:word.suffix(3)': word1[-3:],  # Suffix of previous word
            '-1:word.pos': nlp(" ".join([w[0] for w in sent]))[i-1].pos_,  # POS of previous word
        })
    else:
        features['BOS'] = True  # Beginning of sentence flag

    # If it's not the last word, add features for the next word
    if i < len(sent) - 1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.prefix(3)': word1[:3],  # Prefix of next word
            '+1:word.suffix(3)': word1[-3:],  # Suffix of next word
            '+1:word.pos': nlp(" ".join([w[0] for w in sent]))[i+1].pos_,  # POS of next word
        })
    else:
        features['EOS'] = True  # End of sentence flag

    return features

# Convert a sentence into feature vectors
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Convert a sentence into labels
def sent2labels(sent):
    return [label for token, label in sent]

# Convert a sentence into tokens (words)
def sent2tokens(sent):
    return [token for token, label in sent]

# Prepare features and labels for training
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the CRF model
crf = CRF(algorithm='lbfgs', c2=0.1, max_iterations=100, all_possible_transitions=True)

# Fit the CRF model on the training data
crf.fit(X_train, y_train)

# Predict on the test set
y_pred = crf.predict(X_test)

# Print classification report
report = flat_classification_report(y_pred=y_pred, y_true=y_test, digits=6)
print(report)

