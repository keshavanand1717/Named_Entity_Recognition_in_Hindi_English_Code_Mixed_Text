import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report



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
for i in range(1):
    for j in range(1):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1*(i+1),
            c2=0.1*(j+1),
            max_iterations=100,
            all_possible_transitions=False,
        )

        # Train CRF on training set
        crf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = crf.predict(X_test)

        # Generate Classification Report
        print("c1 c2 :" , (i+1)*0.1, 0.1*(j+1))
        report = flat_classification_report(y_true=y_test, y_pred=y_pred, digits=6)
        print(report)
