import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from numpy.random import seed
from itertools import chain
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional
from sklearn_crfsuite.metrics import flat_classification_report

file = "annotatedData.csv"
data = pd.read_csv(file, encoding = "latin1")
data.head()
data = data.fillna(method = 'ffill')

words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)

print(f"Total number of unique words in dataset: {num_words}")

tags = list(set(data["Tag"].values))
tags.append("empty")
num_tags = len(tags)
num_tags
print("List of tags: " + ', '.join([tag for tag in tags]))
print(f"Total Number of tags {num_tags}")

class Get_sentence(object):
    def __init__(self,data):
        self.n_sent = 1
        self.data = data
        agg_func = lambda s:[(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                    s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sent").apply(agg_func)
        self.sentences = [s for s in self.grouped]

getter = Get_sentence(data)
sentence = getter.sentences
sentence[10]


plt.figure(figsize=(14,7))
plt.hist([len(s) for s in sentence],bins = 50)
plt.xlabel("Length of Sentences")
plt.show()


word_idx = {w : i + 1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(tags)}


max_len = 50
X = [[word_idx[w[0]] for w in s] for s in sentence]
X = pad_sequences(maxlen = max_len, sequences = X, padding = 'post', value = num_words - 1)

y = [[tag_idx[w[1]] for w in s] for s in sentence]
y = pad_sequences(maxlen = max_len, sequences = y, padding = 'post', value = tag_idx['empty'])
y = [to_categorical(i, num_classes = num_tags) for i in  y]


x_train,x_test,y_train,y_test = train_test_split(X, y,test_size = 0.1, random_state = 42)


input_word = Input(shape = (max_len,))
model = Embedding(input_dim = num_words, output_dim = 2*max_len, input_length = max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units = 100,return_sequences = True, recurrent_dropout = 0.1))(model)
out = TimeDistributed(Dense(num_tags,activation = 'softmax'))(model)
model = Model(input_word,out)

model.compile(optimizer = 'adamw',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()

plot_model(model, show_shapes = True)


model.fit(x_train, np.array(y_train), batch_size = 32, verbose = 1, epochs = 25, validation_split = 0.2)
y_pred = model.predict(x_test)

# Convert predictions and actual values to the original label indices
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(np.array(y_test), axis=-1)

# Convert indices to tags
idx2tag = {i: w for w, i in tag_idx.items()}
y_pred_tags = [[idx2tag[idx] for idx in row] for row in y_pred]
y_test_tags = [[idx2tag[idx] for idx in row] for row in y_test_true]


y_test_filtered = []
y_pred_filtered = []

for y_test_sentence, y_pred_sentence in zip(y_test_tags, y_pred_tags):
    filtered_test_sentence = []
    filtered_pred_sentence = []
    
    for y_test_tag, y_pred_tag in zip(y_test_sentence, y_pred_sentence):
        if y_test_tag != 'empty':
            filtered_test_sentence.append(y_test_tag)
            filtered_pred_sentence.append(y_pred_tag)
    
    y_test_filtered.append(filtered_test_sentence)
    y_pred_filtered.append(filtered_pred_sentence)

print(flat_classification_report(y_test_filtered, y_pred_filtered, digits=6))

