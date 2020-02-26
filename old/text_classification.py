import tensorflow as tf
import numpy as np
from tensorflow import keras

print(np.__version__)

data = keras.datasets.imdb

# take only the 10.000 more frequents words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# print(train_data[0])  # ogni recensione e formata da numeri dove ogni numero identifica una parola
# label = positive or negative review

word_index = data.get_word_index()

# metto +3 perche vanno definiti dei caratteri speciali
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# invertiamo il dict in quanto le key sono le parole e i value i numeri ma noi per fare decoding abbiamo bisogno del contrario
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# preprocessing in modo che tutte le review siano lunghe uguali
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                       maxlen=256)


# print(len(test_data[0]))
# print(len(test_data[1]))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])


# model

'''
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))  # perche abbiamo preso 10000 parole
model.add(keras.layers.GlobalAveragePooling1D())  # lowering the dimension
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# batch_size = numero di review che vediamo per volta
fit_model = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5")
'''

model = keras.models.load_model("model.h5")
