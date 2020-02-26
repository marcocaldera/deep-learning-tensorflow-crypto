import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# https://www.tensorflow.org/tutorials/keras/classification

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# labels are between 0 and 9. Below are the real class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Transform rbg in a 01 range
train_images = train_images / 255.0
test_images = test_images / 255.0

# print(train_images[7])  # img of pixel value
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # flatten the input which is a 2d array in a 1d array of 784 neurons
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")  # softmax return a probabilty distribution of the ten output neuron
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epochs = how many time the model see the data (each data is picked randomly)
model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested Acc: ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
