import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# All the images are 28x28 numpy arrays
# print(X_train[7])
# plt.imshow(X_train[7], cmap=plt.cm.binary)
# plt.show()

# Creating the model
model = keras.Sequential([
    # input layer that flatten the 28x28 picture into one array. No of neurons 28x28 = 784
    keras.layers.Flatten(input_shape=(28,28)),
    # the middle layer. No of neurons is 128
    keras.layers.Dense(128, activation="relu"),
    # the output layer. No of neurons is 10
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# For accuracy
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested acc: ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

# print(class_names[np.argmax(prediction[0])])