import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0


# Convert labels to categorical one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Define the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 classes (0-9 digits)


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# Output: 
# Epoch 1/5
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2327 - accuracy: 0.9316
# Epoch 2/5
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0966 - accuracy: 0.9703
# Epoch 3/5
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0686 - accuracy: 0.9784
# Epoch 4/5
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0526 - accuracy: 0.9831
# Epoch 5/5
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0408 - accuracy: 0.9870
# 313/313 [==============================] - 1s 1ms/step - loss: 0.0863 - accuracy: 0.9745
# Test accuracy: 0.9745000004768372
