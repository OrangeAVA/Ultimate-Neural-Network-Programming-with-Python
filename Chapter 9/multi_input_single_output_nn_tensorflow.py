import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import numpy as np


class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Starting training...")


    def on_train_end(self, logs=None):
        print("Finished training.")


    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")


    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")
        print(f"Train loss: {logs['loss']}")


    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training: Starting batch {batch}")


    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}")
        print(f"Train loss: {logs['loss']}")


    def on_test_begin(self, logs=None):
        print("Starting testing...")


    def on_test_end(self, logs=None):
        print("Finished testing.")


    def on_test_batch_begin(self, batch, logs=None):
        print(f"Testing: Starting batch {batch}")


    def on_test_batch_end(self, batch, logs=None):
        print(f"Testing: Finished batch {batch}")
        print(f"Test loss: {logs['loss']}")


# Custom Loss
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Model
inputA = Input(shape=(32,))
inputB = Input(shape=(100,))


# For inputA
x = Embedding(input_dim=10000, output_dim=64)(inputA)
x = Flatten()(x)
x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
x = Model(inputs=inputA, outputs=x)


# For inputB
y = Dense(64, activation='sigmoid', kernel_initializer='glorot_uniform')(inputB)
y = Model(inputs=inputB, outputs=y)


combined = Concatenate()([x.output, y.output])


z = Dense(10, activation="linear")(combined)
z = Dense(1, activation="sigmoid")(z)


model = Model(inputs=[x.input, y.input], outputs=z)


model.compile(loss=custom_loss, optimizer="adam")


# Data
num_samples = 1000
inputA_data = np.random.randint(10000, size=(num_samples, 32))
inputB_data = np.random.rand(num_samples, 100)
labels = np.random.randint(2, size=(num_samples, 1))


# Fit the model
history = model.fit([inputA_data, inputB_data], labels, epochs=50, callbacks=[CustomCallback()])


#Output:
# Training: Starting batch 31
# Training: Finished batch 31
# Train loss: 6.00976136411191e-06
# Finished epoch 49
# Train loss: 6.00976136411191e-06
