import os
import numpy as np

import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
classes = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
           'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
           'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
           'i_love_you', 'thank_you', 'sorry']
num_of_timesteps = 9
num_classes = len(classes)

X, y = [], []
label = 0

for cl in classes:
    for file in os.listdir(f'./dataset/{cl}'):
        print(f'Reading: ./dataset/{cl}/{file}')
        data = pd.read_csv(f'./dataset/{cl}/{file}')
        data = data.values
        n_sample = len(data)
        print(n_sample)
        for i in range(num_of_timesteps, n_sample):
            X.append(data[i - num_of_timesteps : i, :])
            y.append(label)
    label = label + 1


X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.4))
model.add(Dense(units=num_classes, activation="softmax"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="categorical_crossentropy")
model.summary()

# plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)  
history = model.fit(X_train, y_train, epochs=20, batch_size=528, validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save(f"model/model_{num_of_timesteps}.keras")

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
def visualize_accuracy(history, title):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(accuracy))
    plt.figure()
    plt.plot(epochs, accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss");visualize_accuracy(history, "Training and Validation Accuracy")
