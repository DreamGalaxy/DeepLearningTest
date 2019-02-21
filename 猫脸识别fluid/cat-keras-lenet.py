from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
import lr_utils

x_train, y_train, x_test, y_test, classes = lr_utils.load_dataset()
x_train = x_train/255
x_test = x_test/255
y_train = y_train.reshape(y_train.shape[1], 1)
train_set = 1 - y_train
y_test = y_test.reshape(y_test.shape[1], 1)
test_set = 1 - y_test
test = np.hstack((y_test, test_set))
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile('categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=250)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=10)
print('loss=', loss, 'accuracy=', accuracy)
