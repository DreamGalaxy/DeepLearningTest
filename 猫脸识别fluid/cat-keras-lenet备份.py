from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy as np
import lr_utils

x_train, y_train, x_test, y_test,classes = lr_utils.load_dataset()
x_train = x_train/255
x_test = x_test/255
y_train = y_train.reshape(y_train.shape[1],1)
train_set = 1 - y_train
train = np.hstack((y_train,train_set))
print train.shape
y_test = y_test.reshape(y_test.shape[1],1)
test_set = 1 - y_test
test = np.hstack((y_test,test_set))
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

model = Sequential()
model.add(Conv2D(6,(5,5),strides=(1,1),input_shape=(64,64,3),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(2,activation='softmax'))
sgd = SGD(lr=0.005, decay=1e-6, nesterov=True)
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,batch_size=10,epochs=2000,shuffle=True)
model.save('LeNet-5_model.h5')
#[0.10342620456655367 0.9834000068902969]
loss, accuracy=model.evaluate(x_test, y_test,batch_size=10)
print(loss, accuracy)