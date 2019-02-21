import numpy as np
import lr_utils

from tensorflow import keras

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# 定义纬度
DATA_DIM = num_px * num_px * 3

# 转换数据形状
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

# 归一化
train_set_x_flatten = train_set_x_flatten/255.0
test_set_x_flatten = test_set_x_flatten/255.0

# 将标签转为独热码形式


# 定义模型
model = keras.Sequential()
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.categorical_crossentropy, metrics=['acc'])
num_epochs = 1500
for epoch in range(num_epochs):
    history = model.fit(x=train_set_x_flatten, y=train_set_y_one_hot, shuffle=True, batch_size=32)
    print('epoch', epoch + 1, ', loss=', history.history['loss'])

loss, accuracy = model.evaluate(test_set_x_flatten, test_set_y_one_hot)
print('loss=', loss)
print('accuracy=', accuracy * 100, '%')