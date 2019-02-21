from tensorflow import keras
import numpy as np

import os
import csv

# 数据预处理
# 获得当前文件夹
cur_dir = os.path.dirname(os.path.realpath("__file__"))
# 获得文件路径
filename = cur_dir + "/datasets/winequality-red.csv"
# filename = cur_dir + "/datasets/winequality-white.csv"
# 读取数据
with open(filename) as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        data.append([i for i in row[0].split(';')])

# 将数据转为float32格式并删除第0行
data = np.array(data[1:]).astype(np.float32)

# 将数据划分为训练数据集和测试数据集,80%为训练集
ratio = 0.8
data_num = len(data)
slice = int(ratio * data_num)
# 前11列为特征，最后1列为标签
train_set = data[:slice, :11]
test_set = data[slice:, :11]
labels = np.array(data[:, 11]).astype(int)

# 将标签转为独热码形式
labels_one_hot = np.zeros((data_num, 10)).astype(int)
for i in range(data_num):
    labels_one_hot[i, labels[i]] = 1

train_labels = labels_one_hot[:slice, :]
test_labels = labels_one_hot[slice:, :]

# 定义模型
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss=keras.losses.categorical_crossentropy, metrics=['acc'])
num_epochs = 10
for epoch in range(num_epochs):
    history = model.fit(x=train_set, y=train_labels, shuffle=True, batch_size=32)
    print('epoch', epoch + 1, ', loss=', history.history['loss'])

loss, accuracy = model.evaluate(test_set, test_labels)
print('loss=', loss)
print('accuracy=', accuracy * 100, '%')
