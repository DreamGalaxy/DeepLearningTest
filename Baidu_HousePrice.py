from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense
import numpy as np

from matplotlib import pyplot as plt

learning_rate = 0.008
num_epochs = 30

with open('HousePriceData.txt', 'r') as f:
    features = []
    prices = []
    for row in f.readlines():
        row = row.split(', ')
        features.append(np.float32(row[0]))
        prices.append(np.float32(row[1][:-1]))
    features = np.array(features).reshape(len(features), 1)
    prices = np.array(prices).reshape(len(prices), 1)
    # 保留原始数据用于最后画图
    x_f = features
    y_p = prices
    # 缩小数据方便后续计算
    features = features/100
    prices = prices/100


# 定义模型
model = Sequential()
model.add(Dense(1, kernel_initializer='random_normal', bias_initializer='zeros'))
# 编译模型
model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss='mse', metrics=['acc'])
# 训练模型
batch_size = 10
for epoch in range(num_epochs):
    history = model.fit(x=features, y=prices, batch_size=batch_size, shuffle=True)
    print('epoch ', epoch+1, ',loss = ', history.history['loss'])
print("w, b=", model.get_weights())
[w, b] = model.get_weights()


# 画图
# 青色点为数据散点图
# 蓝色方块为预期直线
# 红色圆圈为预测直线
x = np.arange(0, 200, 2)
plt.scatter(x_f, y_p, 3, hold=True, c='c')
plt.scatter(x, 6.7*x-24.42, 13, hold=True, c='b', marker='s')
plt.scatter(x, w*x+b*100, 7, c='r', marker='o')
plt.show()

