import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt


# 画图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

# 读取数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
all_features = pd.concat((train_data.loc[:, 'MSSubClass':'SaleCondition'],
                          test_data.loc[:, 'MSSubClass':'SaleCondition']))

# 预处理数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.fillna(all_features.mean())

n_train = train_data.shape[0]
train_features = all_features[:n_train].values
test_features = all_features[n_train:].values
train_labels = train_data.SalePrice.values

# 定义超参数
k = 5
num_epochs = 800
verbose_epoch = num_epochs - 2
lr = 5
batch_size = 64


# K折交叉验证
fold_size = train_features.shape[0] // k
train_l_sum = 0.0
test_l_sum = 0.0
for test_i in range(k):
    # 定义模型
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, kernel_initializer=keras.initializers.glorot_normal()))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.mean_squared_logarithmic_error, metrics=['acc'])
    X_val_test = train_features[test_i * fold_size: (test_i + 1) * fold_size, :]
    y_val_test = train_labels[test_i * fold_size: (test_i + 1) * fold_size]
    val_train_defined = False
    for i in range(k):
        if i != test_i:
            X_cur_fold = train_features[i * fold_size: (i + 1) * fold_size, :]
            y_cur_fold = train_labels[i * fold_size: (i + 1) * fold_size]
            if not val_train_defined:
                X_val_train = X_cur_fold
                y_val_train = y_cur_fold
                val_train_defined = True
            else:
                X_val_train = np.concatenate((X_val_train, X_cur_fold), axis=0)
                y_val_train = np.concatenate((y_val_train, y_cur_fold), axis=0)
    train_ls = []
    test_ls = []
    for epoch in range(num_epochs):
        history = model.fit(X_val_train, y_val_train, batch_size=batch_size, shuffle=True)
        train_ls.append(history.history['loss'])
        if epoch >= verbose_epoch:
            print('epoch %d, train loss: %f' % (epoch, history.history['loss'][0]))
        test_l = model.evaluate(X_val_test, y_val_test, batch_size=batch_size)
        test_ls.append(test_l)
    train_l = history.history['loss'][0]
    train_l_sum += train_l
    print('test loss: %f' % test_l[0])
    test_l_sum += test_l[0]
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])

print('%d-fold validation: avg train loss: %f, avg test loss: %f'
      % (k, train_l_sum / k, test_l_sum / k))

model = keras.Sequential()
model.add(keras.layers.Dense(1, kernel_initializer=keras.initializers.glorot_normal()))
model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.mean_squared_logarithmic_error, metrics=['acc'])
model.fit(train_features, train_labels, batch_size=batch_size, shuffle=True)
preds = model.predict(test_features)
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)
