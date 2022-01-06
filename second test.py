"""
Model condition: Fine!!!
Model type: multiple input (200 dim) to multiple output (3 dim)

開發日誌
2021/12/24__
使用ANN學習框架3D模型為7*7網格大樓模型
loss: 0.0157 - mae: 0.0927 - mse: 0.0157 - val_loss: 0.1125 - val_mae: 0.2647 - val_mse: 0.1125

"""

#%%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#%%
data = pd.read_csv("D:/00Dissertation_Discuss/file/second_test/test_data_ver2.csv")

data_copy = data.copy()
data_clean = data_copy.drop(columns = data_copy.columns[[0]])
data_clean2 = data_clean.drop(columns = data_clean.columns[[range(198,900)]])

#讓e去除
data_output = data_clean2.iloc[:, -3:]/1000
data_input = data_clean2.iloc[:, :-3]


#%%
#劃分訓練集與測試集
train_dataset = data_input.sample(frac=0.8,random_state=0)
test_dataset = data_input.drop(train_dataset.index)

#%%
#取出預測值
train_labels = data_output.iloc[train_dataset.index]
test_labels = data_output.drop(train_dataset.index)

#%%
#input normalization
#查看資料整體狀況
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

#標準化
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset).fillna(value=0)
normed_test_data = norm(test_dataset).fillna(value=0)

#%%
#output normalization

train_label_stats = train_labels.describe()
train_label_stats = train_label_stats.transpose()

#標準化
def norm(x):
  return (x - train_label_stats['mean']) / train_label_stats['std']
normed_train_data_labels = norm(train_labels).fillna(value=0)
normed_test_data_labels = norm(test_labels).fillna(value=0)


#%%
#建立模型
def build_model():
  model = keras.Sequential([
    layers.Dense(100, activation='sigmoid', input_shape=[len(train_dataset.keys())]),
    layers.Dense(100,  activation='sigmoid'),
    layers.Dropout(0.2),
    layers.Dense(100,  activation='sigmoid'),
    layers.Dropout(0.2),
    layers.Dense(100,  activation='sigmoid'),
    layers.Dropout(0.2),
    layers.Dense(3)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  optimizer = 'adam'

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

#創建模型物件
model = build_model()

#檢查模型
model.summary()

#%%
#提取一些測試模型是否運行
example_batch = normed_train_data[:10].to_numpy()
example_result = model.predict(example_batch)
print(example_result)


#%%
# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: 
        print('.')
    print('=', end='')

EPOCHS = 5000

history = model.fit(
  normed_train_data.to_numpy(), normed_train_data_labels.to_numpy(), batch_size = 40,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  callbacks=[PrintDot()])


#%%
#使用 history 对象中存储的统计信息可视化模型的训练进度
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#%%
#繪製學習曲線
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Total_sun]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([hist["mae"].min(),hist["val_mae"].max()])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Total_sun^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([hist["mse"].min(),hist["val_mse"].max()])
  plt.legend()
  plt.show()


plot_history(history)


#%%
'''
model = build_model()

# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data.to_numpy(), train_labels.to_numpy(), epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
'''

#%%
#預測結果
test_predictions = model.predict(normed_test_data)


#%%
#繪製
x_ax = range(len(normed_test_data))
normed_test_predictions = pd.DataFrame(test_predictions)

plt.scatter(x_ax, normed_test_data_labels.iloc[:,0],  s=6, label="y1-test", color='r')
plt.plot(x_ax, normed_test_predictions.iloc[:,0], label="y1-pred", color='r')
plt.legend()
plt.title("Sun_average")
plt.show()

plt.scatter(x_ax, normed_test_data_labels.iloc[:,1],  s=6, label="y2-test",color='b')
plt.plot(x_ax, normed_test_predictions.iloc[:,1], label="y2-pred", color='b')
plt.legend()
plt.title("total_radiation")
plt.show()

plt.scatter(x_ax, normed_test_data_labels.iloc[:,2],  s=6, label="y3-test", color='g')
plt.plot(x_ax, normed_test_predictions.iloc[:,2], label="y3-pred", color='g')
plt.legend()
plt.title("visibility")
plt.show()

#%%
#繪製比較關係
plt.scatter(normed_test_data_labels.iloc[:,0], normed_test_predictions.iloc[:,0], color='r')
plt.xlabel('True Values [Sun_average]')
plt.ylabel('Predictions [Sun_average]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-3,3])
plt.ylim([-3,3])
_= plt.plot([-100, 100], [-100, 100], 'r')
plt.show()


plt.scatter(normed_test_data_labels.iloc[:,1], normed_test_predictions.iloc[:,1], color='b')
plt.xlabel('True Values [total_radiation]')
plt.ylabel('Predictions [total_radiation]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-3,3])
plt.ylim([-3,3])
_= plt.plot([-100, 100], [-100, 100], 'b')
plt.show()


plt.scatter(normed_test_data_labels.iloc[:,2], normed_test_predictions.iloc[:,2], color='g')
plt.xlabel('True Values [visibility]')
plt.ylabel('Predictions [visibility]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-3,3])
plt.ylim([-3,3])
_ = plt.plot([-100, 100], [-100, 100], 'g')
plt.show()

#%%
"""
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
"""
