import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, TimeDistributed, GRU
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
import os
# 加载数据

train_all = np.load('data/encoded_train.npy')  # (175341, 32)
train_all_label = np.load('data/train_label.npy')  # (175341, 1)
test_all = np.load('data/encoded_test.npy')
test_all_label = np.load('data/test_label.npy')


# 利用TimesereisGenerator生成序列数据
time_steps = 1
batch_size = 1024

# 先把训练集划分出一部分作为验证集
train = train_all[:(172032+time_steps), :]   # 4096 * 42 = 172032
train_label = train_all_label[:(172032+time_steps), :]
test = test_all[:(81920+time_steps), :]  # 4096 * 20 = 81920
test_label = test_all_label[:(81920+time_steps), :]
# val_data = train_all[int(len(train_all)* 0.7):, :]
# val_label = train_all_label[int(len(train_all)* 0.7):, :]
# print(train.shape[0])
# print(val_data.shape[0])
# 数据集生成器
train_label_ = np.insert(train_label, 0, 0, axis=0)
test_label_ = np.insert(test_label, 0, 0, axis=0)
# val_label_ = np.insert(val_label, 0, 0)
train_generator = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
# val_generator = TimeseriesGenerator(val_data, val_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)

# 构造模型
# input_traffic = Input((time_steps, 32))
input_traffic = Input(shape=(time_steps, 32))
# 1 lstm layer, stateful=True
lstm1 = Bidirectional(LSTM(units=24, activation='tanh',
                           return_sequences=True, recurrent_dropout=0.1))(input_traffic)
lstm_drop1 = Dropout(0.5)(lstm1)
# 2 lstm layer, stateful=True
lstm2 = Bidirectional(LSTM(units=12, activation='tanh', return_sequences=False,
                           recurrent_dropout=0.1))(lstm_drop1)
lstm_drop2 = Dropout(0.5)(lstm2)
# lstm3 = Bidirectional(LSTM(units=8, activation='tanh', return_sequences=False,
#                            recurrent_dropout=0.1))(lstm_drop2)
# lstm_drop2 = Dropout(0.5)(lstm_drop1)
# mlp
mlp = Dense(units=6, activation='relu')(lstm_drop2)
mlp2 = Dense(units=1, activation='sigmoid')(mlp)
classifier = Model(input_traffic, mlp2)
optimize = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
classifier.compile(optimizer=optimize, loss='binary_crossentropy', metrics=['accuracy'])

# 设置一些callbacks
save_dir = os.path.join(os.getcwd(), 'models')
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=True,
                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduc_lr = ReduceLROnPlateau(monitor='val_acc', patience=10, mode='max', factor=0.2, min_delta=0.0001)

# 拟合及预测
history = classifier.fit_generator(train_generator, epochs=250, verbose=2, steps_per_epoch=168,
                                   callbacks=[checkpoint, tbCallBack, reduc_lr],
                                   validation_data=test_generator, shuffle=0, validation_steps=80)

classifier.load_weights('./models/best_model.hdf5')
train_probabilities = classifier.predict_generator(train_generator, verbose=1)

train_pred = train_probabilities > 0.5
train_label_original = train_label_[(time_steps-1):-2, :]

test_probabilities = classifier.predict_generator(test_generator, verbose=1)
test_pred = test_probabilities > 0.5
test_label_original = test_label_[(time_steps-1):-2, ]
np.save('data/plot_prediction.npy', test_pred)
np.save('data/plot_original.npy', test_label_original)
# tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
from sklearn.metrics import confusion_matrix, classification_report
print('Trainset Confusion Matrix')
print(confusion_matrix(train_label_original, train_pred))
print('Testset Confusion Matrix')
print(confusion_matrix(test_label_original, test_pred))
print('Classification Report')

print(classification_report(test_label_original, test_pred))
