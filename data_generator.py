from data_processing import load_data
from build_model import build_SAE
import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Reshape, Dropout
import os
# load data
print("Load data...")
train, train_label, test, test_label = load_data()  # data:(, 196) label:(, 10)
print("train shape: ", train.shape)
train_label = train_label.reshape((-1, 10))
test_label = test_label.reshape((-1, 10))
print("train_label shape: ", train_label.shape)

np.save('multi_train_label.npy', train_label)
np.save('multi_test_label.npy', test_label)
# build model
print("Build AE model")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, sSAE_encoder = build_SAE(rho=0.04)

print("Start pre-training....")

# fit the first layer, 在此处添加validation_data=test，加上callbacks，记录的是val_loss，取最小的那个
print("First layer training....")
AE_1_dir = os.path.join(os.getcwd(), 'saved_ae_1')
ae_1_filepath="best_ae_1.hdf5"
ae_1_point = ModelCheckpoint(os.path.join(AE_1_dir, ae_1_filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_1_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_1.fit(train, train, epochs=100, batch_size=1024, validation_data=(test, test), verbose=0, shuffle=True, callbacks=[ae_1_point, ae_1_stops])

autoencoder_1.load_weights('./saved_ae_1/best_ae_1.hdf5')
first_layer_output = encoder_1.predict(train)  # 在此使用loss最小的那个模型
test_first_out = encoder_1.predict(test)
print("The shape of first layer output is: ", first_layer_output.shape)

# fit the second layer
print("Second layer training....")
AE_2_dir = os.path.join(os.getcwd(), 'saved_ae_2')
ae_2_filepath="best_ae_2.hdf5"
ae_2_point = ModelCheckpoint(os.path.join(AE_2_dir, ae_2_filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_2_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_2.fit(first_layer_output, first_layer_output, epochs=100, batch_size=512, verbose=0, validation_data=(test_first_out, test_first_out), shuffle=True, callbacks=[ae_2_point, ae_2_stops])

autoencoder_2.load_weights('./saved_ae_2/best_ae_2.hdf5')
second_layer_output = encoder_2.predict(first_layer_output)
test_second_out = encoder_2.predict(test_first_out)
print("The shape of second layer output is: ", second_layer_output.shape)

# fit the third layer
print("Third layer training....")
AE_3_dir = os.path.join(os.getcwd(), 'saved_ae_3')
ae_3_filepath="best_ae_3.hdf5"
ae_3_point = ModelCheckpoint(os.path.join(AE_3_dir, ae_3_filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_3_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_3.fit(second_layer_output, second_layer_output, epochs=100, batch_size=512, verbose=0, validation_data=(test_second_out, test_second_out), shuffle=True, callbacks=[ae_3_point, ae_3_stops])
autoencoder_3.load_weights('./saved_ae_3/best_ae_3.hdf5')

print("Pass the weights to sSAE_encoder...")
sSAE_encoder.layers[1].set_weights(autoencoder_1.layers[2].get_weights())  # first Dense
sSAE_encoder.layers[2].set_weights(autoencoder_1.layers[3].get_weights())  # first BN
sSAE_encoder.layers[3].set_weights(autoencoder_2.layers[2].get_weights())  # second Dense
sSAE_encoder.layers[4].set_weights(autoencoder_2.layers[3].get_weights())  # second BN
sSAE_encoder.layers[5].set_weights(autoencoder_3.layers[2].get_weights())  # third Dense
sSAE_encoder.layers[6].set_weights(autoencoder_3.layers[3].get_weights())  # third BN
encoded_train = sSAE_encoder.predict(train)
encoded_test = sSAE_encoder.predict(test)

np.save('data/encoded_train.npy', encoded_train)
np.save('data/train_label.npy', train_label)
np.save('data/encoded_test.npy', encoded_test)
np.save('data/test_label.npy', test_label)

# 级联两层Dense 最后加一个softmax
mlp0 = Dense(units=32, activation='relu')(sSAE_encoder.output)
lstm_reshape = Reshape((1, 32))(mlp0)
# lstm1 = LSTM(units=16, activation='tanh', input_shape=(1, 32), return_sequences=True)(lstm_reshape)
# lstm_drop = Dropout(0.3)(lstm1)
lstm2 = LSTM(units=16, activation='tanh', return_sequences=False)(lstm_reshape)
lstm_drop2 = Dropout(0.3)(lstm2)
# lstm3 = LSTM(units=16, activation='tanh', input_shape=(1, 32), return_sequences=False)(lstm_drop)
# mlp_pool = GlobalAveragePooling1D()(lstm2)
mlp = Dense(units=10, activation='relu')(lstm_drop2)
mlp2 = Dense(units=1, activation='sigmoid')(mlp)
classifier = Model(sSAE_encoder.input, mlp2)
optimize = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
classifier.compile(optimizer=optimize, loss='binary_crossentropy', metrics=['accuracy'])

save_dir = os.path.join(os.getcwd(), 'saved_models_temp')
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=True,
                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduc_lr = ReduceLROnPlateau(monitor='val_acc', patience=10, mode='max', factor=0.2, epsilon=0.0001)

train_label_two = np.load('saved_models3/train_label.npy')
test_label_two = np.load('saved_models3/test_label.npy')
history = classifier.fit(train, train_label_two, epochs=100, batch_size=1024, validation_data=(test, test_label_two), callbacks=[checkpoint, tbCallBack, reduc_lr], verbose=2)
classifier.load_weights('saved_models_temp/best_model.hdf5')
# 保存下最好的模型，然后重新构建一个模型，毕竟这里只是一个预训练的过程。
# 然后，在后面的过程中，首先使用的AE，并且对结果进行TimesereisGenerator
# 最后，输入到LSTM中实现分类。
train_y = classifier.predict(train)
train_pred = train_y > 0.5

test_y = classifier.predict(test)
test_pred = test_y > 0.5

from sklearn.metrics import confusion_matrix
print(confusion_matrix(train_label_two, train_pred))
print(confusion_matrix(test_label_two, test_pred))

loss, acc = classifier.evaluate(test, test_label_two)
print("Loss: {:.2f}, Acc: {:.2f}".format(loss, acc))
# test_pred = classifier.predict(test)
# # prediction = np.argmax(test_pred, axis=1)
# # true_digit = np.argmax(test_label, axis=1)
#
# n_correct = np.sum(np.equal(test_pred, test_label).astype(int))
# total = float(len(test_pred))
# print("ACC is : ", round(n_correct/total, 3))

# encoded_train = sSAE_encoder.predict(train)
# encoded_test = sSAE_encoder.predict(test)
# np.save('encoded_train.npy', encoded_train)
# np.save('encoded_test.npy', encoded_test)
