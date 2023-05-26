import tensorflow as tf
import numpy as np
from tensorflow import keras


EPOCHS = 50
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
Dropout = 0.3

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_train.reshape(60000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_train.astype('float32')

#对于数据进行归一化，因为每个像素的灰度值最大为255，所以广播到数组所有位置/255.
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#One-hot
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)


#build the model(add the hidden layers).
model = keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, name= 'dense_layer1', activation='relu'))
model.add(keras.layers.Dropout(Dropout))
model.add(keras.layers.Dense(N_HIDDEN, name= 'dense_layer2', activation='relu'))
model.add(keras.layers.Dropout(Dropout))
model.add(keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED,), name='dense_layer', activation='softmax'))


#Compile the model.

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])



#record the data, and do it in terminal: tensorboard --logdir=logs
tf_callback = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]

#Train the model.
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[tf_callback])



model.summary()
#Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy: ', test_acc)

