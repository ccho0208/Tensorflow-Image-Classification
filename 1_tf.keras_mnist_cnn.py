import os, sys, time, random
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# debugging logs
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# this should be placed before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# reproducibility
seed = 1234567
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))            # train: loss
        self.val_loss.append(logs.get('val_loss'))          # test: loss
        self.train_acc.append(logs.get('accuracy'))         # train: acc
        self.val_acc.append(logs.get('val_accuracy'))       # test: acc


def summarize_history(train_loss, val_loss, train_acc, val_acc, fn):
    """ plot & save accuracy & loss curves """
    x_len = len(train_loss)
    plt.figure(); plt.clf()
    plt.subplot(211)
    plt.plot(range(1,x_len+1), train_loss, color='green')
    plt.plot(range(1,x_len+1), val_loss, color='orange')
    plt.ylabel('loss'); plt.grid(True)
    plt.legend(['train','val'], loc='upper left')

    plt.subplot(212)
    plt.plot(range(1,x_len+1), train_acc, color='green')
    plt.plot(range(1,x_len+1), val_acc, color='orange')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.grid(True)
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(fn); plt.close()


def main():
    start = time.time()

    # 0) Set parameters
    num_classes = 10
    epochs = 10
    batch_size = 128

    # input image dimensions
    img_x, img_y = 28, 28

    history = AccuracyHistory()

    # 1) Prepare dataset: MNIST
    # load MNIST dataset which already splits into train and test sets
    (x_train, y_train), (x_test, y_test) = load_data()

    # reshape that data into a 4D tensor - (samples, x_img_size, y_img_size, num_channels)
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # convert the data to the right types
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # 2) Build a model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # print out the model
    # model.summary()
    # plot_model(model, 'model_cnn.png', show_shapes=True)

    # 3) Train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                     verbose=1, validation_data=(x_test,y_test), callbacks=[history])

    # 4) Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)

    # show the final results
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # plot training curves
    summarize_history(hist.history['loss'], hist.history['val_loss'],
                      hist.history['accuracy'], hist.history['val_accuracy'], 'train_curves.png')

    elapsed = time.time() - start
    print('Elapsed {:.2f} mins'.format(elapsed/60.0))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
