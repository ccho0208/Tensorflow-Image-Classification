# CNN-VGG for CIFAR-10 Photo Classification
#
#   https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
#
#   1) CIFAR-10 Photo Classification Dataset
#   2) Model Evaluation Test Harness
#   3) How to Develop a Baseline Model
#   4) How to Develop an Improved Model
#   5) How to Develop Further Improvements
#   6) How to Finalize the Model and Make Predictions
#
#   Baseline + Increasing Dropout + Data Augmentation + Batch Normalization: 88.620%
#
#   What needs to do: Learning Rates.
#                     Explore alternate learning rates, adaptive learning rates, and learning rate schedules and then
#                     compare performance.
#
import sys
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# parameters
num_classes = 10
num_epochs = 100
batch_size = 64


def main():
    #
    # 1a) Prepare Dataset: CIFAR10
    # trainX: (50000, 32, 32, 3), trainY: (50000, 1)
    # testX:  (10000, 32, 32, 3), testY:  (10000, 1)
    trainX, trainY, testX, testY = load_dataset()

    #
    # 1b) Data Augmentation
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    it_train = datagen.flow(trainX, trainY, batch_size=batch_size)
    steps = int(trainX.shape[0]/batch_size)

    #
    # 2) Define Model Architecture
    model = define_model()

    #
    # 3) System Training
    #hist = model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, validation_data=(testX,testY), verbose=1)
    hist = model.fit_generator(it_train, steps_per_epoch=steps, epochs=num_epochs, validation_data=(testX, testY), verbose=1)

    #
    # 4) System Evaluation
    score = model.evaluate(testX, testY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #
    # 5) Plot Learning Curves
    summarize_diagnostics(hist)

    # save model
    model.save('final_model.h5')


def load_dataset():
    # load dataset: CIFAT10
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    # convert the data to the right types
    trainX, testX = trainX.astype('float32')/255.0, testX.astype('float32')/255.0

    # convert class labels to one-hot encoding
    trainY = tf.keras.utils.to_categorical(trainY, num_classes)
    testY = tf.keras.utils.to_categorical(testY, num_classes)
    return trainX, trainY, testX, testY


def define_model():
    # model architecture
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(hist):
    # plot loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(hist.history['loss'],color='blue',label='train')
    plt.plot(hist.history['val_loss'],color='orange',label='test')
    plt.grid()

    # plot accuracy
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(hist.history['accuracy'], color='blue', label='train')
    plt.plot(hist.history['val_accuracy'], color='orange', label='test')
    plt.grid()

    # save the plot to file
    plt.savefig('test.png')
    plt.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)
