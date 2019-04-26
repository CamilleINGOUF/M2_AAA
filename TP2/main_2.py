import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
# from keras.optimizers import adam

def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255
    test_images = test_images / 255

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    # input dimensions
    num_train, img_rows, img_cols = train_images.shape
    depth = 1
    train_images = train_images.reshape(train_images.shape[0],img_rows, img_cols, depth)
    test_images = test_images.reshape(test_images.shape[0],img_rows, img_cols, depth)
    input_shape = (img_rows, img_cols, depth)
    # number of convolutional filters to use
    nb_filters = 32
    # pooling size
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    nb_classes = 10
    batch_size = 32
    nb_epoch = 10

    # Create a simple model with pooling and dropout
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size=kernel_size,activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    # tf.keras.optimizers.Adam()
    # tf.keras.losses.sparse_categorical_crossentropy
    model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    model.summary()

    model.fit(train_images, train_labels, batch_size=batch_size,epochs=nb_epoch, verbose=1,validation_data=(test_images, test_labels))
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
if __name__ == "__main__":
  main()