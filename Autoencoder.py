from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Dropout
import os


class Autoencoder:
    def __init__(self,
                 data: Tuple[Tuple[Any, Any],
                 Tuple[Any, Any]],
                 with_info: bool,
                 cmap,
                 norm: int,
                 shape_x: int,
                 shape_y: int,
                 shape_z: int,
                 factor: int,
                 kernel_size_small,
                 kernel_size_big,
                 activation: str,
                 padding_model: str,
                 learning_rate: float,
                 loss: str,
                 metrics,
                 epochs: int,
                 batch_size: int,
                 save_path: str,
                 color: int = "black",
                 padding: int = 2,
                 font_size: int = 16,
                 examples_amount: int = 0,
                 examples_row_size: int = 0, ):
        (self.__input_train, self.__output_train), (self.__input_test, self.__output_test) = data
        self.__with_info = with_info
        self.__examples_amount = examples_amount
        self.__examples_row_size = examples_row_size
        self.__cmap = cmap
        self.__color = color
        self.__padding = padding
        self.__font_size = font_size
        self.__input_validation = self.__input_test[:9000]
        self.__input_test = self.__input_test[9000:]
        self.__norm = norm
        self.__shape_x = shape_x
        self.__shape_y = shape_y
        self.__shape_z = shape_z
        self.__factor = factor
        self.__input_train_noise = self.__get_noise(self.__input_train)
        self.__input_test_noise = self.__get_noise(self.__input_test)
        self.__input_validation_noise = self.__get_noise(self.__input_validation)
        self.__kernel_size_small = kernel_size_small
        self.__kernel_size_big = kernel_size_big
        self.__activation = activation
        self.__padding_model = padding_model
        self.__learning_rate = learning_rate
        self.__loss = loss
        self.__metrics = metrics
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__save_path = save_path
        self.__predict = None

        # normalizing and reshaping
        self.__normalize_reshape_all()
        self.__after_init_processing()

        self.__model = self.__init_model()
        self.__compile_model()
        self.__is_learned = False

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------

    def __after_init_processing(self):
        if not self.__with_info:
            return
        self.__plot_examples(self.__input_train)
        self.__plot_examples(self.__input_train_noise)

    def __normalize_reshape_all(self):
        self.__input_validation = self.__normalize_reshape(self.__input_validation, self.__norm,
                                                           self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_test = self.__normalize_reshape(self.__input_test, self.__norm,
                                                     self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_train = self.__normalize_reshape(self.__input_train, self.__norm,
                                                      self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_train_noise = self.__normalize_reshape(self.__input_train_noise, self.__norm,
                                                            self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_test_noise = self.__normalize_reshape(self.__input_test_noise, self.__norm,
                                                           self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_validation_noise = self.__normalize_reshape(self.__input_validation_noise, self.__norm,
                                                                 self.__shape_x, self.__shape_y, self.__shape_z)

    @staticmethod
    def __normalize_reshape(data, norm, shape_x, shape_y, shape_z):
        data = data.astype('float32') / norm
        data = np.reshape(data, (data.shape[0], shape_x, shape_y, shape_z))
        return data

    def __get_noise(self, data):
        result = data + self.__factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        result = np.clip(result, 0., 1.)
        return result

    def __init_model(self):
        model = Sequential()
        # encoder network
        model.add(Conv2D(filters=128, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model, input_shape=(self.__shape_x, self.__shape_y, self.__shape_z)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(Conv2D(filters=256, kernel_size=self.__kernel_size_small, strides=self.__kernel_size_small,
                         activation=self.__activation, padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=self.__kernel_size_big, activation=self.__activation,
                         padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=self.__kernel_size_small, strides=self.__kernel_size_small,
                         activation=self.__activation, padding=self.__padding_model))

        # decoder network
        model.add(Conv2D(filters=512, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))

        model.add(tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=self.__kernel_size_small,
                                                  strides=self.__kernel_size_small, activation='relu',
                                                  padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))

        model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=self.__kernel_size_small,
                                                  strides=self.__kernel_size_small, activation='relu',
                                                  padding=self.__padding_model))
        model.add(Conv2D(filters=64, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(Conv2D(filters=1, kernel_size=self.__kernel_size_small, activation=self.__activation,
                         padding=self.__padding_model))

        if self.__with_info:
            model.summary()

        return model

    def __compile_model(self):
        optimizer = tf.keras.optimizers.Adam(self.__learning_rate)
        self.__model.compile(optimizer=optimizer, loss=self.__loss, metrics=self.__metrics)

    @staticmethod
    def __plot_history(history):
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.legend(loc='best')
        plt.show()

    def __after_train_processing(self):
        self.__is_learned = True
        self.__predict = self.__model.predict(self.__input_test_noise)
        return

    def __plot_examples(self, data):
        for i in range(self.__examples_amount):
            plt.subplot(self.__examples_row_size, self.__examples_row_size, i + 1)
            plt.imshow(data[i], cmap=self.__cmap)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def __predict_by_index(self, index):
        plt.title("real")
        plt.imshow(self.__input_test[index], cmap=self.__cmap)
        plt.show()
        plt.title("noised")
        plt.imshow(self.__input_test_noise[index], cmap=self.__cmap)
        plt.show()
        plt.title("denoised")
        plt.imshow(self.__predict[index], cmap=self.__cmap)
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------

    def teach(self):
        if os.path.exists(self.__save_path + ".index"):
            self.__model.load_weights(self.__save_path)
        else:
            history = self.__model.fit(self.__input_train_noise, self.__input_train, batch_size=self.__batch_size,
                                       epochs=self.__epochs,
                                       validation_data=(self.__input_validation_noise, self.__input_validation))
            self.__model.save_weights(self.__save_path)
            if self.__with_info:
                self.__plot_history(history)
        self.__after_train_processing()

    def get_input_shapes(self):
        return self.__input_train.shape, self.__input_test.shape

    def test_by_index(self):
        max_index = len(self.__input_test) - 1
        while True:
            inp = input(f"\nType index for test from 0 to {max_index - 1}\n")
            if inp == "":
                break
            try:
                index = int(inp)
                if index < 0 or index > max_index:
                    print(f"[start_test_input] wrong index {index}. Index must be from 0 to {max_index}")
                    continue
                self.__predict_by_index(int(inp))
            except:
                print(f"Wrong value {inp}")
