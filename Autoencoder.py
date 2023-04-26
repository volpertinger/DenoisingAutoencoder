from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Dropout


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

        self.__normalize_reshape_all()
        self.__after_init_processing()

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------

    def __after_init_processing(self):
        if not self.__with_info:
            return
        self.plot_examples(self.__input_train)
        self.plot_examples(self.__input_train_noise)

    def __normalize_reshape_all(self):
        self.__input_validation = self.__normalize_reshape(self.__input_validation, self.__norm,
                                                           self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_test = self.__normalize_reshape(self.__input_test, self.__norm,
                                                     self.__shape_x, self.__shape_y, self.__shape_z)
        self.__input_train = self.__normalize_reshape(self.__input_train, self.__norm,
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

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------
    def get_input_shapes(self):
        return self.__input_train.shape, self.__input_test.shape

    def plot_examples(self, data):
        for i in range(self.__examples_amount):
            plt.subplot(self.__examples_row_size, self.__examples_row_size, i + 1)
            plt.imshow(data[i], cmap=self.__cmap)
            plt.xticks([])
            plt.yticks([])
        plt.show()
