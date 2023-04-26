import Autoencoder
import Settings

if __name__ == '__main__':
    encoder = Autoencoder.Autoencoder(Settings.DATASET, Settings.WITH_INFO, Settings.CMAP,
                                      Settings.RESHAPE_NORM, Settings.RESHAPE_X, Settings.RESHAPE_Y, Settings.RESHAPE_Z,
                                      Settings.NOISE_FACTOR, Settings.COLOR,
                                      Settings.PADDING, Settings.FONT_SIZE,
                                      Settings.EXAMPLES_AMOUNT,
                                      Settings.EXAMPLES_ROW_SIZE)
    (train_shape, test_shape) = encoder.get_input_shapes()
    print("Train shape: ", train_shape, "\nTest shape: ", test_shape)
