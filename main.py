import Autoencoder
import Settings

if __name__ == '__main__':
    encoder = Autoencoder.Autoencoder(Settings.DATASET, Settings.WITH_INFO, Settings.CMAP,
                                      Settings.RESHAPE_NORM, Settings.RESHAPE_X, Settings.RESHAPE_Y, Settings.RESHAPE_Z,
                                      Settings.NOISE_FACTOR, Settings.KERNEL_SIZE_SMALL, Settings.KERNEL_SIZE_BIG,
                                      Settings.ACTIVATION, Settings.PADDING_MODEL, Settings.LEARNING_RATE,
                                      Settings.LOSS, Settings.METRICS, Settings.EPOCHS, Settings.BATCH_SIZE,
                                      Settings.SAVE_PATH, Settings.COLOR,
                                      Settings.PADDING, Settings.FONT_SIZE,
                                      Settings.EXAMPLES_AMOUNT,
                                      Settings.EXAMPLES_ROW_SIZE)
    (train_shape, test_shape) = encoder.get_input_shapes()
    print("Train shape: ", train_shape, "\nTest shape: ", test_shape)
    encoder.teach()
    encoder.test_by_index()
