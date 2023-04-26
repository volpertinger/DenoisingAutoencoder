from keras.datasets import mnist
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------

DATASET = mnist.load_data()
RESHAPE_X = 28
RESHAPE_Y = 28
RESHAPE_Z = 1
RESHAPE_NORM = 255
KERNEL_SIZE_SMALL = (2, 2)
KERNEL_SIZE_BIG = (3, 3)
ACTIVATION = "relu"
PADDING_MODEL = "same"
LEARNING_RATE = 0.001
LOSS = 'mean_squared_error'
METRICS = ['accuracy']

# ----------------------------------------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------------------------------------

NOISE_FACTOR = 0.39
WITH_INFO = True
EPOCHS = 5
BATCH_SIZE = 256
SAVE_PATH = "saves/mnist_save"

# ----------------------------------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------------------------------

EXAMPLES_AMOUNT = 25
EXAMPLES_ROW_SIZE = 5
CMAP = plt.cm.binary
COLOR = "black"
PADDING = 1
FONT_SIZE = 16
