# import necessary packages
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

# Load data set
# using cifar10 dataset from keras.datasets library
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data set to 0-to-1 range
# convert values from integers to floats and normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
# convert y_train and y_test to binary matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = Sequential()

# add a convolutional layer with 32 filters of size 3x3, using 'same' padding and ReLU activation function
# input_shape set to (32, 32, 3) for RGB images with 32x32 pixels
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)))

# add another convolutional layer with 32 filters of size 3x3 and ReLU activation function
model.add(Conv2D(32, (3, 3), activation="relu"))

# add another convolutional layer with 64 filters of size 3x3, using 'same' padding and ReLU activation function
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))

# add another convolutional layer with 64 filters of size 3x3 and ReLU activation function
model.add(Conv2D(64, (3, 3), activation="relu"))

# add a flatten layer to flatten the output of the previous layer into a 1-dimensional array
model.add(Flatten())

# add a dense layer with 512 neurons and ReLU activation function
model.add(Dense(512, activation="relu"))

# add a dense layer with 10 neurons and softmax activation function
model.add(Dense(10, activation="softmax"))

# Print a summary of the model
model.summary()
