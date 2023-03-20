# Import required libraries
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# Load CIFAR-10 dataset from keras.datasets module
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values in the range of 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class labels to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the model
model = Sequential()

# Add the first convolutional layer with 32 filters, 3x3 kernel size, 'same' padding, and 'relu' activation
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))

# Add the second convolutional layer with 32 filters, 3x3 kernel size, and 'relu' activation
model.add(Conv2D(32, (3, 3), activation="relu"))

# Add max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout layer with 25% probability
model.add(Dropout(0.25))

# Add another convolutional layer with 64 filters, 3x3 kernel size, 'same' padding, and 'relu' activation
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

# Add another convolutional layer with 64 filters, 3x3 kernel size, and 'relu' activation
model.add(Conv2D(64, (3, 3), activation="relu"))

# Add max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout layer with 25% probability
model.add(Dropout(0.25))

# Flatten the output of the previous layer
model.add(Flatten())

# Add a fully connected layer with 512 units and 'relu' activation
model.add(Dense(512, activation="relu"))

# Add dropout layer with 50% probability
model.add(Dropout(0.5))

# Add output layer with 10 units and 'softmax' activation for multiclass classification
model.add(Dense(10, activation="softmax"))

# Print the model summary to check the model architecture
model.summary()
