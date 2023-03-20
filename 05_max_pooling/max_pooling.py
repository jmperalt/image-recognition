# Import necessary libraries
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# Load the CIFAR-10 data set
# The data set contains 50,000 32x32 RGB training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to the range of 0 to 1
# This is a common preprocessing step for image data
x_train = x_train.astype('float32') # Convert data type to float32
x_test = x_test.astype('float32') # Convert data type to float32
x_train /= 255 # Divide by 255 to normalize the values between 0 and 1
x_test /= 255 # Divide by 255 to normalize the values between 0 and 1

# Convert the labels to one-hot encoded vectors
# This is necessary for the categorical cross-entropy loss function used in training
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the architecture of the neural network model
model = Sequential() # Create an empty model object

# Add the layers to the model
# The Conv2D layers use 3x3 filters and ReLU activation functions
# The MaxPooling2D layers downsample the feature maps by taking the maximum value in each 2x2 block
# The Flatten layer flattens the feature maps into a 1D array
# The Dense layers are fully connected layers with ReLU activation functions
# The output layer uses a softmax activation function to produce a probability distribution over the 10 classes
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Print a summary of the model
# This shows the structure of the model and the number of parameters in each layer
model.summary()
