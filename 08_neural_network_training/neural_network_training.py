import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load CIFAR-10 data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to a range of 0 to 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class labels to binary class matrices (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a Sequential model
model = Sequential()

# Add layers to the model
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu")) # Convolutional layer with 32 filters, 3x3 kernel size, same padding, and ReLU activation
model.add(Conv2D(32, (3, 3), activation="relu")) # Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling layer with 2x2 pool size
model.add(Dropout(0.25)) # Dropout layer to prevent overfitting

model.add(Conv2D(64, (3, 3), padding='same', activation="relu")) # Convolutional layer with 64 filters, 3x3 kernel size, same padding, and ReLU activation
model.add(Conv2D(64, (3, 3), activation="relu")) # Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling layer with 2x2 pool size
model.add(Dropout(0.25)) # Dropout layer to prevent overfitting

model.add(Flatten()) # Flatten layer to convert 2D feature maps to a 1D feature vector
model.add(Dense(512, activation="relu")) # Fully connected layer with 512 units and ReLU activation
model.add(Dropout(0.5)) # Dropout layer to prevent overfitting
model.add(Dense(10, activation="softmax")) # Output layer with 10 units (one for each class) and softmax activation

# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metric
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Train the model on the training data set with specified batch size, number of epochs, validation data set, and shuffle
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)