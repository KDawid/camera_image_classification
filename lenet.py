from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import AveragePooling2D, Conv2D


def get_model(num_classes, input_shape=(50, 50, 3)):
    lenet = Sequential()
    lenet.add(Conv2D(8, (3, 3), input_shape=input_shape, activation='relu'))
    lenet.add(AveragePooling2D())
    lenet.add(Conv2D(16, (3, 3), activation='relu'))
    lenet.add(AveragePooling2D())
    lenet.add(Flatten()),
    lenet.add(Dense(100, activation='relu'))
    lenet.add(Dense(80, activation='relu'))
    lenet.add(Dense(num_classes, activation='softmax'))

    lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lenet
