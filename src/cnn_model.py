from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D , BatchNormalization

def build_model():
    #CNN model
    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5,5),
            padding='Same',
            activation='relu',
            input_shape = (150,150,3)
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2,2)
        )
    )

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            padding='Same',
            activation='relu'
        )
    )

    model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )

    model.add(
        Conv2D(
            filters=96,
            kernel_size=(3, 3),
            padding='Same',
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        Conv2D(
            filters=96,
            kernel_size=(3, 3),
            padding='Same',
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(
        Dense(
            2,
            activation='softmax'
        )
    )

    return model

model = build_model()
model.summary()