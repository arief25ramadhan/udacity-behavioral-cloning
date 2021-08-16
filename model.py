import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

def cnn_model():
    
    print("Building model")
    
    model = Sequential()
    
    model.add(Conv2D(filters=3, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))

    model.compile(loss='mse',optimizer='adam')

    return model