from data_loader import generate_data
from model import cnn_model

import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
 
df = pd.read_csv('dataset/driving_log.csv', names=['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed'])

# Load data
X_train, y_train = generate_data(df)

# Load model
model = cnn_model()
checkpoint = ModelCheckpoint('model_fix_2.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 

print("Fitting the model")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=100, batch_size=64, callbacks=[checkpoint])
# print("Saving model")
# model.save('model_fix_2.h5')