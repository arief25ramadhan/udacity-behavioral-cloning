import cv2
import os
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.image import img_to_array, load_img

TARGET_SIZE = (96, 96)

def preprocess_image(image):
    
    cropped_image = image[45:145, :, :]
    resized_image = cv2.resize(cropped_image, TARGET_SIZE)
    quantized_image = resized_image.astype(np.float32)

    #Normalize image
    normalized_image = quantized_image/255.0 - 0.5
    return normalized_image

def generate_data(data_frame):
    
    camera_images = []
    steering_angles = []
    
    print("Loading original data")
    for i in tqdm(range(len(data_frame))):
        steering_center = data_frame['steer'][i]
        steering_correction = 0.25
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction
        
        path_center = data_frame['center'][i].replace('/root/Desktop/', '')
        path_left = data_frame['left'][i].replace('/root/Desktop/', '')
        path_right = data_frame['right'][i].replace('/root/Desktop/', '')

        img_center = load_img(path_center)
        img_left = load_img(path_left)
        img_right = load_img(path_right)
        
        img_center = preprocess_image(img_to_array(img_center))
        img_left = preprocess_image(img_to_array(img_left))
        img_right = preprocess_image(img_to_array(img_right))
        
        camera_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])
        
        flip_mode = random.randint(1,2)
            
        # flip right
        if flip_mode == 1:
            steering_flip = -1*steering_right
            img_flip = cv2.flip(img_right, 1)
      
        else:
            steering_flip = -1*steering_left
            img_flip = cv2.flip(img_left, 1)
        
        camera_images.extend([img_flip])
        steering_angles.extend([steering_flip])
            
    print("Arranging array")
    X_train, y_train = np.array(camera_images), np.array(steering_angles)
        
    print("Done generating data")
    return X_train, y_train