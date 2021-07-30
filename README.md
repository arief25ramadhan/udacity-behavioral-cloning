# **Behavioral Cloning** 

To do:
* Add Nvidia model image
* Explain briefly about it
* Add image of car
* Image of flipped
* Image of data augmentation
* Image Normal
* Grayscaling


## 1. Project Aim

This project aims to train a neural network to drive a car inside a track without getting out of the lane.

## 2. Steps

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



## 3. Files in the repository

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* utils.py containing the script to preprocess and augment image data

### 3.1. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
### 3.2. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## 4. Dataset Preparation

### 4.1. Dataset

We recorded the car movement through the sandy tracks. The laps we drove through wer:
- 3 laps of forward direction
- 2 laps of reverse direction

We collected the reverse laps to make make sure our model could generalize to various situation, and not overfit to the specified track.

### 4.2. Data Augmentation

Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. 
Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks. We perform data augmentation by flipping (mirroring) the image and add different lighting conditions to help reduce overfitting.

## 4. Model Architecture and Training

###  4.1.Model Architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 
The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

### 4.2. Hyperparameter Tuning and Selection 
The model contains dropout layers in order to reduce overfitting (model.py lines 21). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

## 5. Conclusion

To conclude, we have trained a neural network models based on Nvidia on NVIDIA's End to End Learning for Self-Driving Cars paper (https://arxiv.org/pdf/1604.07316v1.pdf). Our model was able to make the car goes through the lap without ever going out of the track.





