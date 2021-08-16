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

This project aims to train a neural network to drive a car inside a track without getting out of the lane for at least one lap.

## 2. Steps

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
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
* model.py containing the script to build the model
* main.py containing the script to train the model
* data_loader.py where data is loaded and augmented
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### 3.1. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
### 3.2. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## 4. Dataset Preparation

### 4.1. Dataset

We drove the car movement for three laps in the sandy track. We tried to put our car in the middle lane and avoided going off track.

### 4.2. Data Augmentation

We augmented our data to increase the diversity of the training data. We did that by flipping the left and right image of the car.

## 4. Model Architecture and Training

###  4.1.Model Architecture

At first, we mimic the architecture from Nvidia's End to End Learning for Self Driving Car paper. The architecture of our model is displayed by Figure 1 as shown below.  However, this makes our model fit too rigidly to the training set, while not performing well enough during the test. So we drop some of the layers and add dropout to avoid overfit from happening. Figure 2 informs the layer in our neural network model.

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/cnn.PNG"  width="400">
 <br>
 <em>Figure 1 - Nvidia Model Architecture</em>
</p>

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/cnn.PNG"  width="400">
 <br>
 <em>Figure 2 - Our Model Architecture</em>
</p>


### 4.2. Hyperparameter Tuning and Selection 

Our model is build using Sequential API of Keras framework as show in model.py file. To sum up, we use ReLU activation function and Adam optimizer. We train our model for 100 epochs, in which, through our observation the model has reached an optimal value.

The model contains dropout layers and was trained and validated on different data sets to avoid overfitting. The model was tested by running it through the simulator, where it must stay on the track for at least one lap.

## 5. Conclusion

To conclude, we have trained a neural network models based on Nvidia on NVIDIA's End to End Learning for Self-Driving Cars paper (https://arxiv.org/pdf/1604.07316v1.pdf). Our model was able to make the car goes through the lap without ever going out of the track.
