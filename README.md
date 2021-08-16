# **Behavioral Cloning** 

## 1. Project Aim

This project aims to train a neural network to drive a car inside a track without getting out of the lane for at least one lap.

## 2. Steps

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## 3. Files in the repository

My project includes the following files:
* model.py containing the script to build the model
* main.py containing the script to train the model
* data_loader.py where data is loaded and augmented
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### 3.1. Submission includes functional code and video
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```
Furthermore, we can also download the video.mp4 to see the autonomous drive of our car.

## 4. Dataset Preparation

### 4.1. Data Collection and Preparation

We drove the car movement for three laps in the sandy track. We tried to put our car in the middle lane and avoided going off track. The image data from our drive is recorded and processed in the pipeline.

The function preprocess in data_loader.py is where  

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
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/model_sum.png"  width="400">
 <br>
 <em>Figure 2 - Our Model Architecture</em>
</p>


### 4.2. Hyperparameter Tuning and Selection 

Our model is build using Sequential API of Keras framework as show in model.py file. To sum up, we use ReLU activation function and Adam optimizer. We train our model for 100 epochs, in which, through our observation the model has reached an optimal value.

The model contains dropout layers and was trained and validated on different data sets to avoid overfitting. The model was tested by running it through the simulator, where it must stay on the track for at least one lap.

## 5. Conclusion

To conclude, we have trained a neural network models based on Nvidia on NVIDIA's End to End Learning for Self-Driving Cars paper (https://arxiv.org/pdf/1604.07316v1.pdf). Our model was able to make the car goes through the lap without ever going out of the track, as recorded in the video.mp4 file.
