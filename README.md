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

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```
Furthermore, we can also download the video.mp4 to see the autonomous drive of our car.

## 4. Dataset Preparation

### 4.1. Data Collection and Preparation

We drove the car movement for three laps in the sandy track. We tried to put our car in the middle lane and avoided going off track. The image data from our drive is recorded and processed in the pipeline.

The function preprocess_image in data_loader.py is where our preprocessing happens. This includes cropping on the regions of interest, resizing, quantizing, and normalization. The purpose of preprocessing is to extract the relevant info from the data, so it becomes lightweight but still highly functional. 

Figure 1 displays the left, center, and right images as seen by our camera.
<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/left_2021_07_25_09_00_02_678.jpg"  width="200">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/center_2021_07_25_09_00_02_678.jpg"  width="200">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/right_2021_07_25_09_00_02_678.jpg"  width="200">
 <br>
 <em>Figure 1 - Image Data from Recording: Left, Center, and Right Camera</em>
</p>

Next, we cropped the image to concentrate on the region of interest, as shown by Figure 2. 

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/left_cropped.jpg"  width="200">
 <br>
 <em>Figure 2 - Cropped Left Image</em>
</p>

After that, we transformed it into a size of 96x96 pixels. Figure 3 displays the resized left camera image.

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/left_resized.jpg"  width="200">
 <br>
 <em>Figure 3 - Resized Left Image</em>
</p>

Finally, we changed the image format into float32 and normalized it to make it more compact.

### 4.2. Data Augmentation

We augmented our data to increase the diversity of the training data. We did that by flipping the left and right image of the car, as shown by Figure 4.

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/left_flipped.jpg"  width="300">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/sample_images/right_flipped.jpg"  width="300">
 <br>
 <em>Figure 4 - Flipped Image: Left and Right Camera</em>
</p>


## 4. Model Architecture and Training

###  4.1.Model Architecture

At first, we mimic the architecture from Nvidia's End to End Learning for Self Driving Car paper. The architecture of our model is displayed by Figure 5 as shown below.  However, this makes our model fit too rigidly to the training set, while not performing well enough during the test. So we drop some of the layers and add dropout to avoid overfit from happening. Figure 6 informs the layer in our neural network model.

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/cnn.PNG"  width="400">
 <br>
 <em>Figure 5 - Nvidia Model Architecture</em>
</p>

<p align="center">
 <img src="https://github.com/arief25ramadhan/udacity-behavioral-cloning/blob/main/model_sum.png"  width="400">
 <br>
 <em>Figure 6 - Our Model Architecture</em>
</p>


### 4.2. Hyperparameter Tuning and Selection 

Our model is build using Sequential API of Keras framework as show in model.py file. To sum up, we use ReLU activation function, Adam optimizer, and Mean Squared Error loss function. We train the model for 100 epochs, where we observed the model had reached an optimal value.

The model contains dropout layers and was trained and validated on different data sets to avoid overfitting. We tested the model by running it through the simulator, where our car must stay on track for at least one lap drive.

## 5. Conclusion

To conclude, we have trained a neural network models based on Nvidia on NVIDIA's End to End Learning for Self-Driving Cars paper (https://arxiv.org/pdf/1604.07316v1.pdf). Our model was able to make the car goes through the lap without ever going out of the track, as recorded in the video.mp4 file.
