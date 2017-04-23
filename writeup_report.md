# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/2017_04_20_23_30_06_314.jpg "Center Lane Driving"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video capture of an autonomous run

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of Nvidia's neural network architecture consisting of convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 47-51) 

The model includes direct connection layers to bring-in nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 58).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Combining that 
with appropriate angle measurements added assisted in overall training approach.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Lenet. I thought this model might be appropriate because problem Nvidia team was
attempting to solve was in many aspects similar to what we are handling here.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to have added layers of non-linearity however these efforts did not result in a successful predictive system.

Then I changed architecture to Nvidia self-driving team's multi level convolutional system which immediately started to show progress. Additionally I introduced normalization and cropping techniques to reduce overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ran additional tracks with special emphasis on hard to navigate sections of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
* Convolutional layer of size (24,5,5) with Stride of 2 with relu activation
* Convolutional layer of size (36,5,5) with Stride of 2 with relu activation
* Convolutional layer of size (48,5,5) with Stride of 2 with relu activation
* Convolutional layer of size (64,3,3) with Stride of 2 with relu activation
* Convolutional layer of size (64,3,3) with Stride of 2 with relu activation
* Flattening 
* Direct Connection Layer of 100
* Direct Connection Layer of 50
* Direct Connection Layer of 10
* Direct Connection Layer of 1




#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]


After the collection process, I had about 1,100 number of data points. Subsequently, I added right and left camera images with corrected steering values to the data set reaching 8,935. I then preprocessed this data by normalizing via Lambda layer as well as cropping the 25 pixels from bottom (steering wheel and hood) as well as 75 pixels from top (trees and hilltops with sky). 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by quick convergence on loss and continued oscillations around the mean after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.