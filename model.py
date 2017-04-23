import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Cropping2D

lines=[]
with open('C:/Users/derya/Desktop/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    
images=[]
measurements=[]
for line in lines:
    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    img_center = cv2.imread(line[0])
    img_left = cv2.imread(line[1])
    img_right = cv2.imread(line[2])


    images.append(img_left)
    images.append(img_center)
    images.append(img_right)

    measurements.append(steering_left)
    measurements.append(steering_center)
    measurements.append(steering_right)
    



X_train=np.array(images)
y_train=np.array(measurements)
print(X_train.shape)
print(y_train.shape)
model=Sequential()
model.add(Lambda(lambda x:x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,validation_split=0.2,shuffle=True, nb_epoch=5)
model.save('model-test.h5')