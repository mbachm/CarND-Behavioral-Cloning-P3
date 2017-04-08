import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

### Local path to the provided udacity training data for P3
PATH_TO_CSV = './example_data/driving_log.csv'
PATH_TO_IMAGES = './example_data/IMG/'

### Correction angle for left and right images. Will be add to left and subtracted from right image
ANGLE_OFFSET = 0.25

### Read all images and steerings into memory as I didn't need a generator
### With this I am able to use a validation split function
car_images = []
steering_angles = []
with open(PATH_TO_CSV, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        ### steering angles for left and right image
        steering_center = float(row[3])
        steering_left = steering_center + ANGLE_OFFSET
        steering_right = steering_center - ANGLE_OFFSET

        ### read in images from center, left and right camera and change color space to RGB
        img_center_filename = row[0].split('/')[-1]
        img_left_filename = row[1].split('/')[-1]
        img_right_filename = row[2].split('/')[-1]
        srcBGR = cv2.imread(PATH_TO_IMAGES + img_center_filename)
        img_center = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
        srcBGR = cv2.imread(PATH_TO_IMAGES + img_left_filename)
        img_left = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
        srcBGR = cv2.imread(PATH_TO_IMAGES + img_right_filename)
        img_right = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)

        ### add image and steering to training list
        car_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])
        
        ### add flipped image and flipped steering to training list
        car_images.extend([np.fliplr(img_center),np.fliplr(img_left),np.fliplr(img_right)])
        steering_angles.extend([-steering_center, -steering_left, -steering_right])


print('Read all images and steering angels')

def normalization():
    """ Normalize and crops unnecessary parts of the image """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    return model


def nvidia_model():
    """ Build the nvidia model with keras 
        Further reading at:
        https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    dropout_prob = 0.25
    
    model = normalization()
    model.add(Convolution2D(24,5,5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(1, 1), activation="relu"))
    model.add(Convolution2D(64,3,3, subsample=(1, 1), activation="relu"))
    model.add(Convolution2D(64,3,3, subsample=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(dropout_prob))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(dropout_prob))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(dropout_prob))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(dropout_prob))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

### Build model, set loss function to 'mse' and use the adam optimizer.
model = nvidia_model()
model.compile(loss='mse', optimizer='adam')

### Convert the training and validation data to numpy arrays and use the Keras fit function with a validation split of 0.2 for training. Number of epochs is 5
X_train = np.array(car_images)
y_train = np.array(steering_angles)
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training.png')

### Save the model
model.save('model.h5')
