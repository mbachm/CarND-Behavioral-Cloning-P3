import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

PATH_TO_CSV = './data/driving_log.csv'
PATH_TO_IMAGES = './data/IMG/'

car_images = []
steering_angles = []
with open(PATH_TO_CSV, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        steering_center = float(row[3])
        
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center_filename = row[0].split('/')[-1]
        img_left_filename = row[1].split('/')[-1]
        img_right_filename = row[2].split('/')[-1]
        img_center = cv2.imread(PATH_TO_IMAGES + img_center_filename)
        img_left = cv2.imread(PATH_TO_IMAGES + img_left_filename)
        img_right = cv2.imread(PATH_TO_IMAGES + img_right_filename)

        # add images, flipped images and angles to data set
        car_images.extend([img_center, img_left, img_right, np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
        steering_angles.extend([steering_center, steering_left, steering_right, -steering_center, -steering_left, -steering_right])


print('Read all images and steering angels')

def normalization():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0)))) # image shape = 60, 320
    model.add(Lambda(lambda image: K.tf.image.resize_images(image, (66, 200))))
    return model


def nvidia_model():
    model = normalization()
    model.add(Convolution2D(24,5,5, subsample=(2, 2), border_mode="valid", init='he_normal', activation="elu"))
    model.add(Convolution2D(36,5,5, subsample=(2, 2), border_mode="valid", init='he_normal', activation="elu"))
    model.add(Convolution2D(48,5,5, subsample=(1, 1), border_mode="valid", init='he_normal', activation="elu"))
    model.add(Convolution2D(64,3,3, subsample=(1, 1), border_mode="valid", init='he_normal', activation="elu"))
    model.add(Convolution2D(64,3,3, subsample=(1, 1), border_mode="valid", init='he_normal', activation="elu"))
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal', activation="elu"))
    model.add(Dense(100, init='he_normal', activation="elu"))
    model.add(Dense(50, init='he_normal', activation="elu"))
    model.add(Dense(10, init='he_normal', activation="elu"))
    model.add(Dense(1, init='he_normal'))
    return model

model = nvidia_model()
model.compile(loss='mse', optimizer='adam')


X_train = np.array(car_images)
y_train = np.array(steering_angles)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

print('save model')
model.save('model.h5')
print('model saved')


""" Plot visalisation:
    
    import keras.models import Model
    import matplotlib.pyplot as plt
    
    history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=5, verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
    """
