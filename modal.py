 

import numpy as np # linear algebra
import pandas as pd  #dataprocessing
  
#importing the model and layers

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

model=Sequential()

#adding the layers,activation functions and kerner initializers

model.add(Convolution2D(32,(3,3), input_shape=(64,64,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(Flatten())

model.add(Dense(32,input_dim=64,kernel_initializer='uniform',activation='relu'))

model.add(Dense(1,input_dim=1,activation='sigmoid',kernel_initializer='uniform'))

#getting the summary of our model whether it gets overloaded or not

model.summary()

#image preprocessing

from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

#training the data from Kaggle datasets of X-ray

x_train=train_data.flow_from_directory('../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train')
x_test=train_data.flow_from_directory('../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test')

#compiling the dataset

model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])

#fitting the model

model.fit_generator(x_train,steps_per_epoch=163,epochs=10,validation_data=x_test,validation_steps=20)

#save the model

model.save("pcnn.h5")
