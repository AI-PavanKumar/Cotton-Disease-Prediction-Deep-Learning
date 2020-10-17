#Import libraries
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.models import load_model
#re-size all the images
IMAGE_SIZE=[224,224]
train_path='data/train'
valid_path='data/val'

#Import the pretrained InceptionV3 model and preprossing layers
inception=InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#Don't train exisiting weighyts
for layer in inception.layers:
    layer.trainable=False
#Get no of output classes
folders=glob('data/train/*')

#add our layers if required
x=Flatten()(inception.output)
prediction=Dense(len(folders), activation='softmax')(x)

#Create a Model object
model=Model(inputs=inception.inputs, outputs=prediction)
#get Model Summary
model.summary()
#cost and optimization method to use
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Image Data Generator to import the images from the dataset
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
# Make sure you provide the same target size as initialied for the image size
train_set=train_datagen.flow_from_directory('data/train',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
test_set=test_datagen.flow_from_directory('data/val',
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode='categorical'
                                          )
# fit the model
r=model.fit_generator(train_set,
                      validation_data=test_set,
                      epochs=20,
                      steps_per_epoch=len(train_set),
                      validation_steps=len(test_set))
#Save model
model.save('Inception.h5')



