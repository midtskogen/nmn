#!/usr/bin/env python3

# Importing the Keras libraries and packages
from keras.models import Sequential,load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
import numpy as np

if True:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 128, input_dim=60, kernel_initializer='normal', activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./127,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = False)
    test_datagen = ImageDataGenerator(rescale = 1./127)
    training_set = train_datagen.flow_from_directory('dataset/training_set2',
                                                     target_size = (64, 64),
                                                     batch_size = 1,
                                                     class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('dataset/test_set2',
                                                target_size = (64, 64),
                                                batch_size = 1,
                                                class_mode = 'binary')
    model.fit_generator(training_set,
                             steps_per_epoch = 3082,
                             epochs = 50,
                             validation_data = test_set,
                             validation_steps = 2000)
    model.save('model.h5')
else:
    model = load_model('model.h5') 

import numpy as np
import os
from keras.preprocessing import image

dir = 'dataset/test_set2/wrongs/'
print('')
print('Incorrectly identified as meteors:')
for filename in os.listdir(dir):
    import sys
    test_image = image.load_img(dir + filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        sys.stdout.write(dir + filename + ' ')
print('')
    
dir = 'dataset/test_set2/xrights/'
print('Incorrectly identified as not meteors:')
for filename in os.listdir(dir):
    import sys
    test_image = image.load_img(dir + filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] != 1:
        sys.stdout.write(dir + filename + ' ')
print('')
