
# coding: utf-8

# In[52]:

import os
import cv2
import numpy as np
import pandas as pd
from skimage import data
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, Activation, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split


# In[61]:

def get_training_data():
    train_images = []
    train_labels = []
    train_image_files = [x for x in os.listdir('../app/train_images/') if '.jpg' in x]
    for f in train_image_files:
        img = data.imread('../app/train_images/' + f)
        train_images.append(img)
        if 'DZ' in f:
            train_labels.append(1)
        else:
            train_labels.append(0)
    return np.array(train_images), np.array(train_labels)
    
    
X, y = get_training_data()
print X.shape, y.shape

pd.DataFrame(y)[0].value_counts()


# In[51]:

input_shape = (233, 70, 4)
num_classes = 2

# Resize images
def resize_images(imgs):
    imgs_resized = []
    for img in imgs:
        img_resized = cv2.resize(img, input_dim)
        imgs_resized.append(img_resized)
    return np.array(imgs_resized)

X_resized = resize_images(X)


# In[59]:

x_train, x_test, y_train, y_test = train_test_split(X_resized, y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[64]:

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:

history = model.fit(x_train, y_train,
          batch_size=50,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))


# In[ ]:



