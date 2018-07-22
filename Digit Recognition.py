import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# Import Data
train = "../input/train.csv"
test = "../input/test.csv"
output = "submission.csv"
test = pd.read_csv(test)
data = np.loadtxt(train, skiprows=1, dtype='int', delimiter=',')

# Format data to be used in CNN (Reshape, normalize)
x_data = data[:,1:]
x_data = x_data.reshape(-1, 28, 28, 1).astype('float32')/255
y_data = to_categorical(data[:,0])
# x_train, x_test, y_train, y_test = train_test_split(
#     data[:,1:], data[:,0], test_size=0.1)
X_TEST = test.values.astype('float32')
X_TEST = X_TEST.reshape(X_TEST.shape[0], 28, 28,1)/255
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# x_train = x_train.astype("float32")/255.
# x_test = x_test.astype("float32")/255.
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# Construct CNN - Structure of Layers inspired by Peter Grenholm's kernel-------------------------------------
#--------------------https://www.kaggle.com/toregil------------------------------------------------------------
model = Sequential()

# Add convolution and pooling layers, normalize intermediate data and regularize to prevent overfitting
model.add(Convolution2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Convolution2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.25))

# Flatten output in order to feed to dense layer.
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#-----------------------------------------------------------------------------------------------------------------

# Use ImageDataGenerator for image augmentation and feed to model
# Additionally select optimizer and loss function (once again to prevent overfitting)
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                             shear_range = 0.3,
                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])

# Learning rate decreases with each epoch to ensure convergence
Lrate = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

# Fit model and predict results
batches = datagen.flow(x_data, y_data, batch_size=64)
hist = model.fit_generator(batches,
                           steps_per_epoch=batches.n,
                           epochs=4,
                           verbose = 1,
                           callbacks=[Lrate])

predictions = model.predict_classes(X_TEST, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                        "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)