import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR = ".\\detected_lines"
CATEGORIES = ["w", "a", "d"]
IMG_SIZE = 80

min_images_number = 99999

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    image_number = len(os.listdir(path))
    if image_number < min_images_number:
        min_images_number = image_number
        
print (min_images_number)



def create_training_data():
    tr_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        
        class_num = CATEGORIES.index(category)
#         print (class_num)
        
        label = [0, 0, 0]
        label[class_num] = 1
        images_list = os.listdir(path)
        for i in range(min_images_number):
            img = images_list[i]
#             print (img)
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                tr_data.append([new_array, label])
            except Exception as e:
                print (e)
    random.shuffle(tr_data)
    return tr_data

training_data = create_training_data()

X = []
y = []

for img, label in training_data:
    X.append(img)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X[0].shape)
y = np.array(y)

# print(y.shape)

X = X / 255.0

model = Sequential()
model.add(Conv2D(256, (7, 7), input_shape = X[0].shape, activation = "relu", padding = 'same'  )   )
# model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, (3, 3), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(3, activation = "softmax"))

model.summary()

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1)

NAME = "cnn_model_imgsize" + str(IMG_SIZE) + "-" + str(int(time.time()))
# model.save(NAME + ".h5")