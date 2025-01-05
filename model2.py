import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np 
import pandas as pd 
import tensorflow as tf 
# Here we are going to make a neural net that will differentiate between the pictures of a cat and dog
from tensorflow import keras

from tensorflow.keras.layers import Input,Conv2D, Dense, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist 
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_train = x_train/255.0
x_test = x_test/255.0

model = Sequential([
Input(shape=(28,28,1)),
Conv2D(64,kernel_size=3,activation='relu'),
MaxPooling2D(pool_size=2),
Conv2D(128,kernel_size=3,activation='relu'),
MaxPooling2D(pool_size=2),
Flatten(),
Dense(10,activation='softmax')
	])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test),verbose=1)
model.evaluate(x_test,y_test,verbose=1)

from PIL import Image
image_path = "D:/Media Files/image.png"
image = Image.open(image_path).convert("L")
image = image.resize((28,28),Image.Resampling.LANCZOS)

image_array = np.array(image)/255.0
image_array = image_array.reshape(-1,28,28,1)


predicted = model.predict(image_array)
predicted_class = np.argmax(predicted)
print(f"Predicted Class:{predicted_class}")
