# -*- coding: utf-8 -*-
"""Traffic Signs Starter Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YW_g-uuM6AQtr8_WZLiNY7D7INLYtMKa
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import random

!git clone https://bitbucket.org/jadslim/german-traffic-signs  #import German traffic sign set from bitbucket by cloning 
!ls german-traffic-signs  #shows us that 1 csv and three pickled files are in this path

np.random.seed(0)

with open('german-traffic-signs/train.p','rb') as f:   #closes file automatically for us // 'rb' because we are reading from the file
  train = pickle.load(f)  #unpickled data

with open('german-traffic-signs/valid.p','rb') as f:
  val = pickle.load(f)  #unpickled data

with open('german-traffic-signs/test.p','rb') as f:
  test = pickle.load(f)  #unpickled data
  
#print(type(train)) shows us that it is a dicitionary
x_train, y_train = train['features'] ,train['labels']  #labels for classes, features = image data
x_test, y_test = test['features'] ,test['labels']
x_val, y_val = val['features'] ,val['labels']

#print(x_train.shape) #(34799, 32, 32, 3)
#print(x_test.shape)  #(12630, 32, 32, 3)
#print(x_val.shape)   #(4410, 32, 32, 3)


assert(x_train.shape[0]==y_train.shape[0]),   "Amount of Training images not equal to number of labels"
assert(x_test.shape[0]==y_test.shape[0]),   "Amount of Testing images not equal to number of labels"
assert(x_val.shape[0]==y_val.shape[0]),   "Amount of Validation images not equal to number of labels"
assert(x_train.shape[1:]==(32,32,3)),     "Training image is not the correct size or depth"
assert(x_test.shape[1:]==(32,32,3)),     "Training image is not the correct size or depth"
assert(x_val.shape[1:]==(32,32,3)),     "Training image is not the correct size or depth"

#old code modified

data =  pd.read_csv('german-traffic-signs/signnames.csv')  #store signnames as data frame
#data.replace('No vechiles' , 'vehicles')
#just fixing misspelled data
wrong_Signnames = [ "No passing for vehicles over 3.5 metric tons","No Vehicles","Vechiles over 3.5 metric tons prohibited","End of no passing by vehicles over 3.5 metric ..."]
data.loc[10,["SignName"]] = wrong_Signnames[0]
data.loc[15,["SignName"]] = wrong_Signnames[1]
data.loc[16,["SignName"]] = wrong_Signnames[2]
data.loc[42,["SignName"]] = wrong_Signnames[3]


num_of_samples = []
 
cols = 5
num_classes = 43 #now 43 classes compared to 10 with MNIST
 
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
fig.tight_layout()  #improves layout (plots with enough distance between them)
for i in range(cols):
    for j,row in data.iterrows():     #let's us iterate over data as pairs of :(index,series), row to iterate over series (1D array with data (name, class id,...))
        x_selected = x_train[y_train == j]    
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title("Class Number " + " "+ str(j)+ ":" + row["SignName"])
            num_of_samples.append(len(x_selected))

import cv2
 
#plt.imshow(x_train[800])
#plt.axis("off")
print(x_train[800].shape)
print(y_train[800])
def grayscale(img):           #convert to grayscale to reduce channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
  
gray_img = grayscale(x_train[800])
#plt.imshow(gray_img)
#plt.axis("off")

#perform histogram equalization 
#normalizes lighting, "spreads it at the ends"
#enhances contrast
#flattens histogram by reassigning grayscale values

def hist_equalization(image):
  image = cv2.equalizeHist(image)
  return image

equalized = hist_equalization(gray_img)

plt.imshow(equalized)
plt.axis("off")

def preprocessing(image):
  image = grayscale(image)
  image = hist_equalization(image)
  image = image/255                #divide by 255 to normalize for smaller variation of pixels (between 0 and 1)
  return image 

x_train = np.array(list(map(preprocessing,x_train)))  #map iterates through entire array and then passes it through specified function
x_test = np.array(list(map(preprocessing,x_test)))    #same for test
x_val = np.array(list(map(preprocessing,x_val)))      #same for validation

#reshape to add depth to process in CNN
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
#hot encoding
y_train = to_categorical(y_train,43) 
y_test = to_categorical(y_test,43) 
y_val = to_categorical(y_val,43)

def lenet_model():
  model = Sequential()
  model.add(Conv2D(30,(5,5),input_shape= (32,32,1),activation = 'relu')) #30 5x5 filters applied
  #scaled down to 28x28
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(15,(3,3),activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten()) #flatten to input in fully connected layer
  model.add(Dense(500,activation = 'relu'))  
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation = 'softmax'))
  model.compile(Adam(lr=0.01),loss = 'categorical_crossentropy',metrics = ['accuracy'])
  return model

model = lenet_model()
print(model.summary())

h = model.fit(x_train ,y_train,epochs = 15, validation_data = (x_val,y_val),batch_size=400,verbose = 1,shuffle = 1)

plt.plot(h.history['acc'],label = 'acc')
plt.plot(h.history['val_acc'],label = 'val_acc')
plt.legend()
plt.xlabel('Epoch')

score = model.evaluate(x_test,y_test,verbose = 0)
print('Test score : ',score[0])
print('Test Accuracy : ',score[1])  #just about 0.9