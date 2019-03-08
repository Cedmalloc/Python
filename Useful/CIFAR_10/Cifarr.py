from keras.datasets import cifar10
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
import h5py
#load.data
(x_train , y_train ), (x_test, y_test) = cifar10.load_data()
# index = random.randint(0,len(x_train)-1)
# plt.imhow(x_train[index])
#max = np.argmax(x_train[0])
labels = ['airplane','automobile','bird','cat','deer','dog','frog',
'horse','ship','truck']

#hot encode
#just for flattening
#def reshape_array(input_array):
 # output = []
  #for im_array in input_array:
   #     output.append(np.ravel(im_array))

  #return np.asarray(output)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255



def CNN():
    model = Sequential()
    model.add(Conv2D(32, (3,3),input_shape = (32,32,3), activation = 'relu', padding = 'same',kernel_constraint=maxnorm(3)))
    #padding = same to add padding so we do not loose the edge information
    #maxnorm(m) will, if the L2-Norm of your weights exceeds m, scale your whole weight matrix by a factor that reduces the norm to m
    #normalize everything to 3
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(SGD(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
   
   
   def CNN_V3():
  model = Sequential()
  model.add(Conv2D(32,(3,3),input_shape= (32,32,3),activation = 'relu',padding = 'same')) #30 5x5 filters applied
  #model.add(Dropout(0.5))
  #num of parameters : (kernel_size*stride+1)*filters
  model.add(Conv2D(32,(3,3),activation = 'relu',padding = 'same'))
  #scaled down to 28x28
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(100,(3,3),activation = 'relu',padding = 'same'))                  #doubled amount of filters from 30/15 to 60/60/30/30
  #model.add(Dropout(0.5))
  model.add(Conv2D(100,(3,3),activation = 'relu',padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))
  model.add(Flatten()) #flatten to input in fully connected layer
  model.add(Dense(2500,activation = 'relu'))  
  model.add(Dropout(0.5))
  model.add(Dense(1000,activation = 'relu'))  
  model.add(Dropout(0.5))
  model.add(Dense(500,activation = 'relu')) 
  model.add(Dropout(0.5))
  model.add(Dense(100,activation = 'relu'))  
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation = 'softmax'))
  model.compile(Adam(lr=0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])  #lr reduced from 0.01 by factor 10
  return model
   
   
model = CNN()
h = model.fit(x_train, y_train, epochs = 30 , batch_size = 32, shuffle = 1, verbose = 1);
model.save(filepath = 'Image.classifier.h5')
score = model.evaluate(x_train, y_train, verbose = 0)

print("Score is : %s",score[1])
im_data = np.asarray(x_train[0])
prediction = model.predict(im_data)
np.argmax(prediction[0])
