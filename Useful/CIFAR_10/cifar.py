import tensorflow as tf
from PIL import Image
#'/Users/ceddy/Pictures/Screenshots/Yay.png'
from keras.datasets import cifar10
import random
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
import h5py
import numpy as np

# image = input('Input Image Path : ')
# display = Image.open(image)
# display.show()

labels = ['airplane','automobile','bird','cat','deer','dog','frog',
'horse','ship','truck']

(x_train , y_train), (x_test,y_test) = cifar10.load_data()
print(y_train[0])
index = random.randint(0,len(x_train)-1)
display_image = x_train[index]


col_image = Image.fromarray(display_image) #Creates an image memory from an object exporting the array interface (using the buffer protocol).
r, g, b = col_image.split()
plt.imshow(r, cmap = "Reds")
plt.title(str(labels[y_train[index][0]]))
plt.show()

new_x_train = x_train.astype('float32')
new_x_test = y_test.astype('float32')
#convert to between 0 and 1
new_x_train /= 255
new_x_test /= 255
#encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# POSSIBLE IMPROVEMENT USE OF VALIDATION SET BY SPLITTING TRAINING DATA
# sklearn.model_selection.train_test_split

#
#  X_train, X_val, y_train, y_val
#     = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

def CNN():
    model = Sequential()
    model.add(Conv2D(32, (3,3),input_shape = (32,32,3), activation = 'relu', padding = 'same',kernel_constraint=maxnorm(3)))
    #padding = same to add padding so we do not loose the edge information
    #maxnorm(m) will, if the L2-Norm of your weights exceeds m, scale your whole weight matrix by a factor that reduces the norm to m
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(SGD(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = CNN()
h = model.fit(new_x_train, y_train, epochs = 10, batch_size = 32, verbose = 1, shuffle = 1 )
#model.save('CIFAR_model.h5')

#testing
input_im = ('Enter image Path:')
im = Image.open(input_im)
im= im.resize((32,32),resample = Image.LANCZOS) #resize and resample
im_array = np.array(im)
im_array = im_array.asarray('float32')
im_array = im_array/255 #0 and 1
im_array = im_array.reshape(1,32,32,3)
result = model.predict(im_array)
input_im.show()
print(labels[np.argmax(answer)])
