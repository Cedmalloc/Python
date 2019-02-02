
# coding: utf-8

# In[2]:


import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import random
from keras.datasets import mnist
from matplotlib import pyplot as plt


# In[109]:


np.random.seed(0)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()  #mnist dataset shape : (60000,28,28)
#assert statements to create warnings just in case
assert(X_train.shape[0] == Y_train.shape[0]), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == Y_test.shape[0]), "The number of images is not equal to the number of labels"
assert(X_train.shape[1:] == (28,28)), "The image is not of size 28x28"
assert(X_test.shape[1:] == (28,28)),  "The image is not of size 28x28"

num_of_samples = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (5,10) )
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[Y_train == j] #split data associated to label
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected-1)),:,:],cmap  = plt.get_cmap("gray"))
        axs[j][i].axis("off")  #turn coordinate system off
        if i == 2:
         axs[j][i].set_title(str(j)) #add class name
         num_of_samples.append(len(x_selected))  #to get amount of images per class


# In[7]:


#plt.figure(figsize = (12,4))
#plt.bar(range(0,num_classes),num_of_samples) #10 classes from 0 to 9, number of images per class: num_of_samples


# In[110]:


#one hot encoding
Y_train = to_categorical(Y_train,10)
Y_test = to_categorical(Y_test,10)
#normalization
X_train = X_train/255 #scale down features to a range of 0 to 1, lower variance helps better deal with input data
#large range between the values makes finding the right weights, biases and hyperparameters
X_test = X_test/255
num_pixels = 784  #784 = 28 * 28, flatten to 2D
X_train = X_train.reshape(X_train.shape[0],num_pixels)
X_test = X_test.reshape(X_test.shape[0],num_pixels)
#print(X_train.shape)
#print(X_test.shape)
assert(X_train.shape[1]  == 784 ), "The dimensions of X train aren't correct"
assert(X_test.shape[1]  == 784 ), "The dimensions of X test aren't correct"


# In[111]:


def create_model():
    model = Sequential()
    #which factors lead to best generalization ? 
    model.add(Dense(10, input_dim=num_pixels, activation='relu')) #relu is linear better for images
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[116]:


model = create_model()
#print(model.summary())  #prints summary of model, shows total parameters
#this architecture does not work with larger images, especially when we use all channels and not just one
h = model.fit(X_train, Y_train, validation_split = 0.1, epochs = 20, batch_size = 200, verbose = 1, shuffle = 1)
#batch_size 200 ,30 epochs: 2 hidden layers 10/10 Nodes => loss: 7.1089e-04 - acc: 0.9461 - val_loss: 7.6227e-04 - val_acc: 0.9463
#batch_size 200 ,10 epochs: 2 hidden layers 10/10 Nodes => loss: 8.6195e-04 - acc: 0.9346 - val_loss: 7.7828e-04 - val_acc: 0.9435
#batch_size 200 ,10 epochs: 3 hidden layers 10/30/10 Nodes => loss: 0.2719 - acc: 0.9176 - val_loss: 0.2533 - val_acc: 0.9245
#batch_size 200 ,20 epochs: 3 hidden layers 10/30/10 Nodes =>  loss: 0.1872 - acc: 0.9431 - val_loss: 0.1815 - val_acc: 0.9495


# In[117]:


plt.plot(h.history['loss'],label= 'Training Loss')
plt.plot(h.history['val_loss'],label= 'Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')


# In[118]:


score = model.evaluate(X_test,Y_test,verbose = 0)
print(type(score)) #returns list
print('Test score :',score[0])
print('Test accuracy :',score[1])


# In[119]:


import requests
import cv2
from PIL import Image
#download 
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url,stream = True)
img = Image.open(response.raw)
plt.imshow(img)
img_array = np.asarray(img) #converts to array
resized = cv2.resize(img_array,(28,28)) #resize to 28x28 so we can evaluate it
gray_img = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)  #convert to single color channel like in finding_lanes
#notice that the background is 255 and the number is 0 which has to be changed as the NN was trained for the opposite
image = cv2.bitwise_not(gray_img) #use bitwise operation to change colors
plt.imshow(image,cmap  = plt.get_cmap("gray"))
#same as before
image = image.reshape(1,784) #one row since it's only one image
prediction = model.predict_classes(image)
print('Prediction is:',prediction)

#prediction with 2 was not possible at first with 10 epochs, 20 epochs might lead to overfitting, therefore it might be able to recognize the number

