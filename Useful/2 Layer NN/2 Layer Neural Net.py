
# coding: utf-8

# In[1]:


import numpy as np
import keras
from keras.optimizers import Adam 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import datasets




# In[11]:


np.random.seed(0)
num = 500
X , y =  datasets.make_circles(n_samples = num, random_state = 123,noise = 0.1,factor=0.2) #factor determines inner circle size relative to outer circle
#rand. state : determines random number generation for dataset shuffling and noise
#Pass an int for reproducible output across multiple function calls
#print(X)
#print(y)
plt.scatter(X[y==0,0],X[y==0,1],color='green')
plt.scatter(X[y==1,0],X[y==1,1])


# In[12]:


model = Sequential()
model.add(Dense(4,input_shape=(2,),activation = 'sigmoid'))
model.add(Dense(1,activation = 'sigmoid'))   #add 2nd layer with one perceptron
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])
#model.compile(Adam(lr = 0.1), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X,y=y,verbose = 1, batch_size = 20, epochs = 100, shuffle ='true')
#recall that verbose just shows progress


# In[13]:


plt.plot(h.history['acc'],color = 'blue',label='accuracy')
plt.plot(h.history['loss'],color = 'green',label = 'loss')
plt.legend(loc = 'center right')


# In[15]:


def plot_decision_boundary(X,Y,model):
    x_span = np.linspace(min(X[:,0])-0.25,max(X[:,0])+0.25)
    y_span = np.linspace(min(X[:,1])-0.25,max(X[:,1])+0.25)
    
    xx ,yy = np.meshgrid(x_span, y_span)
    xx_ = xx.ravel()
    yy_ = yy.ravel()
    grid = np.c_[xx_,yy_]
    pred_fct = model.predict(grid)
    z = pred_fct.reshape(xx.shape)
    plt.contourf(xx,yy,z)

plot_decision_boundary(X, y, model)

plt.scatter(X[:,0],X[:,1])
x = 0.5
y = 0.5
point = np.array([[x,y]])
plt.plot([x],[y],marker = 'o',markersize=10,color = 'red')
print('Prediction is : ',model.predict(point))

