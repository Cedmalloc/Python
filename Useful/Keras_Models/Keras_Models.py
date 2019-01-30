
# coding: utf-8

# In[1]:


import numpy as np
import keras
from keras.models import Sequential #= linear stack of layers
from keras.layers import Dense #every node from one layer connected to every node in next layer
from keras.optimizers import Adam # adaptive optimization algorithm 
#stochastic gradient descent
#computes adaptive learning rates
#srecommended algorithm to be used
import matplotlib.pyplot as plt


# In[2]:


n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T
 
X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
 
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])


# In[3]:


model = Sequential()
model.add(Dense(units=1,input_shape = (2,),activation = 'sigmoid'))  #add layers to NN
#only 2 inputs (x1,x2)
adam = Adam(lr = 0.1) #minimize error to 0.1
#configure learnung process
model.compile(adam, loss='binary_crossentropy', metrics = ['accuracy'])
#pass in training data, pass in labels assigned to data points :
hist = model.fit(x=X,y=y,verbose=1,batch_size=50,epochs = 10, shuffle = 'true') 
#0 top region, 1 bottom region
# it's important to pass in labels to know if we classified correctly
#batch_size* iterations to get to number of points = n_pts
#amount of epochs is important to avoid overfitting or underfitting
#shuffle makes sure to not get stuck in local minimum by shuffling subsets


# In[4]:


plt.plot(hist.history['acc'],label = 'accuracy')
plt.plot(hist.history['loss'],label = 'loss')
plt.title('accuracy and loss ')
plt.xlabel('Epoch')
plt.legend(loc = 'upper right')


# In[17]:


def plot_decision_boundary (X,y,model):
    #num = 50
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span) #convert to new matrix
    #flatten xx and yy 
    xx_, yy_ = xx.ravel(), yy.ravel() 
    grid = np.c_[xx_, yy_] #short-hand for np.r_['-1,2,0', index expression],
    #arrays stacked along last axis
    pred_fct = model.predict(grid) #trained model test all the points, 
    # returns probabilities
    z = pred_fct.reshape(yy.shape) #reshape to make compatible with grid parts
    plt.contourf(xx,yy,z)  #careful contourf not contour 
    
plot_decision_boundary(X,y,model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 10
y = 10


point = np.array([[x,y]])
prediction = model.predict(point)
plt.plot([x],[y],marker = "o", markersize = 10, color = 'black')
print("Prediction is:",prediction)

