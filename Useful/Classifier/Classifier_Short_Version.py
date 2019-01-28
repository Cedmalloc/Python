import numpy as np
import keras
from keras.models import Sequential #= linear stack of layers
from keras.layers import Dense #every node from one layer connected to every node in next layer
from keras.optimizers import Adam # adaptive optimization algorithm 
#stochastic gradient descent
#computes adaptive learning rates
#srecommended algorithm to be used
import matplotlib.pyplot as plt

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

plt.plot(hist.history['acc'],label = 'accuracy')
plt.plot(hist.history['loss'],label = 'loss')
plt.title('accuracy and loss ')
plt.xlabel('Epoch')
plt.legend(loc = 'upper right')
