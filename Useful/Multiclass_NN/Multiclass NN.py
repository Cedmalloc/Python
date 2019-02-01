
# coding: utf-8

# In[2]:


import numpy as np
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from sklearn import datasets


# In[7]:


from keras.utils.np_utils import to_categorical #allows us to reformat data to one hot encoded form


# In[37]:


num = 500
center_of_blops = [[-1,1],[-1,-1],[1,-1],[1,1],[0,0],[2,2]]  #specify centers of blob areas      
X ,Y = datasets.make_blobs(n_samples = num, random_state = 123, centers= center_of_blops,cluster_std = 0.4 )
#cluster_std is standard deviation of data, increase to make harder to seperate data
plt.scatter(X[Y==0,0],X[Y==0,1],color = 'orange')  
plt.scatter(X[Y==1,0],X[Y==1,1],color = 'green') 
plt.scatter(X[Y==2,0],X[Y==2,1],color = 'blue') #3rd class new now compared to earlier binary sets 
#later on : more classes added
plt.scatter(X[Y==3,0],X[Y==3,1],color = 'magenta')  
plt.scatter(X[Y==4,0],X[Y==4,1],color = 'yellow') 
#plt.scatter(X[Y==5,0],X[Y==5,1],color = 'navy')
       


# In[41]:


y_cat = to_categorical(Y, 5 ) #2nd argument can be left blank when number of classes is 3, otherwise input number of classes
#print(y_cat)
#outputs one hot encoded form of our labeled data
# 0 => 1 0 0
# 1 => 0 1 0
#2  => 0 0 1
model = Sequential()
model.add(Dense(units = 5 , input_shape= (2,), activation = 'softmax')) #sigmoid is not able to solve this multiclass problem
model.compile(Adam(lr = 0.01),loss = 'categorical_crossentropy', metrics = ['accuracy'])
h = model.fit (x = X, y = y_cat, verbose = 1, batch_size = 50, epochs = 50 )
#note that number of epochs should be around the amount when it reaches maximum accuracy, so we avoid under or overfitting
#number of epochs depends on dataset and has to be tested out 


# In[40]:


def plot_decision_boundary(X,Y,model):
    x_span = np.linspace(min(X[:,0]),max(X[:,0]))
    y_span = np.linspace(min(X[:,1]),max(X[:,1]))
    xx , yy = np.meshgrid(x_span, y_span)
    #xx_ , yy _ = np.ravel(xx), np.ravel(yy_)
    grid = np.c_[np.ravel(xx),np.ravel(yy)]
    pred_func = model.predict_classes(grid).reshape(xx.shape) #now classes function is not just predict(not binary set)
    plt.contourf(xx,yy,pred_func)
    
plot_decision_boundary(X,y_cat,model)
plt.scatter(X[Y==0,0],X[Y==0,1],color = 'orange')  #0
plt.scatter(X[Y==1,0],X[Y==1,1],color = 'red')    #1
plt.scatter(X[Y==2,0],X[Y==2,1],color = 'Aqua')  #2
plt.scatter(X[Y==3,0],X[Y==3,1],color = 'magenta') #3  
plt.scatter(X[Y==4,0],X[Y==4,1],color = 'yellow')  #4
#purple zone is class 0
#green zone is class 1
# yellow zone is class 2  
##=> for 3 classes
x = 0.5
y = 0.5
point = np.array([[x,y]])

plt.plot(point[0][0],point[0][1],marker = 'o', markersize = 10, color = 'Lime')
pred = model.predict_classes(point)
print("Prediction is: " ,model.predict_classes(point))

