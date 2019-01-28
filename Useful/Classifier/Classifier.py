import numpy as np
from matplotlib import pyplot as plt



def sigmoid(score):
    return (1/(1+np.exp(-score)))

def line(x1,x2):
    ln = plt.plot(x1,x2,'-')
    #plt.pause(0.0001)
    #ln[0].remove()

def gradient_descent(linear_param, points, y, alpha):
    num_iterations = 2000
    for i in range(num_iterations): # calcula
        p = sigmoid(points*linear_param)
        gradient = points.T * (p-y)* (alpha/points.shape[0])
        # calculate batch gradient (problematic for larger number of points)
#alpha is the learning rate, small error to ensure the steps aren't too big
        linear_param = linear_param - gradient
    #substracting gradient from [w1,w2,b]


        w1,w2,b = (linear_param[0:3,0])

        #w1 = linear_param.item(0)
        #w2 = linear_param.item(1)
        #b = linear_param.item(2)


        x1 = np.array([bottom[:,0].min(),top[:,0].max()])
#from x1*w1+x2*w2+b = 0 :
        x2 = -float(b)/float(w2) + (x1*(-float(w1)/float(w2)))
    line(x1,x2)



def calc_errors(line_parameters, points , y):     #calculate error of point with cross entropy
     p = sigmoid(points*line_parameters)
     cross_entropy =-(1/points.shape[0])*(np.log(p).T*y + np.log(1-p).T*(1-y))
        #logarithm according to cross entropy
        #average
        #
     return cross_entropy

pts = 100 #set number of points here
np.random.seed(0)
bias = np.ones(pts)
     #create random points
top = np.array([np.random.normal(10,2,pts), np.random.normal(12,2,pts), bias]).T
bottom = np.array([np.random.normal(5,2,pts), np.random.normal(6,2,pts), bias]).T
all = np.vstack((top,bottom))

w1 = -0.3 #weight 1
w2 = -0.35 #weight 2
b = 3.5 #bias
linear_param = np.matrix([w1, w2, b]).T #T to ensure Matrix multiplication with all
print(linear_param.shape)
x1 = np.array([bottom[:,0].min(),top[:,0].max()])
#from x1*w1+x2*w2+b = 0 :
x2= -b/w2 + (x1*(-w1/w2))
linear_comb = all* linear_param
prob = sigmoid(linear_comb)   #prob of point being positive
y = np.array([np.zeros(pts),np.ones(pts)]).reshape(pts*2,1) #array with labels, top region : 0, bottom region :1

_, ax = plt.subplots(figsize = (4,4))
ax.set_title('Classifier')
ax.scatter(top[:,0],top[:,1],color = 'r')
ax.scatter(bottom[:,0],bottom[:,1],color = 'b')
alpha = 0.06 #set learning rate here
gradient_descent(linear_param, all, y , alpha)
fig = plt.gcf()
fig.canvas.set_window_title('Classifier')
plt.show()
print(calc_errors(linear_param,all, y ))
