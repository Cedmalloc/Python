
# coding: utf-8

# In[16]:


from sympy import Symbol
from sympy.solvers import solve
from matplotlib import pyplot as plt
import numpy as np





m = 1620;
fr = 0.012;
c = 0.33;
r = 0.308;
A = 2.2;
rho =1.23;
g = 9.81;

x = Symbol('x');

eqn = (((83.8e+3-70.7e+3)/(61.6-46.2))*(x-46.2)+70.7e+3)*0.85-(m*g*fr+c*rho*0.5*A*x*x)*x;
print(solve(eqn,x))



# In[61]:


y1 =[]
z = []
for x in range(0,70):
   z.append(x*3.6)
   y1.append( ((((83.8e+3-70.7e+3)/(61.6-46.2))*(x-46.2)+70.7e+3)*0.85))
   y2.append((m*g*fr+c*rho*0.5*A*x*x)*x)
   

plt.plot(z,y1,color= 'red', label= 'Fahrwiderstandskurve')
plt.plot(z,y2,color= 'blue', label= 'Antriebskraftskurve')
plt.legend()
plt.xlabel('Geschwindigkeit in km/h')
plt.ylabel('Kraft')

del y2[:]
del z[:]
del y1[:]

