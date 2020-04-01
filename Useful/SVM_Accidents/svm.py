import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14]],dtype=np.float64)
y = np.array([5091,4949,4477,4152,3648,4009,3600,3339,3377,3459,3206,3180,3275,3059],dtype=np.float64)

x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]],dtype=np.float64)
y = np.array([4009,3600,3339,3377,3459,3206,3180,3275,3059],dtype=np.float64)


clf_poly = SVR(kernel='poly',degree=2,C=1e3)
clf_lin = SVR(kernel='linear',C=1e3)
clf_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)

clf_poly.fit(x,y)
print(clf_poly.score(x, y))
clf_lin.fit(x,y)
print(clf_lin.score(x, y))
clf_rbf.fit(x,y)
print(clf_rbf.score(x, y))

x1 = clf_rbf.predict([[10]])
x2 = clf_rbf.predict([[11]])
x3 = clf_rbf.predict([[12]])

x11 = clf_poly.predict([[10]])
x22 = clf_poly.predict([[11]])
x33 = clf_poly.predict([[12]])


print(x11,x22,x33)
print(x1,x2,x3)
years = np.array([[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013],[2014],[2015],[2016],[2017],[2018],[2019],[2020]],dtype=np.float64)
print(years.shape)
fatalities1 = np.array([5091,4949,4477,4152,3648,4009,3600,3339,3377,3459,3206,3180,3275,3095,x1],dtype=np.float64)
fatalities2 = np.array([5091,4949,4477,4152,3648,4009,3600,3339,3377,3459,3206,3180,3275,3095,x11],dtype=np.float64)
pred1 = np.array([x1,x2,x3],dtype=np.float64)
pred2 = np.array([x11,x22,x33],dtype=np.float64)
year_pred = np.array([[2020],[2021],[2022]],dtype=np.float64)

l1, = plt.plot(years, fatalities1, '-o',label = 'Aktuelle Zahlen')
plt.plot(years, fatalities2, '-o')
l2, = plt.plot(year_pred, pred1, '-o',label='Prädiktion SVM mit RBF Kernel')
l3, = plt.plot(year_pred, pred2, '-o',label='Prädiktion SVM mit Poly Kernel von Grad2')
plt.legend()
plt.xlabel('Jahr')
plt.ylabel('Verkehrstote')
plt.title('Verkehrstote pro Jahr in Deutschland')
plt.show()
