from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))
model.compile(SGD(lr=0.001), loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.fit(x_train, y_train , epochs = ..., verbose = 1, shuffle=1, batch_size = ...)
model.evaluate()
model.predict()
