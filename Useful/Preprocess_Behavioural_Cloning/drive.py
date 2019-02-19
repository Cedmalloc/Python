import socketio
from flask import Flask #create web application and set up bidirectional connection
import eventlet
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()
#sockets are created for Realtime communication betwen client and server
app = Flask(__name__) #set to '_main_' when connected

vmax = 10 #speed limit
@sio.on('Prepocessing')
def Prepocessing(img):
  #crop image
  img = img[60:135,:,:]  #crop trees and scenery (irrelevant in terms of just predicting the steering angle)
  #132 is making sure to cut out the hood
  #relying on Nvidia Model : yuv color space required
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)   #there's a difference between cv2.COLOR_RGB2YUV and cv2.COLOR_BGR2YUV
  #Gaussian blur to reduce image noise
  img = cv2.GaussianBlur(img,  (3, 3), 0)
  #image = cv2.resize(image, (image.shape[1],old_size))
  img = cv2.resize(img, (200, 66))
  #normalization
  img = img/255
  return img

@sio.on('telemetry')
def telemetry(sid, data):
  speed = float(data['speed'])
  #listen to updates send to simulator from telemetry
  image = Image.open(BytesIO(base64.b64decode(data['image']))) #decode image
  #buffer to mimic data like a normal file
  image = np.asarray(image)
  image = Prepocessing(image)
  image = np.array([image]) #enclose to get another Dimension appropriate for model
  steering_angle = float(model.predict(image)) #steer based on current image
  throttle = 1.0 - speed/vmax
  print('Steering Angle : {} / Throttle value : {} / Speed : {}'.format(steering_angle, throttle, speed))
  control(steering_angle,throttle) #send to simulation


@sio.on('connect')  #3 event names are reserved disconnect, message
def connect(sid, environ):
     print('Connected')
     #control(0,1) #just go straight in the beginning to test
     control(0,0) #no movement initally


@sio.on('disconnect')  #3 event names are reserved disconnect, message
def disconnect(sid, environ):
     print('Disonnected')

def control(steering_angle,throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
    #data is handled as string by simulator, executed right as connection
    #is established

if __name__ == '__main__':
  model = load_model('model.h5')  #load Nvidia model
  #Middleware to dispatch traffic to socketio Server
  app = socketio.Middleware(sio,app)
  #webserver gateway interface to have webserver send requests by client to app
  eventlet.wsgi.server(eventlet.listen(('',4567)),app)  #first argument is ip address
  #second one is port
  #creates listening sockets
