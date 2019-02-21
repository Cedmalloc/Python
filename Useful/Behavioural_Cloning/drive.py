import socketio
from flask import Flask #create web application and set up bidirectional connection
import eventlet

sio = socketio.Server()
#sockets are created for RT communication betwen client and server
app = Flask(__name__) #'_main_'

@sio.on('connect')  #3 event names are reserved disconnect, message
 def connect(sid, environment):
     print('connected')
if __name__ == '__main__':
  app.run(port = 3000)
  #Middleware to dispatch traffic to socketio Server
  app = socketio.Middleware(sio,app)
  #webserver gateway interface to have webserver send requests by client to app
  eventlet.wsgi.server(eventlet.listen(('',4567)),app)  #first argument is ip address
  #second one is port
  #creates listening sockets
