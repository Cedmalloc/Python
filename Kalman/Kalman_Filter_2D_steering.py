import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]

options['DRIVE_IN_CIRCLE'] = True
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = False
options['RECIEVE_INPUTS'] = False

class KalmanFilter:
    def __init__(self):
        # Initial State
        self.x = np.matrix([[0.],   #new state vector
                            [0.],   #      x,y,v,theta,theta_dot
                            [0.],   # x = x + 
                            [0.],   # y = y +
                            [0.]])  # 


        self.u = np.matrix([[0.],
                            [0.],
                            [0.],    
                            [0.],
                            [0.]])
        # Uncertainity Matrix
        self.P = np.matrix([[100., 0., 0., 0.,0.],
                            [0., 100., 0., 0.,0.],
                            [0., 0., 100., 0.,0.],
                            [0., 0., 0., 100.,0.],
                            [0., 0., 0., 0.,100.]])

        # Next State Function
        self.A = np.matrix([[1., 0., 0., 0.,0.],
                            [0., 1., 0., 0.,0.],
                            [0., 0., 1., 0.,0.],
                            [0., 0., 0., 1.,0.],
                            [0., 0., 0., 0.,1.]])
        # Measurement Function
        self.H = np.matrix([[1.,0.,0.,0.,0.],
                            [0.,1.,0.,0.,0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[5.,0.],
                            [0.,5.]])

        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0.,0.],
                            [0., 1., 0., 0.,0.],
                            [0., 0., 1., 0.,0.],
                            [0., 0., 0., 1.,0.],
                            [0., 0., 0., 0.,1.]])
    #self.Q The process noise covariance matrix not implemented
    def predict(self, dt):
        self.A[0,2] = dt
        self.A[1,3] = dt
        self.A[3,4] = dt
        #self.x = self.A * self.x 
        #self.P = self.A * self.P * np.transpose(self.A)
        self.x = np.dot(self.A,self.x) 
        self.P = np.dot(np.dot(self.A,self.P),np.transpose(self.A))
        return 

    def measure_and_update(self,measurements, dt):
        self.A[0,2] = dt
        self.A[1,3] = dt
        self.A[3,4] = dt
        #Z = np.matrix(measurements)
        #y = np.transpose(Z) - (self.H * self.x)
        #S = self.H * self.P * np.transpose(self.H) + self.R
        #K =  self.P * np.transpose(self.H) * np.linalg.inv(S)
        #self.x = self.x + K * y
        #self.P = (self.I - (K*self.H))*self.P 
        Z = np.matrix(measurements)
        y = np.transpose(Z) - np.dot(self.H,self.x)
        S = np.dot(np.dot(self.H,self.P),np.transpose(self.H)) + self.R 
        K = np.dot(np.dot(self.P,np.transpose(self.H)),np.linalg.inv(S)) #Kalman gain
        self.x = self.x + np.dot(K,y)
        self.P = self.I - np.dot(np.dot(K,self.H),self.P) 
        #add some uncertainy, so the filter is not overconfident
        self.P[0,0] += 0.1
        self.P[1,1] += 0.1
        self.P[2,2] += 0.1
        self.P[3,3] += 0.1
        self.P[4,4] += 0.1
        return [self.x[0],self.x[1]]
        

    def receive_inputs(self, u_steer, u_pedal):
        #u = np.matrix([[0.],
         #              [0.] 
          #             [u_pedal]   
           #            [u_steer]])
        return 

sim_run(options,KalmanFilter)
