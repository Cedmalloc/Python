import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilter:
    def __init__(self):
        self.v = 0
        self.prev_time = 0
        # Initial State
        self.x = np.matrix([[0.],
                            [3.]])

        # Uncertainity Matrix
        self.P = np.matrix([[10000., 0.],
                            [0., 10000.]])

        # Next State Function
        self.F = np.matrix([[1., 0.],
                            [0., 1.]])

        # Measurement Function
        self.H = np.matrix([[1., 0.]])  #x is measured

        # Measurement Uncertainty
        self.R = np.matrix([[0.01]])

        # Identity Matrix
        self.I = np.matrix([[1., 0.],
                            [0., 1.]])
    def predict(self,t):
        # Calculate dt.
        dt = t - self.prev_time
        # Put dt into the state transition matrix.
        self.F[0,1] = dt
        self.x = self.F * self.x
        self.P = self.F * self.P * np.transpose(self.F)
        return self.x[0,0]
    def measure_and_update(self,measurements,t):
        dt = t - self.prev_time
        self.F[0,1] = dt
        Z = np.matrix(measurements)
        y = np.transpose(Z) - np.dot(self.H ,self.x)
        S = np.dot(np.dot(self.H, self.P ),np.transpose(self.H)) + self.R
        K =  np.dot(np.dot(self.P,np.transpose(self.H)),np.linalg.inv(S))
        #Z = np.matrix(measurements)
        #y = np.transpose(Z) - (self.H * self.x)
        #S = self.H * self.P * np.transpose(self.H) + self.R
        #K =  self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (self.I - (K*self.H))*self.P 
        self.v = self.x[1,0]
        self.prev_time = t
        #add some uncertainy, so the filter is not overconfident
        self.P[0,0] += 0.1
        self.P[1,1] += 0.1
        return


sim_run(options,KalmanFilter)
