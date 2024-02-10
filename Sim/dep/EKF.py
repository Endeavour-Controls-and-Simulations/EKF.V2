import numpy as np
from numpy.linalg import inv
import math
from numpy.random import randn

g = 9.81
K = 0.000375

class EKF:
    x = np.array([[0.1,      # X Position
                   0.1,      # Y Position
                   0.1,      # Z Position
                   0.1,      # X Velocity
                   0.1,      # Y Velocity
                   0.1,      # Z Velocity
                   0.0000375]]).T  # p/B   AtmosDensity / Ballistic Coef
    P = np.eye(7) * 10**6
    P[6][6] = 12.86 * (10**-14) * math.exp(-7.38 * (10**-5) * 10) 

    R  = np.eye(3) * 20.

    H  = np.array([[1., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0.]])
    
    f = np.array([[]])
    Phi = np.array([[]])
    
    Q = np.eye(7) * 1
    Q[0][0] = 0
    Q[1][1] = 0
    Q[2][2] = 0
    Q[6][6] = 2.0 * (10**-18)

    def __init__(self):
        self.updatePhi(0.1)
        self.updateF()


    def get(self):
        #return (self.x[0][0], self.x[2][0])
        return (self.x[0][0], self.x[1][0], self.x[2][0])

    def update(self, (zx, zy, zz)):
        z   = np.array([[zx, zy, zz]]).T
        y   = z - np.dot(self.H, self.x)     
        
        A   = inv(self.R + np.dot(self.H, np.dot(self.P, self.H.T)))
        Gain   = np.dot(self.P, np.dot(self.H.T, A))

        self.x  = self.x + np.dot(Gain, y)
        self.P  = self.P - np.dot(Gain, np.dot(self.H, self.P))
        

    def predict(self, epoch):
        #K = self.x[2] /100
        self.updateF()
        self.updatePhi(epoch)

        self.x  = self.x + (self.f * epoch)                 # Add Noise
        self.P  = np.dot(self.Phi, np.dot(self.P, self.Phi.T)) + self.Q# Add Q?   

    def updateF(self):  
        V = math.sqrt((self.x[3]**2) + (self.x[4]**2) + (self.x[5]**2))
        self.f  = np.array([[self.x[3][0],
                             self.x[4][0],
                             self.x[5][0],
                             -.5 * g * self.x[3][0] * self.x[6][0] * V,
                             -.5 * g * self.x[4][0] * self.x[6][0] * V,
                             (-.5 *g * self.x[5][0] * self.x[6][0] * V) - g,
                            -K * self.x[6][0] * self.x[5][0]]]).T


    def updatePhi(self, epoch):
        V   = math.sqrt((self.x[3] ** 2) + (self.x[4] ** 2) + (self.x[5] ** 2))
        F4  = np.array([(
                        -0.5 * g * self.x[6] * (((V ** 2) + (self.x[3] ** 2)) / V)),
                        (-0.5 * g * self.x[6] * self.x[3] * self.x[4]) / V,
                        (-0.5 * g * self.x[6] * self.x[3] * self.x[5]) / V,
                        -0.5 * g * self.x[3] * V])
        F5  = np.array([
                        F4[1],
                        -0.5 * g * self.x[6] * (((V ** 2) + (self.x[4] ** 2)) / V),
                        (-0.5 * g * self.x[6] * self.x[4] * self.x[5]) / V,
                        -0.5 * g * self.x[4] * V])
        F6  = np.array([
                        F4[2],
                        F5[2],
                        -0.5 * g * self.x[6] * (((V ** 2) + (self.x[5] ** 2)) / V),
                        -0.5 * g * self.x[5] * V])
        F7  = np.array([-K * self.x[6], -K * self.x[5]])

        self.Phi =np.array([[0., 0., 0.,    1.,    0.,    0.,    0.],
                            [0., 0., 0.,    0.,    1.,    0.,    0.],
                            [0., 0., 0.,    0.,    0.,    1.,    0.],
                            [0., 0., 0., F4[0], F4[1], F4[2], F4[3]],
                            [0., 0., 0., F5[0], F5[1], F5[2], F5[3]],
                            [0., 0., 0., F6[0], F6[1], F6[2], F6[3]],
                            [0., 0., 0., 0.   ,    0., F7[0], F7[1]]
                            ])
        self.Phi  =  np.eye(7) + (epoch * self.Phi)
    
   
