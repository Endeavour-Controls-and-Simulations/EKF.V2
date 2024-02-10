import numpy as np
from N1800 import N1800

class Rocket_Data():

    internalClock   = 0.
    motor           = N1800()
    x               = np.array([[0., 0., -9.8]]).T #Pos, Vel, Acc

    a, b            = np.array([[0., 300, 0.]]).T, np.array([[0., 310, 0.]]).T
    drag_SA     = 0.02
    drag_Array = np.array([0])
    
    rocketmass   = 26.67123 # Kg

    altitude_Air_Density = np.array([(0,1.225),(1000,1.112),(2000,1.007),(3000,0.9093),(4000,0.8194),(5000,0.7364),(6000,0.6601),(7000,0.5900),(8000,0.5258)])

    ProcessVar, SensorVar = 0, 0


    def __init__(self, PV = np.array([[.01, 0.004, 0.0002]]).T, SV = np.array([[20., 0.3, 0.1]]).T):
        print("Init...")
        self.ProcessVar    = PV
        self.SensorVar     = SV

    def update_Clock(self, epoch):
        self.internalClock += epoch

    def update(self, epoch):
        self.update_Clock(epoch)
        self.update_Acceleration(epoch) 
        self.update_Attributes(epoch)     


    def openAirBreaks(self, per): # Pass percentage ie 0.0-1.0
        self.drag_SA = self.drag_SA + (per * 0.008)
        
    def noise(self, avg, std):
        return np.random.normal(avg, std, size=(1))[0]
    

    def update_Attributes(self, epoch): # TimeStampk: k
        updateMatrix        = np.array([[1, epoch, epoch**2 * .5],
                                        [0,     1,          epoch],
                                        [0,     0,             1.]])
        self.x              = np.dot(updateMatrix, self.x) + self.ProcessVar
        self.a              = np.dot(updateMatrix, self.a) + self.ProcessVar
        self.b              = np.dot(updateMatrix, self.b) + self.ProcessVar

        if self.x[0] < 0.:
            self.x          = np.array([[0., 0., 0.]]).T


    def update_Acceleration(self, timeStampk):
        current_drag        = self.drag(self.x[0], self.x[1])
        self.drag_Array     = np.append(self.drag_Array,[current_drag/self.rocketmass])
        
        self.x[2]           = ((self.motor.thrustProfile(self.internalClock) - current_drag) / self.rocketmass) - 9.81

        self.a[2]           = (-self.drag(self.a[0], self.a[1]) / self.rocketmass)
        self.b[2]           = (-self.drag(self.b[0], self.b[1]) / self.rocketmass)

    def drag(self, altk_1, veck_1):
        air_density = 0
        for i in range(len(self.altitude_Air_Density)):
            if (altk_1 < self.altitude_Air_Density[i][0]):
                per = (altk_1 - self.altitude_Air_Density[i-1][0]) / (self.altitude_Air_Density[i][0] - self.altitude_Air_Density[i-1][0])
                air_density = self.altitude_Air_Density[i-1][1] +  (per * (self.altitude_Air_Density[i][1] - self.altitude_Air_Density[i-1][1]))
                break
        drag_coeff = 0.6#0.18
        return ((veck_1*veck_1) * drag_coeff * air_density * self.drag_SA) / 2

    def returnXY(self, Noise=True):
        Q   = np.array([[self.noise(0, self.SensorVar[0]), self.noise(0, self.SensorVar[1]), self.noise(0, self.SensorVar[2])]]).T
        return ((self.a[0][0] + Q[0][0], self.b[0][0] + Q[0][0]))

    def SensorReading(self, Noise=True):
        Q   = np.array([[self.noise(0, self.SensorVar[0]), self.noise(0, self.SensorVar[1]), self.noise(0, self.SensorVar[2])]]).T
        if Noise:
            return Q + self.x
        return self.x
