import matplotlib.pyplot as plt
from dep.ADA import Rocket_Data
from dep.EKF import EKF

Rocket = Rocket_Data()
'''
    For initializing Rocket_Data, you can pass the **Process Noise** Matrix and
    **Sensor Noise** Matrix as: np.array([[alt Var, vel Var, Acc Var]]).T.

    When passing in Sensor Noise Matrix, go crazy it just affects your readings but...
    don't go crazy with the process noise as it makes the system act unrealistically (you can try)
'''

Fil = EKF()

altx, alty, altz = [], [], []
estx, esty, estz = [], [], []
flight = []
error =[]


px, py, pz = 0, 0, 0
for i in range(700): #5000
    Rocket.update(.1)
    z = Rocket.SensorReading()[0][0]
    x, y = Rocket.returnXY()

    altz.append(z)
    altx.append(x)
    alty.append(y)

    flight.append(Rocket.x[0])
    if i == 50:
        Fil.x[0][0] = x
        Fil.x[3][0] = (x - px) / 0.1
        Fil.x[1][0] = y
        Fil.x[4][0] = (y - py) / 0.1
        Fil.x[2][0] = z
        Fil.x[5][0] = (z - pz) / 0.1
    px, py, pz = x, y, z    
    if i > 50:
        Fil.predict(.1)
        Fil.update((x,y,z))
        #if i > 60:
        #    Fil.update((x, y, z))

    estx.append(Fil.x[0][0])
    esty.append(Fil.x[1][0]) 
    estz.append(Fil.x[2][0])
    error.append(abs(Rocket.x[0] - Fil.x[2][0]))


plt.plot(error[50:300])
plt.show()
print(sum(error[50:300])/len(error[50:300]))

#plt.plot(flight)
plt.plot(estx)
plt.plot(altx)
plt.show()

plt.plot(esty)
plt.plot(alty)
plt.show()

plt.plot(estz)
plt.plot(altz)
plt.show()

print(max(estz))
print(max(flight))
#print(repr(Fil.P))
'''
plt.plot([(flight[i] - est[i])**2 for i in range(65, 300)])
plt.show()  

plt.plot(flight[65:300])
plt.plot(est[65:300])
plt.plot(alt[65:300])
plt.show()
'''
