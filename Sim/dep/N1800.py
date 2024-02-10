class N1800():
    def __init__(self):
        self.thrust          = [1769.444, 2205.556, 2041.667, 1988.889, 1994.444, 2144.444, 2119.444, 2077.778, 1786.111, 1525.0, 1444.444, 1352.778, 1261.111, 994.444, 738.889, 622.222, 355.556, 75.0, 0.0]
        self.thrustTimeStamp = [0.077, 0.123, 0.193, 0.339, 0.744, 2.398, 2.726, 2.977, 3.933, 4.643, 4.986, 5.152, 5.225, 5.348, 5.437, 5.541, 5.73, 5.915, 5.931]
        
    def thrustProfile(self, time):
        if (time > max(self.thrustTimeStamp) or time <= 0):
            return 0
        matchedIndex = 0
        match = False
        for index, i in enumerate(self.thrustTimeStamp):
            if (time < i):
                if time == i:
                    print("A point")
                    match = True
                matchedIndex = index
                break
        if match:
            return self.thrust[matchedIndex]
        
        per = (time - self.thrustTimeStamp[matchedIndex-1]) / (self.thrustTimeStamp[matchedIndex] - self.thrustTimeStamp[matchedIndex-1])
        return self.thrust[matchedIndex-1] + (per * (self.thrust[matchedIndex]- self.thrust[matchedIndex-1]))


