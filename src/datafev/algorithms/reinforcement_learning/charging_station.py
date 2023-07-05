import numpy as np
import random

class ChargingStation:
    def __init__(self, id, coord, peak_traffic=10, max_load=66000, start_time=0):
        self.id = id # ID of station
        self.coord = coord # Long and Lat of station
        self.peak_traffic = peak_traffic # Typical peak traffic
        self.max_load = max_load # Load threshold in watts
        self.traffic = 0 # Amount of vehicles at station
        self.t = start_time # Amount of time passed in minutes
        self.charger_per_hour = 12500 # Average charge per hour in watts

    def charge(self):
        self.traffic += 1

        # Evenly distribute load if needed
        if self.charge_per_hour * self.traffic > self.max_load:
            output = self.max_load / self.traffic
        else:
            output = self.charge_per_hour

        return output

    def leave(self):
        self.traffic -= 1

    def update_traffic(self):
        base_traffic = self.peak_traffic * abs(np.sin(self.t * np.pi / 720))  # 720 minutes in 12 hours, creating a cycle.
        random_noise = random.gauss(0, self.peak_traffic * 0.1)  # Gaussian noise
        self.traffic = max(0, int(base_traffic + random_noise))  # Ensure traffic never goes negative
        self.t += 1