from itertools import combinations

class Network:
    def __init__(self, satellites):
        self._satellites = satellites
        self._connections = []

    def satellites(self):
        return self._satellites

    def connections(self):
        return self._connections

    def update_network(self):

        # Remove connections from previous timestep
        self._connections = []

        for sat1, sat2 in combinations(self.satellites(), 2):
            if sat1.can_connect_to(sat2):
                self._connections.append((sat1.name(), sat2.name()))
    