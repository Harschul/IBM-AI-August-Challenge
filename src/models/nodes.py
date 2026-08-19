import numpy as np

class Satellite:
    def __init__(
        self,
        radius,
        longitude=0,
        inclination=0,
        angular_velocity=0,
        connection_range=0
    ):
        self._radius = radius
        self._longitude = longitude
        self._inclination = inclination
        self._angular_velocity = angular_velocity
        self._connection_range = connection_range
        self._vertices = []

    def radius(self):
        return self._radius

    def longitude(self):
        return self._longitude

    def inclination(self):
        return self._inclination

    def angular_velocity(self):
        return self._angular_velocity

    def connection_range(self):
        return self._connection_range

    def vertices(self):
        return self._vertices

    def pos(self):
        r = self.radius()
        longitude = self.longitude()
        inclination = self.inclination()
        x = r * np.cos(longitude)
        y = r * np.sin(longitude) * np.cos(inclination)
        z = r * np.sin(longitude) * np.sin(inclination)
        return np.array([x, y, z])

    def propagate(self, time):
        self._longitude += self.angular_velocity() * time
        self._longitude %= 2 * np.pi
        self._vertices.append(self.pos())

    def distance_to(self, other):
        return np.linalg.norm(
            self.pos() - other.pos())

    def can_connect_to(self, other):
        return self.distance_to(other) <= self.connection_range()
