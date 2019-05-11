import unittest
from numpy import random
from api import cart2pol, pol2cart, cart2sphere, sphere2cart


class TestCoords(unittest.TestCase):
    def test_pol(self):
        for _ in range(10):
            x, y = random.random(), random.random()
            r, ang = cart2pol(x, y)
            x1, y1 = pol2cart(r, ang)
            self.assertAlmostEqual(x, x1)
            self.assertAlmostEqual(y, y1)

    def test_sphere(self):
        for _ in range(10):
            x, y, z = random.random(), random.random(), random.random()
            r, t, p = cart2sphere(x, y, z)
            x1, y1, z1 = sphere2cart(r, t, p)
            self.assertAlmostEqual(x, x1)
            self.assertAlmostEqual(y, y1)
            self.assertAlmostEqual(z, z1)
