import numpy as np


__all__ = ['cart2pol', 'pol2cart', 'cart2sphere', 'sphere2cart']


def _fix_angle(a):
    while a > np.pi:
        a -= np.pi
    while a < -np.pi:
        a += np.pi
    return a


def cart2pol(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    ang = np.arctan2(y, x)
    return r, ang


def pol2cart(r, ang):
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return x, y


def cart2sphere(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sphere2cart(r, theta, phi):
    theta = _fix_angle(theta)
    phi = _fix_angle(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
