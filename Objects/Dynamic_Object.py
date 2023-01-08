import pygame
import math
import os
import numpy as np

class Dynamic_Object:
    def __init__(
        self,
        x,
        y,
        v_x,
        v_y,
        omega,
        M,
        dt
    ):
        self.x = x
        self.y = y
        self.omega = omega
        self.M = M

        # velocity and acceleration
        self.v_x = v_x
        self.v_y = v_y
        self.a_x = 0
        self.a_y = 0

        # time iteration
        self.dt = dt
        # trace
        self.trace = []

    def update(self, targets, G):
        # mass and position from the massive objects
        M_1 = targets[0].M
        M_2 = targets[1].M
        x_1 = targets[0].x
        y_1 = targets[0].y
        x_2 = targets[1].x
        y_2 = targets[1].y

        # reduced mass
        # mu = M_2 / (M_1 + M_2)

        # distance to the objects
        p1 = math.sqrt((self.x - x_1)**2 + (self.y-y_1)**2)
        p2 = math.sqrt((self.x - x_2)**2 + (self.y-y_2)**2)

        # effective potential
        U_1x = G * M_1 * (self.x - x_1) / (p1**3)
        U_2x = G * M_2 * (self.x - x_2) / (p2**3)
        U_1y = G * M_1 * (self.y - y_1) / (p1**3)
        U_2y = G * M_2 * (self.y - y_2) / (p2**3)

        # equations of motion
        self.a_x = 2 * self.omega * self.v_y + self.omega**2 * self.x - U_1x - U_2x
        self.a_y = -2 * self.omega * self.v_x + self.omega**2 * self.y - U_1y - U_2y

        self.v_x += self.a_x * self.dt
        self.v_y += self.a_y * self.dt

        self.x += self.v_x * self.dt
        self.y += self.v_y * self.dt

        self.trace.append((self.x, self.y))
