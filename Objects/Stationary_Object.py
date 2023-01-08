import pygame
import math
import os
import numpy as np

class Stationary_Object:
    def __init__(
        self,
        x,
        y,
        M,
        radius,
        initial_spin,  # starting postion of spin
        spin_vel
    ):
        self.x = x
        self.y = y
        self.M = M
        self.radius = radius
        self.initial_spin = initial_spin
        self.spin = initial_spin
        self.spin_vel = spin_vel

    def update(self, current_time=0):
        self.M = self.M
        self.x = self.x
        self.y = self.y
        self.spin = self.initial_spin + self.spin_vel * current_time
