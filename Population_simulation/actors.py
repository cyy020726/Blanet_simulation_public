import pygame
import math
import os
import numpy as np
import random
class Alien:
    def __init__(
        self,
        x,
        y,
        radius,
        boundary,  #[x1,y1,x2,y2]
        dt,
        hp,
        hunger,
        target,
        repulsion_radius,
    ):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.radius = radius
        self.boundary = boundary
        self.dt = dt
        self.label = 1  # 0 = dead, 1 = alive
        self.hp = hp
        self.hunger = hunger
        self.capture_radius = 3 * self.radius
        self.target = target
        self.repulsion_radius = repulsion_radius
        self.D = math.sqrt((self.x - target[0])**2 + (self.y - target[1])**2)

    def update(self, displacement):
        if displacement == [0,0]:
            # attracted towards the target
            Fx = 50000 / (self.D**3) * (self.target[0] - self.x)
            Fy = 50000 / (self.D**3) * (self.target[1] - self.y)
            self.ax = Fx
            self.ay = Fy

            # give the alien a random push to simulate the chaotic movement
            theta = random.uniform(0, 2*np.pi)
            r = random.uniform(0, 1)
            if self.D <= self.repulsion_radius:
                self.vx = 0 * self.ax * self.dt + r * np.cos(theta)
                self.vy = 0 * self.ay * self.dt + r * np.sin(theta)
            else:
                self.vx = self.ax * self.dt + r * np.cos(theta)
                self.vy = self.ay * self.dt + r * np.sin(theta)
        else:
            self.vx = displacement[0]
            self.vy = displacement[1]
            
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # periodic boundary
        if self.x <= self.boundary[0]:
            self.x = self.boundary[2] - 1
        elif self.x >= self.boundary[2]:
            self.x = self.boundary[0] + 1
        if self.y <= self.boundary[1]:
            self.y = self.boundary[3] - 1
        elif self.y >= self.boundary[3]:
            self.y = self.boundary[1] + 1

        # hunger will subtract hp
        self.hp -= self.hunger * self.hp
        # low hp could kill the alien
        if self.hp < 1:
            if random.uniform(0,1) < 0.01:
                self.label = 0

        # update distance to the target
        self.D = math.sqrt((self.x - self.target[0])**2 + (self.y - self.target[1])**2)


class Prey:
    def __init__(
        self,
        radius,
        boundary,  #[pos, radius]
        dt
    ):
        self.radius = radius
        self.boundary = boundary
        R = random.uniform(0, self.boundary[1]-1)
        phi = random.uniform(0, 2*np.pi)
        self.x = self.boundary[0][0] + R * np.cos(phi)
        self.y = self.boundary[0][1] + R * np.sin(phi)
        self.vx = 0
        self.vy = 0
        self.dt = dt
        self.label = 1  # 0 = dead, 1 = alive

    def update(self, current_time=0):
        change = random.uniform(0,1)
        if change > 0.99:
            # give the alien a random push to simulate the chaotic field
            theta = random.uniform(0, 2*np.pi)
            r = random.uniform(0, 1)
            self.vx = r * np.cos(theta)
            self.vy = r * np.sin(theta)

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # circular randomize boundary
        # distance to the center of the boundary zone
        D = math.sqrt((self.x - self.boundary[0][0])**2 + (self.y - self.boundary[0][1])**2)
        if D >= self.boundary[1] - 5:
            # randomize location inside boundary
            R = random.uniform(0, self.boundary[1]-1)
            phi = random.uniform(0, 2*np.pi)
            self.x = self.boundary[0][0] + R * np.cos(phi)
            self.y = self.boundary[0][1] + R * np.sin(phi)
