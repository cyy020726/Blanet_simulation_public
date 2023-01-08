import pygame
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from Objects.Stationary_Object import Stationary_Object
from Objects.Dynamic_Object import Dynamic_Object
from Objects.Constants import Constants
from Objects.Simulation import Simulation

simulation = Simulation(
    WIDTH = 800,
    HEIGHT = 800,
    scale = 120 / (0.2 * Constants.D_moon),
    dt = 0.1,  #[s]
    steps = 10000,
    Constants = Constants,
    M_blackhole = 1200 * Constants.M_solar,
    r_blackhole = (3/2) * 1200 * Constants.r_sch,
    M_blanet = 3 * Constants.M_earth,
    r_blanet = Constants.r_earth,
    initial_spin_blanet = 1,  # starting position of spin
    spin_vel_blanet = -2 * np.pi,  # per day
    R = 1,  # in AU
    alpha = 0.70,  # absorptivity of the blanet
    M_moon = 0.005 * Constants.M_earth,
    x_moon = -0.2 * Constants.D_moon,  # with respect to the blanet
    y_moon = 0,  # with respect to the blanet
    v_moon = [0,1]# initial velocity (vector) of the moon in terms of v_esc
)

simulation.simulation()
simulation.evaluation()
