import pygame
import math
import os
import numpy as np

class Constants:
    G = 6.67 * 10**(-11)
    sigma = 5.67 * 10**(-8)
    M_earth = 6 * 10**(24)
    M_solar = 1.988 * 10**30
    D_moon = 400000 * 10**3  #[m] earth moon distance
    AU = 1.496 * 10**11 #[m]
    r_sch = 3 * 10**3  # Schwarzschild radius in solar mass
    r_earth = 6371 * 10**3 #[m]
    k_love = 0.3
