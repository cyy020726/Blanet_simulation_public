import pygame
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time 
from Population_simulation.Simulation import Simulation

sim = Simulation(
        WIDTH=600,
        HEIGHT=600,
        BOUNDARY=(100,100,500,500),
        dt=0.5,
        alien_population=30,
        prey_population=100,
        alien_hp=100,
        alien_hunger=0.1,
        alien_birthrate=0.003,
        prey_birthrate=0.8,
        max_prey = 500
    )
sim.simulation()
print(f"Number of iteration simulated: {len(sim.alien_population_data)}")

for i in sim.tide_moments:
    plt.axvspan(i[0], i[1], alpha=0.3, color='blue', label="_nolegend_")
plt.plot([i for i in range(0,len(sim.alien_population_data))], sim.alien_population_data, color=(1,0,0))
plt.plot([i for i in range(0,len(sim.prey_population_data))], sim.prey_population_data, color=(0,1,0))

plt.legend(["Alien", "Prey"])
plt.show()