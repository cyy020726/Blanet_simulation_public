import pygame
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time 
from Population_simulation.actors import Alien, Prey

class Simulation:
    def __init__(
        self,
        WIDTH,
        HEIGHT,
        BOUNDARY,
        dt,
        alien_population,
        prey_population,
        alien_hp,
        alien_hunger,
        alien_birthrate,
        prey_birthrate,
        max_prey
    ):
        # simulation parameters
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.BOUNDARY = BOUNDARY  #[x1,x2,y1,y2]
        self.dt = dt
        self.alien_population_data = [alien_population]
        self.prey_population_data = [prey_population]
        self.tide_moments = []

        # Alien parameters
        self.alien_population = alien_population
        self.prey_population = prey_population
        self.alien_hp = alien_hp
        self.alien_hunger = alien_hunger
        self.alien_birthrate = alien_birthrate
        self.prey_birthrate = prey_birthrate
        self.max_prey = max_prey
    def simulation(self):
        # set up pygame display
        WINDOW = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Simulation")

        # create actors
        Aliens = []
        for i in range(self.alien_population):
            randtheta = random.uniform(0,2*np.pi)
            alien = Alien(
                x=55 * np.cos(randtheta) + self.WIDTH//2,
                y=55 * np.sin(randtheta) + self.WIDTH//2,
                radius=5,
                boundary=self.BOUNDARY,
                dt=self.dt,
                hp=self.alien_hp,
                hunger=self.alien_hunger,
                target=(self.WIDTH//2, self.HEIGHT//2),
                repulsion_radius=50
            )
            Aliens.append(alien)

        Preys = []
        for i in range(self.prey_population):
            prey = Prey(
                radius=2,
                boundary=[(self.WIDTH//2, self.HEIGHT//2), 50],
                dt=self.dt
            )
            Preys.append(prey)

        self.tide_moments = []

        # colors
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (0, 0, 128)
        red = (255,0,0)

        # frame rate
        FPS = 64

        # start pygame
        pygame.init()

        # initialize font
        font = pygame.font.Font('freesansbold.ttf', 18)

        # initialize clock
        clock = pygame.time.Clock()

        running = True
        
        current_it = 0

        # birth countdown, when it 100 alien/prey gives birth
        prey_birth_countdown = random.randint(0, 50)
        alien_birth_countdown = random.randint(0, 50)
        allow_alien_birth = True

        is_tide = False  # check is tidal current is going on
        tide_progress = 0
        allow_capture = True  # allow alien to capture prey
        tide_moments = []
        # time.sleep(5)  # 5 sec before starting the loop
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break


            # DRAWING
            WINDOW.fill((255,255,255))

            # draw the boundary
            pygame.draw.rect(
                WINDOW,
                (0,0,0),
                pygame.Rect(
                    self.BOUNDARY[0],
                    self.BOUNDARY[1],
                    self.BOUNDARY[2] - self.BOUNDARY[0],
                    self.BOUNDARY[3] - self.BOUNDARY[1]
                ),
                2
            )

            pygame.draw.circle(
                WINDOW,
                red,
                (self.WIDTH//2, self.HEIGHT//2),
                50,
                2
            )
            for alien in Aliens:
                pygame.draw.circle(
                    WINDOW,
                    (0,0,0),
                    (alien.x, alien.y),
                    alien.radius
                )
                pygame.draw.circle(
                    WINDOW,
                    (0,0,0),
                    (alien.x, alien.y),
                    alien.capture_radius,
                    1
                )
            for prey in Preys:
                # print(prey.x, prey.y)
                pygame.draw.circle(
                    WINDOW,
                    green,
                    (prey.x, prey.y),
                    prey.radius
                )

            # DISPLAY CURRENT DATA
            # display population
            alien_population_text = f"Number of aliens: {self.alien_population}"
            alien_population_text = font.render(alien_population_text, True, (0,0,0), (255,255,255))
            alien_population_text_rect = alien_population_text.get_rect()
            alien_population_text_rect.topleft = (50,20)
            WINDOW.blit(alien_population_text, alien_population_text_rect)

            prey_population_text = f"Number of preys: {self.prey_population}"
            prey_population_text = font.render(prey_population_text, True, (0,0,0), (255,255,255))
            prey_population_text_rect = prey_population_text.get_rect()
            prey_population_text_rect.topleft = (50,60)
            WINDOW.blit(prey_population_text, prey_population_text_rect)

            # ACTION
            # note population
            self.alien_population = len(Aliens)
            self.prey_population = len(Preys)
            self.alien_population_data.append(self.alien_population)
            self.prey_population_data.append(self.prey_population)

            # define capture
            if allow_capture:
                for alien in Aliens:
                    for prey in Preys:
                        # Distance between prey and alien
                        D = math.sqrt((alien.x - prey.x)**2 + (alien.y-prey.y)**2)
                        if D < alien.capture_radius:
                            # 10% chance of killing the prey
                            if random.randint(0,10) == 0:
                                prey.label = 0
                                alien.hp += 10
                    # update the Preys list
                    Preys = [prey for prey in Preys if prey.label == 1]

            # remove dead aliens
            Aliens = [alien for alien in Aliens if alien.label == 1]

            # define chaotic tidal current
            if is_tide == False:
                # 0.09% chance generate current
                if random.uniform(0,1) < 0.0009:
                    # define a duration
                    duration = random.randrange(100,200)

                    # define the current
                    r = random.uniform(20,50)
                    theta = random.uniform(0, 2*np.pi)
                    omega = random.uniform(-0.5,0.5) * (2 * np.pi / duration)
                    # start tidal current
                    is_tide = True
                    # disable capture
                    allow_capture = False
                    # disable alien birth 
                    allow_alien_birth = False 

            # birth after certain iterations
            if prey_birth_countdown > 100:
                # birth of prey
                N_prey = self.prey_birthrate * len(Preys) * (1 - (len(Preys) / self.max_prey))
                for i in range(0, int(N_prey)):
                    prey = Prey(
                        radius=2,
                        boundary=[(self.WIDTH//2, self.HEIGHT//2), 50],
                        dt=self.dt
                    )
                    Preys.append(prey)
                # reset countdown 
                prey_birth_countdown = random.randint(0, 50)

            if alien_birth_countdown > 100:
                # birth of alien
                if allow_alien_birth == True:
                    for i in range(0, int(self.alien_birthrate * len(Aliens) * len(Preys))):
                        # birth around the preys
                        randtheta = random.uniform(0,2*np.pi)
                        alien = Alien(
                            x=55 * np.cos(randtheta) + self.WIDTH//2,
                            y=55 * np.sin(randtheta) + self.WIDTH//2,
                            radius=5,
                            boundary=self.BOUNDARY,
                            dt=self.dt,
                            hp=self.alien_hp,
                            hunger=self.alien_hunger,
                            target=(self.WIDTH//2, self.HEIGHT//2),
                            repulsion_radius=50
                        )
                        Aliens.append(alien)
                # reset countdown 
                alien_birth_countdown = random.randint(0, 50)

            # UPDATE
            for alien in Aliens:
                if is_tide:
                    alien.update(displacement = [
                        r * np.cos(theta + omega * tide_progress),
                        r * np.sin(theta + omega * tide_progress)]
                    )

                    # display tide
                    tidal_vector_x = round(r * np.cos(theta + omega * tide_progress),1)
                    tidal_vector_y = round(r * np.sin(theta + omega * tide_progress),1)
                    # draw tidal vector on aliens
                    pygame.draw.line(
                        WINDOW,
                        blue,
                        (alien.x, alien.y),
                        (alien.x + tidal_vector_x, alien.y + tidal_vector_y),
                        3
                    )
                    # draw the triangular tip
                    top_point = (alien.x + tidal_vector_x, alien.y + tidal_vector_y)
                    # apply rotation matrix to obtain the left and right point
                    left_point_x = (np.cos(0.3) * (0.5 * tidal_vector_x) - np.sin(0.3) * (0.5 * tidal_vector_y)) + alien.x
                    left_point_y = (np.sin(0.3) * (0.5 * tidal_vector_x) + np.cos(0.3) * (0.5 * tidal_vector_y)) + alien.y
                    left_point = (left_point_x, left_point_y)

                    right_point_x = (np.cos(0.3) * (0.5 * tidal_vector_x) + np.sin(0.3) * (0.5 * tidal_vector_y)) + alien.x
                    right_point_y = (-np.sin(0.3) * (0.5 * tidal_vector_x) + np.cos(0.3) * (0.5 * tidal_vector_y)) + alien.y
                    right_point = (right_point_x, right_point_y)

                    pygame.draw.polygon(WINDOW, blue, (top_point, left_point , right_point))
                    tidal_vector_text = f"Tidal current vector: ({tidal_vector_x}, {tidal_vector_y})"
                    tidal_vector_text = font.render(tidal_vector_text, True, (0,0,0), (255,255,255))
                    tidal_vector_text_rect = tidal_vector_text.get_rect()
                    tidal_vector_text_rect.topleft = (50,530)
                    WINDOW.blit(tidal_vector_text, tidal_vector_text_rect)
                else:
                    tidal_vector_text = "Tidal current vector:"
                    tidal_vector_text = font.render(tidal_vector_text, True, (0,0,0), (255,255,255))
                    tidal_vector_text_rect = tidal_vector_text.get_rect()
                    tidal_vector_text_rect.topleft = (50,530)
                    WINDOW.blit(tidal_vector_text, tidal_vector_text_rect)
                    alien.update(displacement = [0,0])

            for prey in Preys:
                prey.update()

            # update tidal progress
            if is_tide:
                # when tidal current is gone
                if tide_progress > duration:
                    tide_progress = 0
                    is_tide = False
                    allow_capture = True
                    allow_alien_birth = True

                    self.tide_moments.append((tide_moments[0], tide_moments[-1]))
                    tide_moments = []

                else:
                    tide_progress += 1
                    tide_moments.append(current_it)
            # update the birth countdown
            prey_birth_countdown += 1
            alien_birth_countdown += 1

            # update current iteration 
            current_it += 1

            pygame.display.update()
        pygame.quit()


if __name__ == '__main__':
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

