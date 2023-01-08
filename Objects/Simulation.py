import pygame
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from Objects.Stationary_Object import Stationary_Object
from Objects.Dynamic_Object import Dynamic_Object

class Simulation:
    def __init__(
        self,
        WIDTH,
        HEIGHT,
        scale,
        dt,
        steps,
        Constants,
        M_blackhole,
        r_blackhole,
        M_blanet,
        r_blanet,
        initial_spin_blanet,
        spin_vel_blanet,
        R,  # in AU
        alpha,  # absorptivity of the blanet
        M_moon,
        x_moon,  # with respect to the blanet
        y_moon,  # with respect to the blanet
        v_moon # initial velocity (vector) of the moon in terms of v_esc

    ):
        # simulation parameters
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.scale = scale
        self.dt = dt
        self.steps = steps

        # load the constants
        self.C = Constants
        self.G = Constants.G
        self.sigma = Constants.sigma
        self.M_earth = Constants.M_earth
        self.M_solar = Constants.M_solar
        self.D_moon = Constants.D_moon  # earth moon distance
        self.k_love = Constants.k_love  # (imaginary part of) the second love number

        # parameters of the massive bodies
        self.M_blackhole = M_blackhole
        self.r_blackhole = r_blackhole
        self.M_blanet = M_blanet
        self.r_blanet = r_blanet
        self.R = R * Constants.AU # distance between the massive bodies
        self.mu = self.M_blanet / (self.M_blanet + self.M_blackhole)
        self.r_1 = self.R * self.mu
        self.r_2 = self.R * (1 - self.mu)
        self.alpha = alpha
        self.omega = math.sqrt((self.G*(self.M_blanet + self.M_blackhole) / (self.R**3)))
        self.initial_spin_blanet = initial_spin_blanet
        self.spin_vel_blanet = spin_vel_blanet

        # parameters of the moon
        self.M_moon = M_moon
        self.x_moon = x_moon
        self.y_moon = y_moon
        self.D_moon = math.sqrt(self.x_moon**2 + self.y_moon**2)
        self.v_esc = math.sqrt((self.G * self.M_blanet) / self.D_moon)
        self.vx_moon = v_moon[0] * self.v_esc
        self.vy_moon = v_moon[1] * self.v_esc

        # create list to store data from simulation
        self.F_tidal = []
        self.pos_data = []  # record the position of the moon
        self.d_data = []  # distance between blanet and moon
        self.n_data = []  # for mean orbital motion
        self.water_level = []  # sum water level
        self.water_level_moon = []  # contribution from the moon
        self.water_level_BH = []  # contribution from the BH

    # coordinate transformation between real coordiantes and pixel coordinates
    def cart_to_pixel(self, x, y):
        # translate (set the blanet at the origin)
        x -= self.r_2
        # scale to pixel length
        x *= self.scale
        y *= -self.scale
        # shift (0,0) to (0,5 width, 0.5 height)
        pixel_x = x + 0.5 * self.WIDTH
        pixel_y = y + 0.5 * self.HEIGHT
        return pixel_x, pixel_y

    # calculate tidal force (blanet + blackhole)
    def tidal_force(self, blanet, moon, blackhole):
        xb = blanet.x
        yb = blanet.y
        xm = moon.x
        ym = moon.y
        Mb = blanet.M
        Mm = moon.M
        MB = blackhole.M
        D = math.sqrt((xb-xm)**2 + (yb-ym)**2)
        r = blanet.radius
        R = self.R
        # compute the tidal forces
        Ft_moon = (2 * self.G * Mb * Mm * 2 * r) / (D**3)
        Ft_blackhole = ((xm - xb) / D) * (2 * self.G * MB* Mb * 2 * r) / (R**3)
        return Ft_moon + Ft_blackhole

    # calculate tidal heating k\cdot\frac{21}{2}\cdot\frac{GM_{h}^{2}r^{5}nh^{2}}{a^{6}}
    def tidal_heating(self, Mh, R, k, n, e, a):
        P = k * (21/2) * self.G * Mh**2 * R**5 * n * e**2 / a**6
        return P

    def tidal_bulge(self, blanet, moon, blackhole, spin, points):
        xb = blanet.x
        yb = blanet.y
        xm = moon.x
        ym = moon.y
        Mb = blanet.M
        Mm = moon.M
        MB = blackhole.M
        DQ = math.sqrt((xb-xm)**2 + (yb-ym)**2)
        DP = self.R
        r = blanet.radius
        R = self.R
        # calculate epsilon angle
        dx = xm - xb
        dy = ym - yb

        # calculate epsilon
        epsilon = np.arctan(abs(ym - yb) / abs(xm - xb))
        if dx < 0 and dy > 0:
            epsilon = np.pi - epsilon
        elif dx < 0 and dy < 0:
            epsilon = np.pi + epsilon
        elif dx > 0 and dy < 0:
            epsilon = 2 * np.pi - epsilon

        H = []
        for i in range(0, points):
            theta_1 = 2 * np.pi * i / points
            theta_2 = np.pi - (theta_1 + epsilon)
            # height contribution due to the moon
            HQ = (1/2) * (Mm / Mb) * (r**4 / DQ**3) * (2 * np.cos(theta_1)**2 - np.sin(theta_1)**2)
            HP = (1/2) * (MB / Mb) * (r**4 / DP**3) * (2 * np.cos(theta_2)**2 - np.sin(theta_2)**2)
            h = 60000 * (HQ+HP) + 3.5 * r
            x = h * np.cos(theta_1 + epsilon) + xb
            y = h * np.sin(theta_1 + epsilon) + yb
            xc = 3.5 * r * np.cos(theta_1 + epsilon) + xb
            yc = 3.5 * r * np.sin(theta_1 + epsilon) + yb
            H.append((x,y,xc,yc))

        # calculate the water level at a certain point totating
        HQ_spin = (1/2) * (Mm / Mb) * (r**4 / DQ**3) * (2 * np.cos(spin - epsilon)**2 - np.sin(spin - epsilon)**2)
        HP_spin = (1/2) * (MB / Mb) * (r**4 / DP**3) * (2 * np.cos(np.pi - spin)**2 - np.sin(np.pi - spin)**2)
        h_spin = 60000 * (HQ_spin + HP_spin) + 3.5 * r
        h_level = (HQ_spin + HP_spin)
        x_spin = h_spin * np.cos(spin) + xb
        y_spin = h_spin * np.sin(spin) + yb
        level = (x_spin, y_spin, h_level, HQ_spin, HP_spin)
        return H, level

    # define the simulation
    def simulation(self):
        # reset data storage
        self.F_tidal = []
        self.pos_data = []
        self.d_data = []
        self.n_data = []  # for mean orbital motion
        self.water_level = []
        self.water_level_moon = []
        self.water_level_BH = []

        # create the objects
        blackhole = Stationary_Object(
            x=-self.r_1,
            y=0,
            M=self.M_blackhole,
            radius=self.r_blackhole,
            initial_spin=0,
            spin_vel=0
        )

        blanet = Stationary_Object(
            x=self.r_2,
            y=0,
            M=self.M_blanet,
            radius=self.r_blanet,
            initial_spin=self.initial_spin_blanet,  # starting postion of spin
            spin_vel=self.spin_vel_blanet
        )

        moon = Dynamic_Object(
            x=self.x_moon + self.r_2,
            y=self.y_moon,
            v_x=self.vx_moon,
            v_y=self.vy_moon,
            omega=self.omega,
            M=self.M_moon,
            dt=self.dt
        )

        # set up pygame display
        WINDOW = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Simulation")

        # colors
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (0, 0, 128)
        red = (255,0,0)

        # frame rate
        FPS = 60

        # start pygame
        pygame.init()

        # initialize font
        font = pygame.font.Font('freesansbold.ttf', 23)

        # initialize clock
        clock = pygame.time.Clock()

        running = True
        current_time = 0
        it = 0

        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break


            # DRAWING
            WINDOW.fill((255,255,255))


            pygame.draw.circle(
                WINDOW,
                (0,0,0),
                self.cart_to_pixel(blackhole.x, blackhole.y),
                15)

            pygame.draw.circle(
                WINDOW,
                red,
                self.cart_to_pixel(blanet.x, blanet.y),
                10)

            Tidal_bulge, level = self.tidal_bulge(blanet, moon, blackhole, blanet.spin, 100)
            for each in Tidal_bulge:
                pygame.draw.circle(
                    WINDOW,
                    (0,0,255),
                    self.cart_to_pixel(each[0], each[1]),
                    1)

                pygame.draw.circle(
                    WINDOW,
                    (0,0,128),
                    self.cart_to_pixel(each[2], each[3]),
                    1)

            pygame.draw.line(
                WINDOW,
                (255,255,0),
                self.cart_to_pixel(level[0], level[1]),
                self.cart_to_pixel(blanet.x, blanet.y),
                3
            )

            # add water level data
            self.water_level.append((current_time, level[2]))
            self.water_level_moon.append((current_time, level[3]))
            self.water_level_BH.append((current_time, level[4]))

            # UPDATING
            for i in range(self.steps):
                moon.update(
                    targets=[blackhole, blanet],
                    G=self.G
                )

                # calculate tidal force
                Tidal_force = self.tidal_force(
                    blanet,
                    moon,
                    blackhole
                )
                self.F_tidal.append(Tidal_force)

                self.pos_data.append((moon.x, moon.y))

                # calculate blanet moon distance
                dmoon = math.sqrt((moon.x-blanet.x)**2 + (moon.y-blanet.y)**2)
                self.d_data.append(dmoon)

                # calculate angular velocity of the moon
                v_moon = math.sqrt(moon.v_x**2 + moon.v_y**2)
                dtheta = v_moon / dmoon
                self.n_data.append(dtheta)

            # SHOW TRACE
            for i in range(0, len(moon.trace),100):
                x = moon.trace[i][0]
                y = moon.trace[i][1]
                pygame.draw.circle(
                    WINDOW,
                    (0,0,0),
                    self.cart_to_pixel(x, y),
                    1)

            pygame.draw.circle(
                WINDOW,
                (0,0,0),
                self.cart_to_pixel(moon.x, moon.y),
                5)

            # DELETE OLD TRACES
            if len(moon.trace) >= 50 * self.steps:
                moon.trace = moon.trace[self.steps::]


            # DISPLAY TIME
            time_text = "t[earth day]" + " = " + str(round(current_time,4))
            time_text = font.render(time_text, True, (0,0,0), (255,255,255))
            time_text_rect = time_text.get_rect()
            time_text_rect.topleft = (50,20)
            WINDOW.blit(time_text, time_text_rect)

            # DISPLAY ORBITAL PERIOD
            P_text = "P[earth day]" + "=" + str(round((2*math.pi) / (self.omega * 60 * 60 * 24),2))
            P_text = font.render(P_text, True, (0,0,0), (255,255,255))
            P_text_rect = P_text.get_rect()
            P_text_rect.topleft = (50,50)
            WINDOW.blit(P_text, P_text_rect)

            # DISPLAY MOON VELOCITY
            vmoon = math.sqrt(moon.v_x**2 + moon.v_y**2)
            vmoon_text = "v_moon=" +str(round(vmoon, 4))
            vmoon_text = font.render(vmoon_text, True, (0,0,0), (255,255,255))
            vmoon_text_rect = vmoon_text.get_rect()
            vmoon_text_rect.topleft = (50,80)
            WINDOW.blit(vmoon_text, vmoon_text_rect)

            # DISPLAY DISTANCE MOON BLANET
            dmoon_text = "D_moon=" +str(round(dmoon, 4))
            dmoon_text = font.render(dmoon_text, True, (0,0,0), (255,255,255))
            dmoon_text_rect = dmoon_text.get_rect()
            dmoon_text_rect.topleft = (50,110)
            WINDOW.blit(dmoon_text, dmoon_text_rect)

            # DISPLAY TIDAL FORCE
            tidal_text = "F[N]=" +str(round(self.F_tidal[-1], 4))
            tidal_text = font.render(tidal_text, True, (0,0,0), (255,255,255))
            tidal_text_rect = tidal_text.get_rect()
            tidal_text_rect.topleft = (50,140)
            WINDOW.blit(tidal_text, tidal_text_rect)

            # update time
            current_time += self.steps * self.dt / (60 * 60 * 24)
            # update spin
            blanet.update(current_time)

            pygame.display.update()
        pygame.quit()

    # evaluate the data (time_int is the averaging interval * dt for tidal heating)
    def evaluation(self, time_int = 60*60*24):
        if self.F_tidal == [] and self.d_data == []:
            self.simulation()

        # numerical differentiating the d_data set to find minima and maxima
        t_critical = []  # time of the critical points
        for i in range(len(self.d_data) - 1):
            delta = (self.d_data[i+1] - self.d_data[i]) / self.dt 
            epsilon = 0.05  # threshold for the slope. Slope flatter than 0.05 considered a critical point
            if abs(delta) < epsilon:
                t_critical.append(i * self.dt)
            else:
                t_critical.append(None)
        
        # print(len(t_critical))

        # determine local minima and maxima:
        d_data_max = []
        d_data_min = []

        # buffer 
        scanning = True
        av_d_data_crit = [] 
        av_d_data_noncrit = []
        i = 0
        mode = 0  # 0 when scanning critical points and 1 when scanning non-critical points
        while scanning:
            if i >= len(t_critical) - 1:
                scanning = False
                break

            # apply alternating search
            if mode == 0:
                # scanning for critical points 
                j = i

                d_crit = []
                t_crit = []

                while t_critical[j] != None:
                    d_crit.append(self.d_data[j])
                    t_crit.append(j * self.dt)
                    if j < len(t_critical) - 1:
                        j += 1
                    else:
                        break 
                if d_crit != []:
                    av_d_crit = sum(d_crit) / len(d_crit)
                    av_t_crit = sum(t_crit) / len(t_crit)
                    av_d_data_crit.append((av_t_crit, av_d_crit))
                print(f"Critical point found between t={self.dt * i} and ={self.dt * (j - 1)}")
                i = j
                mode = 1
            else:
                # scanning for non critical points 
                k = i

                d_noncrit = []
                t_noncrit = []
                while t_critical[k] == None:
                    d_noncrit.append(self.d_data[k])
                    t_noncrit.append(k * self.dt)
                    if k < len(t_critical) - 1:
                        k += 1
                    else:
                        break 

                if d_noncrit != []:
                    av_d_noncrit = sum(d_noncrit) / len(d_noncrit)
                    av_t_noncrit = sum(t_noncrit) / len(t_noncrit)
                    av_d_data_noncrit.append((av_t_noncrit, av_d_noncrit))
                print(f"Non critical point found between t={self.dt * i} and ={self.dt * (k - 1)}")
                i = k
                mode = 0
            
        # determine whether we start we non critical (1) or critical points (0)
        start = 1
        if av_d_data_crit[0][0] < av_d_data_noncrit[0][0]:
            start = 0
        # first point 
        if av_d_data_crit[0][1] > av_d_data_noncrit[start + 0][1]:
            d_data_max.append(av_d_data_crit[0])
        else:
            d_data_min.append(av_d_data_crit[0])

        # inbetween
        for i in range(1, len(av_d_data_crit)-1):  # -1 for buffer
            if av_d_data_crit[i][1] > av_d_data_noncrit[start + i][1] and av_d_data_crit[i][1] > av_d_data_noncrit[start + i - 1][1]:
                d_data_max.append(av_d_data_crit[i])
            elif av_d_data_crit[i][1] < av_d_data_noncrit[start + i][1] and av_d_data_crit[i][1] < av_d_data_noncrit[start + i - 1][1]:
                d_data_min.append(av_d_data_crit[i])

        # the last critical point 
        # check if we end on the critical point 
        if av_d_data_crit[-1][0] > av_d_data_noncrit[-1][0]:
            # if end on critical point
            if av_d_data_crit[-1][1] > av_d_data_noncrit[-1][1]:
                d_data_max.append(av_d_data_crit[-1])
            else:
                d_data_min.append(av_d_data_crit[-1])
        else:
            # if not end on critical point
            if av_d_data_crit[-1][1] > av_d_data_noncrit[-2][1] and av_d_data_crit[-1][1] > av_d_data_noncrit[-1][1]:
                d_data_max.append(av_d_data_crit[-1])
            elif av_d_data_crit[-1][1] < av_d_data_noncrit[-2][1] and av_d_data_crit[-1][1] < av_d_data_noncrit[-1][1]:
                d_data_min.append(av_d_data_crit[-1])

        print(d_data_max)
        print(d_data_min)
        if d_data_max == []:
            return "No maxima found, try to run the simulation for a longer period of time"
        if d_data_min == []:
            return "No minima found, try to run the simulation for a longer period of time"

        # tidal heating using the second love number
        # mean orbital motion
        n = sum(self.n_data) / (len(self.n_data))
        # perihelion and aphelion
        rp = sum([i[1] for i in d_data_min]) / len(d_data_min)
        print(f"r_p:{rp}")
        ra = sum([i[1] for i in d_data_max]) / len(d_data_max)
        print(f"r_a:{ra}")
        # eccentricity
        e = (ra - rp) / (ra + rp)
        print(f"Eccentricity:{e}")
        # average distance
        a = sum(self.d_data) / (len(self.d_data))
        # tidal heating
        E_dot = self.tidal_heating(self.M_moon, self.r_blanet, self.k_love, n, e, a)
        #greenhouse effect
        # alpha * (in + out) = 2 * out >> (2 - alpha) * out = alpha * in
        E_dot_gr = (self.alpha / (2 - self.alpha)) * E_dot
        # temperature assuming black body
        T = (E_dot / (4 * np.pi * self.r_blanet**2 * self.sigma))**(1/4)
        # temperature with greenhouse effect
        T_gr = ((E_dot + E_dot_gr) / (4 * np.pi * self.r_blanet**2 * self.sigma))**(1/4)
        print(f"Equilibrium temperature due to tidal heating {T} K")
        print(f"Greenhouse temperature due to tidal heating {T_gr} K")

        # plot data
        # position data
        xvalues = [self.pos_data[i][0] for i in range(0, len(self.pos_data), self.steps//10)]
        yvalues = [self.pos_data[i][1] for i in range(0, len(self.pos_data), self.steps//10)]
        plt.scatter(xvalues, yvalues, s=1)
        plt.scatter(self.r_2, 0)
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend(["orbit of the moon", "position of the blanet"], loc="upper right")
        plt.show() 
        
        # distance data
        plt.plot([i * self.dt / (60 * 60 * 24) for i in range(len(self.d_data))], [each for each in self.d_data])
        plt.scatter([i[0] / (60 * 60 * 24) for i in d_data_max], [i[1] for i in d_data_max])
        plt.scatter([i[0] / (60 * 60 * 24) for i in d_data_min], [i[1] for i in d_data_min])
        plt.xlabel("t[days]")
        plt.ylabel("Blanet moon distance [m]")
        plt.legend(["distance blanet-moon", r"r_a", r"r_p"], loc="upper right")
        plt.show()

        # tidal data
        plt.plot([i[0] for i in self.water_level], [i[1] for i in self.water_level])
        plt.plot([i[0] for i in self.water_level_moon], [i[1] for i in self.water_level_moon], linestyle="dashed")
        plt.plot([i[0] for i in self.water_level_BH], [i[1] for i in self.water_level_BH], linestyle="dashed")
        plt.legend(['sum', 'contribution from the moon', 'contribution from the black hole'], loc="upper right")
        plt.xlabel("t[days]")
        plt.ylabel("Tidal bulge level [m]")
        plt.show()

        plt.plot([i[0] for i in self.water_level], [i[1] for i in self.water_level])
        plt.legend(['sum'], loc="upper right")
        plt.xlabel("t[days]")
        plt.ylabel("Tidal bulge level [m]")
        plt.show()

        plt.plot([i[0] for i in self.water_level_moon], [i[1] for i in self.water_level_moon])
        plt.legend(['contribution from the moon'], loc="upper right")
        plt.xlabel("t[days]")
        plt.ylabel("Tidal bulge level [m]")
        plt.show()

        plt.plot([i[0] for i in self.water_level_BH], [i[1] for i in self.water_level_BH])
        plt.legend(['contribution from the black hole'], loc="upper right")
        plt.xlabel("t[days]")
        plt.ylabel("Tidal bulge level [m]")
        plt.show()

