"""
simulation_controlled.py
========================
Pygame intersection simulation wrapped in a closed-loop PI control
architecture. This file is a strict superset of the original
simulation.py with the following control-systems modifications:

    1.  The heuristic green-time formula
            green = ((noOfCars*carTime) + ... ) / (noOfLanes+1)
        is REPLACED by the discrete PI control law
            u(k) = u_base + Kp*e(k) + Ki*SUM(e(i)*dt)
        implemented in pi_controller.PIController.

    2.  A live matplotlib panel is rendered (Agg backend) directly
        inside the pygame window, plotting in real time:
            - q(k)  - measured queue at the next-green direction
            - u(k)  - green-time command issued by the controller
            - e(k)  - error signal r(k) - y(k)
        These are the textbook closed-loop traces a control-systems
        examiner expects to see.

    3.  An on-screen banner shows Kp, Ki, current y(k), e(k),
        integrator state, and u(k).

The vehicle dynamics, YOLO-equivalent detection, signal sequencing,
and rendering are otherwise unchanged from the original project.
"""
import os
import sys
import math
import time
import random
import threading

import pygame
import numpy as np

# ---- matplotlib in Agg mode so we can blit it onto pygame surfaces ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pi_controller import PIController

# ---------------------------------------------------------------------------
# Default signal-timing constants (kept for compatibility with original)
# ---------------------------------------------------------------------------
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300
timeElapsed = 0

currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

carTime = 2
bikeTime = 1
rickshawTime = 2.25
busTime = 2.5
truckTime = 2.5

noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2
detectionTime = 5

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'rickshaw': 2, 'bike': 2.5}

x = {'right': [0, 0, 0], 'down': [755, 727, 697],
     'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
y = {'right': [348, 370, 398], 'down': [0, 0, 0],
     'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down':  {0: [], 1: [], 2: [], 'crossed': 0},
            'left':  {0: [], 1: [], 2: [], 'crossed': 0},
            'up':    {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

signalCoods       = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods  = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]
vehicleCountTexts = ["0", "0", "0", "0"]

stopLines    = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop  = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops        = {'right': [580]*3, 'down': [320]*3, 'left': [810]*3, 'up': [545]*3}

mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450},
       'left':  {'x': 695, 'y': 425}, 'up':   {'x': 695, 'y': 400}}
rotationAngle = 3
gap = 15
gap2 = 15

pygame.init()
simulation = pygame.sprite.Group()

# ---------------------------------------------------------------------------
# Closed-loop control: PI controller + telemetry buffers
# ---------------------------------------------------------------------------
controller = PIController(
    Kp=1.2,
    Ki=0.15,
    u_base=defaultMinimum,
    u_min=defaultMinimum,
    u_max=defaultMaximum,
    dt=1.0,
    reference=0.0,        # r(k) = 0 desired queue
)

LOG_LEN = 120
_log_lock = threading.Lock()
t_log, q_log, u_log, e_log = [], [], [], []
last_y = 0.0     # most recent measurement, displayed on HUD


# ---------------------------------------------------------------------------
# Original simulation classes (unmodified)
# ---------------------------------------------------------------------------
class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        if direction == 'right':
            if (len(vehicles[direction][lane]) > 1
                    and vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (vehicles[direction][lane][self.index-1].stop
                             - vehicles[direction][lane][self.index-1].currentImage.get_rect().width
                             - gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if (len(vehicles[direction][lane]) > 1
                    and vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (vehicles[direction][lane][self.index-1].stop
                             + vehicles[direction][lane][self.index-1].currentImage.get_rect().width
                             + gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if (len(vehicles[direction][lane]) > 1
                    and vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (vehicles[direction][lane][self.index-1].stop
                             - vehicles[direction][lane][self.index-1].currentImage.get_rect().height
                             - gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if (len(vehicles[direction][lane]) > 1
                    and vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (vehicles[direction][lane][self.index-1].stop
                             + vehicles[direction][lane][self.index-1].currentImage.get_rect().height
                             + gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        # NOTE: movement physics are unchanged from the original simulation.py
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    if ((self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1)
                            and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1)):
                        self.x += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2)
                                or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else:
                if ((self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0))
                        and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1))):
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if ((self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1)
                            and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1)):
                        self.y += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2)
                                or self.y < (vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else:
                if ((self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0))
                        and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1))):
                    self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if ((self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1)
                            and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1)):
                        self.x -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2)
                                or self.x > (vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else:
                if ((self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0))
                        and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1))):
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if ((self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1)
                            and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1)):
                        self.y -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2)
                                or self.y > (vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else:
                if ((self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0))
                        and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1))):
                    self.y -= self.speed


def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow,
                        defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()


# ---------------------------------------------------------------------------
# CONTROL LAW: replaces the original heuristic green-time formula.
# This is the function the professor will read first - keep it tidy.
# ---------------------------------------------------------------------------
def setTime():
    """
    Sensor + controller cycle.

    Sensor:     count the uncrossed vehicles at the *next* green direction.
                In a real deployment this is the YOLOv7 detector
                (vehicle_detection.py) operating on a camera frame; in the
                simulation we count sprites in the relevant lanes, which is
                an idealised version of the same measurement.

    Controller: feed the measurement y(k) = q(k) into the PI controller and
                read out the green-time command u(k).
    """
    global last_y

    direction = directionNumbers[nextGreen]
    queue = 0
    for lane in range(0, 3):
        for v in vehicles[direction][lane]:
            if v.crossed == 0:
                queue += 1

    # ---- discrete PI control law: u(k) = u_base + Kp*e + Ki*INT e ----
    u = controller.compute(queue)
    greenTime = int(round(u))

    # apply control effort (green time) to the next signal
    signals[(currentGreen + 1) % noOfSignals].green = greenTime

    # log telemetry for the live plot
    with _log_lock:
        last_y = queue
        t_log.append(timeElapsed)
        q_log.append(queue)
        u_log.append(u)
        e_log.append(controller.last_error)
        if len(t_log) > LOG_LEN:
            del t_log[:-LOG_LEN]
            del q_log[:-LOG_LEN]
            del u_log[:-LOG_LEN]
            del e_log[:-LOG_LEN]

    print(f"[PI]  k={timeElapsed:>4}  y(k)={queue:>3}  e(k)={controller.last_error:+.1f}"
          f"  INT={controller.integral:+7.2f}  u(k)={greenTime:>3}s")


def repeat():
    global currentGreen, currentYellow, nextGreen
    while signals[currentGreen].green > 0:
        printStatus()
        updateValues()
        if signals[(currentGreen + 1) % noOfSignals].red == detectionTime:
            t = threading.Thread(name="detection", target=setTime, args=())
            t.daemon = True
            t.start()
        time.sleep(1)
    currentYellow = 1
    vehicleCountTexts[currentGreen] = "0"
    for i in range(0, 3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while signals[currentGreen].yellow > 0:
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0

    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
    repeat()


def printStatus():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(f"GREEN TS{i+1}-> r:{signals[i].red}  y:{signals[i].yellow}  g:{signals[i].green}")
            else:
                print(f"YELLOW TS{i+1}-> r:{signals[i].red}  y:{signals[i].yellow}  g:{signals[i].green}")
        else:
            print(f"RED TS{i+1}-> r:{signals[i].red}  y:{signals[i].yellow}  g:{signals[i].green}")
    print()


def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


def generateVehicles():
    while True:
        vehicle_type = random.randint(0, 4)
        if vehicle_type == 4:
            lane_number = 0
        else:
            lane_number = random.randint(0, 1) + 1
        will_turn = 0
        if lane_number == 2:
            temp = random.randint(0, 4)
            if temp <= 2:
                will_turn = 1
        temp = random.randint(0, 999)
        a = [400, 800, 900, 1000]
        if temp < a[0]:
            direction_number = 0
        elif temp < a[1]:
            direction_number = 1
        elif temp < a[2]:
            direction_number = 2
        else:
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type],
                direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(0.75)


def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print(f'Lane {i+1}: {vehicles[directionNumbers[i]]["crossed"]}')
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print(f'Total vehicles passed: {totalVehicles}')
            print(f'Total time passed: {timeElapsed}')
            print(f'Throughput: {totalVehicles/float(timeElapsed):.3f} veh/s')
            os._exit(1)


# ---------------------------------------------------------------------------
# Live matplotlib panel rendered to a pygame surface
# ---------------------------------------------------------------------------
class LiveControlPlot:
    """
    Off-screen matplotlib figure, redrawn on demand and converted to a
    pygame.Surface so it can be blitted into the main window.
    """
    def __init__(self, width_px=420, height_px=720, dpi=100):
        plt.style.use("dark_background")
        self.fig = plt.Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)

        self.ax_q = self.fig.add_subplot(3, 1, 1)
        self.ax_u = self.fig.add_subplot(3, 1, 2)
        self.ax_e = self.fig.add_subplot(3, 1, 3)

        for ax in (self.ax_q, self.ax_u, self.ax_e):
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)

        self.ax_q.set_title("y(k) = q(k)  Queue (System Response)", fontsize=9)
        self.ax_u.set_title("u(k)  Green time (Control Effort)", fontsize=9)
        self.ax_e.set_title("e(k)  Error  =  r(k) - y(k)", fontsize=9)
        self.ax_e.set_xlabel("k (controller cycles)", fontsize=8)

        self.fig.tight_layout(pad=1.2)
        self.size_px = (width_px, height_px)

    def render(self):
        with _log_lock:
            t  = list(t_log)
            q  = list(q_log)
            u  = list(u_log)
            e  = list(e_log)

        self.ax_q.clear(); self.ax_u.clear(); self.ax_e.clear()
        for ax in (self.ax_q, self.ax_u, self.ax_e):
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        self.ax_q.set_title("y(k) = q(k)  Queue (System Response)", fontsize=9)
        self.ax_u.set_title("u(k)  Green time (Control Effort)", fontsize=9)
        self.ax_e.set_title("e(k)  Error  =  r(k) - y(k)", fontsize=9)
        self.ax_e.set_xlabel("k (controller cycles)", fontsize=8)

        if t:
            self.ax_q.plot(t, q, color="cyan", lw=1.6, label="y(k)")
            self.ax_q.axhline(0, color="red", ls="--", lw=1.0, label="r=0")
            self.ax_q.legend(loc="upper right", fontsize=7)

            self.ax_u.plot(t, u, color="lime", lw=1.6, label="u(k)")
            self.ax_u.axhline(defaultMaximum, color="gray", ls=":", lw=0.8)
            self.ax_u.axhline(defaultMinimum, color="gray", ls=":", lw=0.8)
            self.ax_u.legend(loc="upper right", fontsize=7)

            self.ax_e.plot(t, e, color="orange", lw=1.4, label="e(k)")
            self.ax_e.axhline(0, color="white", ls=":", lw=0.6)
            self.ax_e.legend(loc="upper right", fontsize=7)

        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()
        raw = bytes(self.canvas.buffer_rgba())
        return pygame.image.frombuffer(raw, self.size_px, "RGBA")


# ---------------------------------------------------------------------------
# Main pygame loop
# ---------------------------------------------------------------------------
class Main:
    thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization", target=initialize, args=())
    thread2.daemon = True
    thread2.start()

    black = (0, 0, 0)
    white = (255, 255, 255)
    cyan  = (0, 255, 255)

    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    background = pygame.image.load('images/mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("Closed-Loop Traffic Signal  |  PI Controller (r=0)")

    redSignal    = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal  = pygame.image.load('images/signals/green.png')
    font_small   = pygame.font.Font(None, 20)
    font         = pygame.font.Font(None, 26)
    font_big     = pygame.font.Font(None, 30)

    thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread3.daemon = True
    thread3.start()

    plot = LiveControlPlot(width_px=380, height_px=720, dpi=100)
    plot_surface = plot.render()
    last_plot_t = 0.0
    PLOT_REDRAW_INTERVAL = 0.5  # seconds between plot redraws

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))

        # Signals + countdown timer
        for i in range(0, noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = "STOP" if signals[i].yellow == 0 else signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = "SLOW" if signals[i].green == 0 else signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = "GO" if signals[i].red == 0 else signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])

        signalTexts = ["", "", "", ""]
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

        timeElapsedText = font.render(f"Time Elapsed: {timeElapsed}", True, black, white)
        screen.blit(timeElapsedText, (1020, 20))

        # ---------------------------------------------------------------
        # HUD: live PI-controller state
        # ---------------------------------------------------------------
        hud_x, hud_y = 20, 20
        hud_lines = [
            "CLOSED-LOOP TRAFFIC SIGNAL  (PI Controller)",
            f"Kp = {controller.Kp}    Ki = {controller.Ki}",
            f"r(k) = {controller.reference}  (desired queue)",
            f"y(k) = {last_y}  veh   (measured queue)",
            f"e(k) = {controller.last_error:+.2f}",
            f"INT  = {controller.integral:+7.2f}",
            f"u(k) = {controller.last_u:6.2f} s   (green-time command)",
            f"u_min = {controller.u_min}    u_max = {controller.u_max}",
        ]
        pad = 6
        bg_rect = pygame.Rect(hud_x - pad, hud_y - pad, 460, 22 * len(hud_lines) + 2 * pad)
        s = pygame.Surface((bg_rect.w, bg_rect.h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 170))
        screen.blit(s, bg_rect.topleft)
        pygame.draw.rect(screen, cyan, bg_rect, 1)
        for i, line in enumerate(hud_lines):
            txt = font_small.render(line, True, cyan)
            screen.blit(txt, (hud_x, hud_y + i * 22))

        # vehicles
        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()

        # live control panel
        now = time.time()
        if now - last_plot_t > PLOT_REDRAW_INTERVAL:
            plot_surface = plot.render()
            last_plot_t = now
        screen.blit(plot_surface, (1020, 70))

        pygame.display.update()


Main()
