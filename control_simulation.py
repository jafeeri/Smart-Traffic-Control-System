"""
control_simulation.py
=====================
Standalone closed-loop simulation of the traffic-signal PI controller.

This script is the "control-systems showpiece" of the project. It runs the
PI controller against the discrete plant and produces a LIVE animated plot
showing:

    * y(k) = q(k)   - queue length (system response) vs reference r=0
    * u(k)          - control effort (green-time command)
    * A(k)          - exogenous arrivals (disturbance / step input)
    * e(k)          - error signal

The arrival profile is a step disturbance: a quiet baseline followed by a
sudden surge of vehicles at k = STEP_TIME. This is the canonical test
input used to read off classical step-response metrics:

    * rise time
    * peak overshoot  Mp
    * settling time   Ts
    * steady-state error  ess

Run:
    python control_simulation.py

Tune Kp and Ki below to demonstrate underdamped, critically damped, and
overdamped responses for the viva.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pi_controller import PIController
from plant_model import IntersectionPlant

# ---------------------------------------------------------------------------
# Tuning parameters (vary these for the demo / viva)
# ---------------------------------------------------------------------------
Kp = 1.2
Ki = 0.15
u_base = 10.0
u_min = 10.0
u_max = 60.0
dt = 1.0                  # one cycle per discrete step

SATURATION_FLOW = 0.5     # veh / s of green
SIM_STEPS = 200
STEP_TIME = 30            # cycle index at which arrivals surge
NOMINAL_ARRIVAL = 5.0     # veh / cycle baseline
SURGE_ARRIVAL  = 22.0     # veh / cycle after step disturbance
NOISE_STD = 1.0           # gaussian arrival noise (models YOLO / driver variation)

# ---------------------------------------------------------------------------
# Plant + controller instances
# ---------------------------------------------------------------------------
plant = IntersectionPlant(saturation_flow=SATURATION_FLOW, q0=0.0)
controller = PIController(Kp=Kp, Ki=Ki, u_base=u_base,
                          u_min=u_min, u_max=u_max, dt=dt, reference=0.0)

t_log, q_log, u_log, A_log, e_log = [], [], [], [], []


def arrival_signal(k):
    base = SURGE_ARRIVAL if k >= STEP_TIME else NOMINAL_ARRIVAL
    return max(0.0, base + np.random.normal(0.0, NOISE_STD))


# ---------------------------------------------------------------------------
# Live figure
# ---------------------------------------------------------------------------
plt.style.use('dark_background')
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle('Closed-Loop Traffic Signal: PI Control of Queue Length q(k)',
             fontsize=13, fontweight='bold')

ax_q, ax_u, ax_a = axes

# --- output / system response ---
line_q, = ax_q.plot([], [], color='cyan', lw=2.0, label='y(k) = q(k)  measured queue')
line_r, = ax_q.plot([], [], color='red', ls='--', lw=1.2, label='r(k) = 0  reference')
ax_q.set_ylabel('Queue (veh)')
ax_q.set_xlim(0, SIM_STEPS)
ax_q.set_ylim(-2, max(60, SURGE_ARRIVAL * 2))
ax_q.legend(loc='upper right', fontsize=9)
ax_q.grid(alpha=0.3)
ax_q.set_title('System response  y(t)')

# --- control effort ---
line_u, = ax_u.plot([], [], color='lime', lw=2.0, label='u(k)  green time (s)')
ax_u.axhline(u_max, color='gray', ls=':', lw=0.8, label='u_max')
ax_u.axhline(u_min, color='gray', ls=':', lw=0.8, label='u_min')
ax_u.set_ylabel('u(k) [s]')
ax_u.set_xlim(0, SIM_STEPS)
ax_u.set_ylim(0, u_max + 10)
ax_u.legend(loc='upper right', fontsize=9)
ax_u.grid(alpha=0.3)
ax_u.set_title('Control effort  u(t)')

# --- disturbance ---
line_a, = ax_a.plot([], [], color='magenta', lw=1.5, label='A(k)  arrivals (disturbance)')
ax_a.set_ylabel('A(k) [veh]')
ax_a.set_xlabel('Discrete time index  k  (cycles)')
ax_a.set_xlim(0, SIM_STEPS)
ax_a.set_ylim(0, SURGE_ARRIVAL + 10)
ax_a.legend(loc='upper right', fontsize=9)
ax_a.grid(alpha=0.3)
ax_a.set_title('Exogenous disturbance  d(t)')

text_handle = ax_q.text(
    0.015, 0.65, '', transform=ax_q.transAxes, fontsize=9,
    family='monospace',
    bbox=dict(facecolor='black', edgecolor='cyan', alpha=0.6),
)


def update(frame):
    k = frame
    A = arrival_signal(k)

    u = controller.compute(plant.q)
    q, D = plant.step(u, A)

    t_log.append(k)
    q_log.append(q)
    u_log.append(u)
    A_log.append(A)
    e_log.append(controller.last_error)

    line_q.set_data(t_log, q_log)
    line_r.set_data([0, SIM_STEPS], [0.0, 0.0])
    line_u.set_data(t_log, u_log)
    line_a.set_data(t_log, A_log)

    text_handle.set_text(
        f"Kp = {Kp}    Ki = {Ki}\n"
        f"q(k) = {q:6.2f} veh\n"
        f"u(k) = {u:6.2f} s\n"
        f"e(k) = {controller.last_error:+.2f}\n"
        f"INT  = {controller.integral:+6.2f}"
    )
    return line_q, line_r, line_u, line_a, text_handle


def main():
    ani = FuncAnimation(
        fig, update, frames=SIM_STEPS, interval=80, blit=False, repeat=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
