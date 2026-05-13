"""
plant_model.py
==============
Discrete-time model of the intersection used as the *plant* in the
closed-loop block diagram.

Difference equation (as required by the project specification):

        q(k+1) = q(k) + A(k) - D(k)

where:
    q(k) = queue length at cycle k                          [veh]
    A(k) = arrival rate (vehicles arriving during cycle k)  [veh/cycle]
           - exogenous DISTURBANCE d(k)
    D(k) = departure rate (vehicles departing during cycle k) [veh/cycle]
           - controlled by u(k) via the saturation-flow model

Departure model:
        D(k) = min( q(k) + A(k), s * u(k) )

where s [veh/s] is the saturation flow rate (vehicles that can clear
the stop-line per second of green) and u(k) [s] is the green-time
command issued by the controller. This is the standard
saturation-flow / capacity model used in queuing theory for
signalised intersections (Webster's model).

Linearised around an operating point this gives the discrete plant:

        Q(z)/U(z) = -s / (z - 1)

i.e. an integrator (Type-1) with gain -s. That is exactly why a PI
(rather than P alone) is theoretically justified: the integrator on
the controller cancels the steady-state error, and the integrator on
the plant gives the system inherent low-frequency disturbance
rejection. Stability is set by the location of the closed-loop pole,
which is a function of (Kp, Ki, s).
"""


class IntersectionPlant:
    def __init__(self, saturation_flow=0.5, q0=0.0):
        """
        saturation_flow : vehicles cleared per second of green time [veh/s]
                          (typical real-world value 0.4 - 0.6 for a single lane;
                          the value can be chosen so the simulation feels right.)
        q0              : initial queue length [veh]
        """
        self.s = saturation_flow
        self.q = q0

    def step(self, u, A):
        """
        Advance one discrete time step of the plant.
            u : green time commanded by controller [s]
            A : arrivals during this cycle (disturbance) [veh]
        Returns: (q_next, D)
        """
        capacity = self.s * u
        D = min(self.q + A, capacity)
        self.q = max(0.0, self.q + A - D)
        return self.q, D

    def reset(self, q0=0.0):
        self.q = q0
