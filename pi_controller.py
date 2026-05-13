"""
pi_controller.py
================
Discrete-time Proportional-Integral (PI) controller for adaptive traffic
signal green-time computation.

Control law (textbook PI form, as required by the project specification):

        u(k) = u_base + Kp * e(k) + Ki * SUM_{i=0..k} e(i) * dt

where:
    r(k)    = reference (desired queue length)        [veh]
    y(k)    = measured queue length (from YOLO)       [veh]
    e(k)    = r(k) - y(k)                             [veh]
    u(k)    = control effort (green time)             [s]
    u_base  = nominal/minimum green time              [s]
    Kp, Ki  = proportional and integral gains
    dt      = sampling period (one signal cycle)      [s]

Because the reference r(k) = 0 and the measured queue y(k) >= 0, the raw
error e(k) = r(k) - y(k) is non-positive. To keep Kp and Ki positive we
work with the deviation magnitude:

        e_dev(k) = -e(k) = y(k) - r(k) = q(k)

This is mathematically identical to the original equation with negative
gains; using e_dev keeps the gains conventional and the sign of u
intuitive (more queue -> more green time).

Saturation and integral anti-windup (clamping/back-calculation) are
included so that integrator wind-up cannot drive u(k) to non-physical
values during prolonged saturation.
"""


class PIController:
    def __init__(self, Kp=1.2, Ki=0.15, u_base=10.0,
                 u_min=10.0, u_max=60.0, dt=1.0, reference=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.u_base = u_base
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.reference = reference

        self.integral = 0.0
        self.last_error = 0.0
        self.last_u = u_base

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_u = self.u_base

    def compute(self, measured_queue):
        """
        One discrete control step.

        Returns u(k), the green-time command, saturated to [u_min, u_max].
        """
        # e(k) = r(k) - y(k); deviation magnitude e_dev(k) = -e(k) = q(k) - r
        error_signed = self.reference - measured_queue
        e_dev = -error_signed

        # tentative integral update (rectangular integration)
        new_integral = self.integral + e_dev * self.dt

        # PI control law in deviation form (positive gains)
        u_unsat = self.u_base + self.Kp * e_dev + self.Ki * new_integral

        # actuator saturation
        u = max(self.u_min, min(self.u_max, u_unsat))

        # anti-windup (clamping):
        #   only commit the integral update if the controller is not
        #   saturated, OR if the new error pushes the controller back
        #   toward the linear region. This prevents integral wind-up
        #   when the green time hits its physical upper bound.
        saturated_high = u_unsat > self.u_max
        saturated_low  = u_unsat < self.u_min
        if not (saturated_high or saturated_low):
            self.integral = new_integral
        elif saturated_high and e_dev < 0:
            self.integral = new_integral
        elif saturated_low and e_dev > 0:
            self.integral = new_integral
        # else: clamp the integrator (do not accumulate further)

        self.last_error = error_signed
        self.last_u = u
        return u


if __name__ == "__main__":
    # Quick self-test: simulate a step input on the plant input
    pi = PIController(Kp=1.2, Ki=0.15, u_base=10, u_min=10, u_max=60, dt=1.0)
    for q in [0, 5, 10, 20, 35, 35, 35, 20, 10, 5, 0]:
        u = pi.compute(q)
        print(f"q={q:>4}  e={pi.last_error:+.1f}  I={pi.integral:+6.2f}  u={u:5.2f}")
