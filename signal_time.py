"""
signal_time.py
==============
Original purpose: heuristic if/else update of green/yellow/red signal times.

This file has been refactored to delegate the green-time decision to a
proper discrete-time Proportional-Integral controller (see pi_controller.py).
The same TrafficSignalController public surface is preserved so any code
that imported the previous version still works, but internally the
update_signal_timings(vehicle_count) method now executes the control law:

        u(k) = u_base + Kp * e(k) + Ki * SUM(e(i) * dt)

with reference r = 0 (zero queue) and measured queue y = vehicle_count.
"""
from pi_controller import PIController


class TrafficSignalController:
    def __init__(self, Kp=1.2, Ki=0.15,
                 min_green_time=10, max_green_time=60,
                 yellow_time=5, red_time=20):
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.max_yellow_time = 10
        self.min_yellow_time = 3
        self.max_red_time = 40
        self.min_red_time = 10

        # PI controller drives the green-time command u(k)
        self.controller = PIController(
            Kp=Kp, Ki=Ki,
            u_base=min_green_time,
            u_min=min_green_time,
            u_max=max_green_time,
            dt=1.0,
            reference=0.0,
        )

        self.green_time = min_green_time
        self.yellow_time = yellow_time
        self.red_time = red_time

    # ------------------------------------------------------------------
    # Closed-loop update: replaces the old if/else block
    # ------------------------------------------------------------------
    def update_signal_timings(self, vehicle_count):
        """
        Closed-loop update.
            measurement   y(k) = vehicle_count   (queue length, from YOLO)
            reference     r(k) = 0
            error         e(k) = r(k) - y(k)
            control       u(k) = u_base + Kp e(k) + Ki INT e(k)
        Returns the new green time u(k).
        """
        u = self.controller.compute(vehicle_count)
        self.green_time = int(round(u))

        # red and yellow are derived from green so that the cycle remains
        # phase-balanced; this is bookkeeping, not part of the control law.
        self.red_time = max(self.min_red_time,
                            min(self.max_red_time, int(self.green_time * 1.4)))
        self.yellow_time = max(self.min_yellow_time,
                               min(self.max_yellow_time, 5))
        return u

    def reset(self):
        self.controller.reset()

    def print_signal_timings(self):
        print(f"Green Signal Time : {self.green_time} s   "
              f"(u(k), control output)")
        print(f"Yellow Signal Time: {self.yellow_time} s")
        print(f"Red Signal Time   : {self.red_time} s")
        print(f"Controller state  : e(k)={self.controller.last_error:+.2f}, "
              f"INT={self.controller.integral:+.2f}")


def main():
    sc = TrafficSignalController(Kp=1.2, Ki=0.15)
    try:
        n = int(input("Enter the vehicle count (queue length y(k)): "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return
    sc.update_signal_timings(n)
    sc.print_signal_timings()


if __name__ == "__main__":
    main()
