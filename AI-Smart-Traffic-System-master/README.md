# Closed-Loop Traffic Signal Control

**A Discrete-Time PI Feedback Control System with YOLOv7 Vision Sensing and Pygame Plant Simulation.**

> EE-330 Linear Control Systems &nbsp;|&nbsp; NUST-SEECS &nbsp;|&nbsp; 6th Semester &nbsp;|&nbsp; Spring 2026

---

## Overview

This project transforms a conventional AI-based traffic detection system into a **strict, mathematically defensible closed-loop control system**. Where the original implementation used rule-based if/else heuristics for signal timing, this version wraps the same intersection in a discrete-time Proportional-Integral feedback law with rigorous stability analysis in the z-domain.

The deep-learning component (YOLOv7) is reframed not as the "AI brain" but as a **measurement element** — the sensor that produces the feedback signal $y(k)$ consumed by a classical PI controller. The control decision itself is fully analytic.

### Block diagram

```
   r(k)=0   +     e(k)    +-----+   u_unsat   +-----+   u(k)   +-------+
   ----+------->O-------->| PI  |------------>| SAT |--------->| PLANT |---> y(k) = q(k)
            ^   - ^       +-----+             +-----+          +-------+    |
            |     |                                              ^         |
            |     |                                            A(k)        |
            |     |                                       (disturbance)    |
            |     +----------+ YOLOv7 Sensor +<-------------------+--------+
            |                  (measurement)
```

| Block         | Implementation               | Source file                 |
|---------------|------------------------------|-----------------------------|
| Plant         | Pygame intersection          | `simulation_controlled.py`  |
| Sensor        | YOLOv7 vehicle detector      | `vehicle_detection.py`      |
| Controller    | Discrete PI law              | `pi_controller.py`          |
| Plant model   | $q(k+1) = q(k) + A(k) - D(k)$ | `plant_model.py`           |
| Stability     | Jury / Root locus / Bode     | `stability_analysis.py`     |

---

## Mathematical foundation

### Plant model (discrete-time difference equation)

$$q(k+1) = q(k) + A(k) - D(k)$$

where $q(k)$ is queue length, $A(k)$ is the stochastic arrival rate (disturbance), and $D(k) = \min(q(k) + A(k), s \cdot u(k))$ is the departure rate, controlled through the saturation flow model with $s \approx 0.5$ veh/s of green.

In z-domain:

$$G_p(z) = \frac{Q(z)}{U(z)} = \frac{-s}{z-1}$$

### PI control law

$$u(k) = u_{\text{base}} + K_p \cdot e(k) + K_i \sum_{i=0}^{k} e(i)\,dt$$

where $e(k) = r(k) - y(k)$ is the error signal and $r(k) = 0$ is the regulator setpoint (ideal empty road). The controller includes **clamping anti-windup** to prevent integrator wind-up during actuator saturation.

In z-domain:

$$C(z) = K_p + \frac{K_i \cdot dt \cdot z}{z-1} = \frac{(K_p + K_i\,dt)\,z - K_p}{z-1}$$

### Closed-loop characteristic polynomial

$$P(z) = z^2 + (s K_p + s K_i\,dt - 2)\,z + (1 - s K_p)$$

At the operating point ($K_p = 1.2$, $K_i = 0.15$, $s = 0.5$, $dt = 1$):

$$P(z) = z^2 - 1.325\,z + 0.4$$

### Closed-loop poles

$$z_1 = 0.860, \quad z_2 = 0.465$$

Both real, both inside the unit circle &rarr; **stable, overdamped response**.

---

## Stability analysis

### Jury (Schur-Cohn) conditions

For $z^2 + b z + c$:

| Condition | Expression | Operating-point value | Verdict |
|-----------|------------|------------------------|---------|
| J1: $P(1) > 0$    | $s K_i\,dt > 0$               | 0.075 | PASS |
| J2: $P(-1) > 0$   | $2 s K_p + s K_i\,dt < 4$     | 1.275 | PASS |
| J3: $\|c\| < 1$   | $0 < s K_p < 2$                | 0.600 | PASS |

### Frequency-domain margins

| Quantity      | Value             |
|---------------|-------------------|
| Phase margin  | **61.4&deg;** at $\omega_{gc} = 0.66$ rad/s |
| Gain margin   | effectively infinite |

### Visual results

| Root locus (Kp and Ki sweeps) | Discrete Bode with margins | Step disturbance response |
|---|---|---|
| ![Root locus](root_locus.png) | ![Bode margins](bode_margins.png) | ![Step response](step_response.png) |

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/jafeeri/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
# or:
source .venv/bin/activate             # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run a demo

```bash
# Smallest demo - asks for queue length, prints PI-derived green time
python signal_time.py

# Standalone closed-loop demo with live matplotlib animation
python control_simulation.py

# Z-domain stability analysis (root locus + Bode + Jury check)
python stability_analysis.py

# Full Pygame intersection + embedded live control panel
python simulation_controlled.py

# Standalone YOLOv7 vehicle detector on a traffic video
python detect_vehicles_video.py path/to/video.mp4
```

### Requirements

- Python 3.10 or 3.11 (3.12+ may have OpenCV wheel issues)
- `numpy`, `matplotlib`, `pygame`, `opencv-python`

`yolov7.weights`, `yolov7.cfg`, and `coco.names` are pre-trained COCO assets included in the repo for the YOLO sensor.

---

## Project structure

```
.
├── pi_controller.py            # Discrete PI control law + clamping anti-windup
├── plant_model.py              # Discrete-time queue model q(k+1)=q(k)+A(k)-D(k)
├── control_simulation.py       # Standalone closed-loop demo with live matplotlib
├── simulation_controlled.py    # Full pygame intersection + embedded control panel
├── simulation.py               # Original baseline (heuristic, no PI)
├── signal_time.py              # Backwards-compatible API; delegates to PIController
├── stability_analysis.py       # Z-domain: Jury conditions, root locus, Bode, margins
├── vehicle_detection.py        # YOLOv7 still-image batch detector
├── detect_vehicles_video.py    # YOLOv7 video detector with live count overlay
│
├── yolov7.weights              # Pre-trained COCO weights (Darknet format)
├── yolov7.cfg                  # YOLOv7 configuration
├── coco.names                  # 80 COCO class names
│
├── images/                     # Pygame sprites (cars, buses, signals, intersection)
├── test_images/                # Sample stills for YOLO detection
├── output_images/              # Annotated outputs from vehicle_detection.py
│
├── root_locus.png              # Pre-rendered z-plane root locus
├── bode_margins.png            # Pre-rendered discrete Bode with PM/GM annotations
├── step_response.png           # Pre-rendered step-disturbance response
│
├── AI_Traffic_Control_Presentation.pptx   # Defense slide deck (15 slides)
├── DEFENSE_CHEATSHEET.md       # Viva preparation cheat sheet
├── PRESENTATION_GUIDE.txt      # Slide-by-slide presenter scripts + Q&A bank
├── TRANSFER_FUNCTIONS.txt      # One-page transfer-function reference card
│
├── requirements.txt
├── README.md                   # this file
└── LICENSE
```

---

## Module reference

### `pi_controller.py`
Implements `PIController` with:
- Discrete PI control law in deviation form
- Clamping (back-calculation) anti-windup
- Actuator saturation `[u_min, u_max]`
- Public `compute(measured_queue)` &rarr; `u(k)`

### `plant_model.py`
Implements `IntersectionPlant`:
- Discrete-time queue evolution $q(k+1) = q(k) + A(k) - D(k)$
- Saturation-flow departure model

### `control_simulation.py`
Standalone closed-loop sim with `FuncAnimation`. Step disturbance at $k = 30$ shows the textbook system response.

### `simulation_controlled.py`
The headline demo. Pygame intersection wrapped in PI feedback, with a live matplotlib panel rendered through the Agg backend and blitted onto a Pygame surface.

### `stability_analysis.py`
Runs the Jury test, plots z-plane root locus across $K_p$ and $K_i$ sweeps, evaluates discrete Bode on the unit circle, and computes gain/phase margins.

### `detect_vehicles_video.py`
Frame-by-frame YOLOv7 inference on a traffic video, with green bounding boxes and live vehicle-count overlay. Demonstrates the sensor side of the loop independently of the simulator.

---

## Tuning

| Parameter | Value | Role |
|-----------|-------|------|
| $K_p$       | 1.2   | Proportional gain |
| $K_i$       | 0.15  | Integral gain |
| $u_{\text{base}}$ | 10 s | Nominal / minimum green time |
| $u_{\text{min}}$  | 10 s | Lower actuator saturation (pedestrian safety) |
| $u_{\text{max}}$  | 60 s | Upper actuator saturation (engineering convention) |
| $dt$        | 1 s   | Controller sampling period |
| $s$         | 0.5 veh/s | Saturation flow rate (Webster's literature) |
| $r$         | 0     | Regulator setpoint |

Gains were tuned by empirical Ziegler-Nichols sweep on the simulator, then verified analytically against the Jury bounds.

---

## Performance results

| Metric | Value | Notes |
|--------|-------|-------|
| Peak overshoot (step disturbance) | 21.2 veh | Surge of arrivals at $k=25$ |
| Settling time | ~26 cycles | Within $\pm 1.5$ veh of $r=0$ |
| Steady-state error | ~0.3 veh | Limited by $u_{\text{max}}$ saturation |
| Phase margin | 61.4&deg; | Above 45&deg; rule of thumb |
| Throughput improvement | ~38 % | vs fixed-time baseline at high arrival rate |
| Dominant time constant | 6.6 cycles | $\tau = -dt / \ln(0.86)$ |

---

## Defense materials

For viva preparation, see:

- **[DEFENSE_CHEATSHEET.md](DEFENSE_CHEATSHEET.md)** &mdash; condensed control-systems Q&A
- **[PRESENTATION_GUIDE.txt](PRESENTATION_GUIDE.txt)** &mdash; full slide-by-slide presenter scripts plus Q&A bank for both control and DL examiners
- **[TRANSFER_FUNCTIONS.txt](TRANSFER_FUNCTIONS.txt)** &mdash; one-page reference card with every transfer function and pole location

---

## Authors

| Name              | Registration |
|-------------------|--------------|
| Syed Ali Medhi    | 454054       |
| Abdullah Latif    | 469180       |
| Haseeb Javaid     | 478317       |
| Obaid             | 460745       |

**Course Instructor:** Dr. Farid Gul
**Lab Engineer:** Yasit Rizwan

---

## Acknowledgements

We are deeply grateful to Dr. Farid Gul for his rigorous guidance throughout EE-330 Linear Control Systems and for insisting on the classical-control discipline that pushed this project from a software demo into a defensible feedback-control design. We also thank Lab Engineer Yasit Rizwan for hands-on lab supervision and his suggestions on the AI module, and NUST-SEECS for the academic environment that made this work possible.

The base simulation environment is adapted from the open-source [AI-Smart-Traffic-System](https://github.com/) project; the control-systems wrapping, mathematical model, stability analysis, and PI controller are original work.

---

## License

This project is released under the terms of the [MIT License](LICENSE).
