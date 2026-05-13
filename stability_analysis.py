"""
stability_analysis.py
=====================
Z-domain stability analysis of the closed-loop traffic-signal PI controller.

Produces three figures:

    1. Root locus in the z-plane as Kp is swept (Ki fixed), and as Ki is
       swept (Kp fixed). The unit circle is drawn as the stability
       boundary; closed-loop poles inside |z|<1 are stable.

    2. Discrete Bode plot of the open-loop transfer function

           L(z) = s * C(z) / (z - 1)

       with the gain crossover frequency, phase crossover frequency,
       gain margin (GM) and phase margin (PM) annotated.

    3. A summary panel listing the analytic Jury stability conditions
       for the chosen (Kp, Ki) and the numerical closed-loop poles.

Derivation (memorise for the viva)
----------------------------------
Plant (linearised, ignoring saturation):

        q(k+1) = q(k) + A(k) - s * u(k)
   =>   (z-1) Q(z) = A(z) - s * U(z)
   =>   G_p(z) = Q(z)/U(z) = -s / (z - 1)             (pole at z=1)

Controller (positional discrete PI in deviation form, e_dev = y - r = y):

        u(k) = u_base + Kp*y(k) + Ki*dt * sum y(i)
   =>   C(z) = Kp + Ki*dt * z / (z - 1)               (pole at z=1, zero at z=Kp/(Kp+Ki*dt))

Closed loop  Q(z)/A(z) = 1 / [ (z-1) + s*C(z) ]

Multiplying through by (z-1):

   Char. eqn:   z^2  +  (s*Kp + s*Ki*dt - 2) * z  +  (1 - s*Kp)  =  0

Jury (Schur-Cohn) stability conditions for z^2 + b*z + c = 0:

   (J1)  P(1)  > 0  =>  s*Ki*dt > 0
   (J2)  P(-1) > 0  =>  2*s*Kp + s*Ki*dt < 4
   (J3)  |c|   < 1  =>  0 < s*Kp < 2

Run:
    python stability_analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Default operating point (matches control_simulation.py defaults)
# ---------------------------------------------------------------------------
KP_DEFAULT = 1.2
KI_DEFAULT = 0.15
S_DEFAULT  = 0.5     # saturation flow [veh / s of green]
DT_DEFAULT = 1.0     # sampling period (1 cycle)


# ---------------------------------------------------------------------------
# Closed-loop poles + Jury conditions
# ---------------------------------------------------------------------------
def closed_loop_poles(Kp, Ki, s=S_DEFAULT, dt=DT_DEFAULT):
    """Roots of z^2 + (s*Kp + s*Ki*dt - 2)*z + (1 - s*Kp) = 0."""
    b = s * Kp + s * Ki * dt - 2.0
    c = 1.0 - s * Kp
    return np.roots([1.0, b, c])


def jury_check(Kp, Ki, s=S_DEFAULT, dt=DT_DEFAULT):
    j1 = s * Ki * dt > 0
    j2 = 2 * s * Kp + s * Ki * dt < 4
    j3 = 0 < s * Kp < 2
    return {
        "J1 s*Ki*dt > 0":              (j1, s * Ki * dt),
        "J2 2*s*Kp + s*Ki*dt < 4":     (j2, 2 * s * Kp + s * Ki * dt),
        "J3 0 < s*Kp < 2":             (j3, s * Kp),
        "all_stable":                  (j1 and j2 and j3),
    }


# ---------------------------------------------------------------------------
# Open-loop frequency response (discrete Bode)
# ---------------------------------------------------------------------------
def open_loop(Kp, Ki, s=S_DEFAULT, dt=DT_DEFAULT, n_pts=2000):
    """
    Evaluate L(z) = s * C(z) / (z - 1) on z = e^{j w dt} for w in (0, pi/dt).
    Returns (omega, mag_dB, phase_deg).
    """
    omega = np.logspace(-3, np.log10(np.pi / dt - 1e-4), n_pts)
    z = np.exp(1j * omega * dt)
    C = Kp + Ki * dt * z / (z - 1)
    L = s * C / (z - 1)
    mag_db    = 20.0 * np.log10(np.abs(L))
    phase_deg = np.unwrap(np.angle(L)) * 180.0 / np.pi
    return omega, mag_db, phase_deg


def stability_margins(omega, mag_db, phase_deg):
    """
    Returns (GM_dB, PM_deg, w_gc, w_pc).

    Gain crossover  w_gc : |L| = 1   (mag_db = 0)
    Phase crossover w_pc : phase = -180 deg
    PM = 180 + phase(w_gc)
    GM = -mag_db(w_pc)        (in dB)
    """
    pm, gm, w_gc, w_pc = None, None, None, None

    sgn = np.sign(mag_db)
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross):
        i = cross[0]
        w_gc = omega[i]
        # linear interp between i and i+1 on omega
        m1, m2 = mag_db[i], mag_db[i + 1]
        p1, p2 = phase_deg[i], phase_deg[i + 1]
        frac = m1 / (m1 - m2) if (m1 - m2) != 0 else 0.0
        phase_at = p1 + frac * (p2 - p1)
        pm = 180.0 + phase_at

    sgn = np.sign(phase_deg + 180.0)
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross):
        # take the FIRST crossing AFTER w_gc if it exists, else the first
        candidates = cross
        if w_gc is not None:
            after = [c for c in cross if omega[c] > w_gc]
            if after:
                candidates = after
        i = candidates[0]
        w_pc = omega[i]
        gm = -mag_db[i]

    return gm, pm, w_gc, w_pc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_root_locus(ax, sweep_var, sweep_vals, fixed_kw, title):
    """
    sweep_var in {'Kp','Ki'}.
    fixed_kw  has the *other* parameter plus s, dt.
    """
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color='red', ls='--', lw=1.0,
            label='|z| = 1  (stability boundary)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    real_pts, imag_pts, c_pts = [], [], []
    for v in sweep_vals:
        if sweep_var == 'Kp':
            poles = closed_loop_poles(v, fixed_kw['Ki'],
                                      fixed_kw['s'], fixed_kw['dt'])
        else:
            poles = closed_loop_poles(fixed_kw['Kp'], v,
                                      fixed_kw['s'], fixed_kw['dt'])
        for p in poles:
            real_pts.append(p.real)
            imag_pts.append(p.imag)
            c_pts.append(v)

    sc = ax.scatter(real_pts, imag_pts, c=c_pts, cmap='viridis', s=14, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=sweep_var)

    # Mark the operating point
    if sweep_var == 'Kp':
        op_poles = closed_loop_poles(KP_DEFAULT, fixed_kw['Ki'],
                                     fixed_kw['s'], fixed_kw['dt'])
        op_label = f"Operating  Kp={KP_DEFAULT}, Ki={fixed_kw['Ki']}"
    else:
        op_poles = closed_loop_poles(fixed_kw['Kp'], KI_DEFAULT,
                                     fixed_kw['s'], fixed_kw['dt'])
        op_label = f"Operating  Kp={fixed_kw['Kp']}, Ki={KI_DEFAULT}"

    ax.scatter(op_poles.real, op_poles.imag, marker='x', s=120, c='red',
               linewidths=2.5, label=op_label, zorder=5)

    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)


def plot_bode(Kp, Ki, s=S_DEFAULT, dt=DT_DEFAULT):
    omega, mag, phase = open_loop(Kp, Ki, s, dt)
    gm, pm, w_gc, w_pc = stability_margins(omega, mag, phase)

    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Discrete Bode of open-loop  L(z) = s·C(z)/(z-1)   "
                 f"[Kp={Kp}, Ki={Ki}, s={s}, dt={dt}]", fontsize=11)

    ax_m.semilogx(omega, mag, color='cyan', lw=1.8)
    ax_m.axhline(0, color='gray', ls='--', lw=0.8)
    ax_m.set_ylabel('|L(jω)|  [dB]')
    ax_m.grid(which='both', alpha=0.3)

    ax_p.semilogx(omega, phase, color='lime', lw=1.8)
    ax_p.axhline(-180, color='gray', ls='--', lw=0.8)
    ax_p.set_ylabel('∠L(jω)  [deg]')
    ax_p.set_xlabel('ω  [rad/s]   (Nyquist  ωN = π/dt)')
    ax_p.grid(which='both', alpha=0.3)

    if w_gc is not None:
        ax_m.axvline(w_gc, color='orange', ls=':', lw=1.0)
        ax_p.axvline(w_gc, color='orange', ls=':', lw=1.0)
        pm_txt = "∞" if pm is None else f"{pm:.1f}°"
        ax_p.annotate(f"PM = {pm_txt}\nω_gc = {w_gc:.3f}",
                      xy=(w_gc, -180), xytext=(w_gc, -120),
                      arrowprops=dict(arrowstyle='->', color='orange'),
                      color='orange', fontsize=9,
                      bbox=dict(facecolor='black', alpha=0.6, edgecolor='orange'))

    if w_pc is not None and gm is not None:
        ax_m.axvline(w_pc, color='magenta', ls=':', lw=1.0)
        ax_p.axvline(w_pc, color='magenta', ls=':', lw=1.0)
        ax_m.annotate(f"GM = {gm:.1f} dB\nω_pc = {w_pc:.3f}",
                      xy=(w_pc, 0), xytext=(w_pc, 30),
                      arrowprops=dict(arrowstyle='->', color='magenta'),
                      color='magenta', fontsize=9,
                      bbox=dict(facecolor='black', alpha=0.6, edgecolor='magenta'))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, gm, pm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plt.style.use('dark_background')

    Kp, Ki, s, dt = KP_DEFAULT, KI_DEFAULT, S_DEFAULT, DT_DEFAULT

    # ---- Console summary of analytic stability test ----
    print("=" * 64)
    print(" Z-Domain stability analysis -- closed-loop traffic signal")
    print("=" * 64)
    print(f"Operating point: Kp={Kp}, Ki={Ki}, s={s} veh/s, dt={dt} s\n")
    print(" Plant      G_p(z) = -s / (z - 1)")
    print(" PI         C(z)   = Kp + Ki*dt * z / (z - 1)")
    print(" Char. eqn  z^2 + (s*Kp + s*Ki*dt - 2)*z + (1 - s*Kp) = 0\n")

    poles = closed_loop_poles(Kp, Ki, s, dt)
    print(" Closed-loop poles:")
    for p in poles:
        inside = "INSIDE  unit circle  (stable)" if abs(p) < 1 else "OUTSIDE unit circle  (UNSTABLE)"
        print(f"    z = {p.real:+.4f} {p.imag:+.4f}j   |z| = {abs(p):.4f}   {inside}")

    j = jury_check(Kp, Ki, s, dt)
    print("\n Jury (Schur-Cohn) stability conditions:")
    for k, v in j.items():
        if k == "all_stable":
            print(f"    Overall verdict          : {'STABLE [OK]' if v else 'UNSTABLE [!!]'}")
        else:
            ok, val = v
            mark = "[OK]" if ok else "[!!]"
            print(f"    {k:<32s}  =  {val:+.4f}   {mark}")

    # ---- Root locus figures ----
    fig1, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_root_locus(
        axes[0], 'Kp', np.linspace(0.0, 4.0, 200),
        {'Ki': Ki, 's': s, 'dt': dt},
        f'Root locus  —  sweep Kp ∈ [0, 4]  (Ki = {Ki})'
    )
    plot_root_locus(
        axes[1], 'Ki', np.linspace(0.0, 1.0, 200),
        {'Kp': Kp, 's': s, 'dt': dt},
        f'Root locus  —  sweep Ki ∈ [0, 1]  (Kp = {Kp})'
    )
    fig1.tight_layout()
    fig1.savefig('root_locus.png', dpi=120)
    print("\n Saved: root_locus.png")

    # ---- Bode + margins ----
    fig2, gm, pm = plot_bode(Kp, Ki, s, dt)
    fig2.savefig('bode_margins.png', dpi=120)
    print(" Saved: bode_margins.png")
    if gm is not None:
        print(f"\n Gain margin  : {gm:6.2f} dB")
    else:
        print("\n Gain margin  : INF  (phase never crosses -180 deg above w_gc)")
    if pm is not None:
        print(f" Phase margin : {pm:6.2f} deg")
    else:
        print(" Phase margin : INF  (no gain crossover)")

    print("\n Showing figures — close windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
