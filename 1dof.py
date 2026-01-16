# forced damped oscillator demo for a bridge-like sdof model
# goal: show resonance and how individual fourier harmonics contribute
# outputs: plots saved under ./plots
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import matplotlib.cm as cm
import csv

os.makedirs("plots", exist_ok=True)

############################### Bridge Model ####################################

#   m x'' + c x' + k x = f(t)
# choosing m and k so that the natural frequency matches a target value
# (this is a convenient way to set a realistic frequency scale)

# rough tacoma narrows (1940) fryesequency scale
# important: the real event involved torsional motion + aeroelastic flutter
# here we are not modelling flutter; we only match the rough natural frequency scale
# using an equivalent sdof setup
f_n_target = 0.2          # frequency reported during the incident in 1940
# omega = 2*pi/T where T=1/f => omega = 2*pi*f
omega_n_target = 2.0 * np.pi * f_n_target

# rough effective (lumped) mass estimate
# this is an order-of-magnitude pick so that m is in a realistic range
# (effective modal mass is not the full bridge mass, but this is ok for a simple demo)
m = 6.7e6                 # kg

# choosing a stiffness that fits the target frequency
# for an undamped sdof oscillator: omega_n = sqrt(k/m)  ->  k = m * omega_n^2
k = m * omega_n_target**2  # N/m

# damping ratio
# smaller zeta means a sharper resonance peak and larger amplification near omega ~= omega_n
zeta = 0.005                # damping ratio (0.5%)

# derived quantities we will reuse
# omega_n: natural angular frequency (rad/s)
# f_n: natural frequency (hz)
# c: viscous damping coefficient computed from zeta
omega_n = np.sqrt(k / m)
c = 2.0 * zeta * m * omega_n
#  omega = 2*pi*f => f = omega/2*pi
f_n = omega_n / (2.0 * np.pi)

print(f"omega_n = {omega_n:.4f} rad/s, f_n = {f_n:.4f} Hz, c = {c:.3e} N*s/m")

############################### Single Sinusoidal Forcing ####################################

# -----------------------------
# 2) frequency response (steady-state amplitude vs forcing frequency)
# -----------------------------
# single sinusoidal force: f(t) = f0 cos(omega t)
# for a linear sdof system, the steady-state amplitude has a closed form
# plotting that amplitude against omega/omega_n shows the resonance peak

# forcing amplitude (newtons)
# this just scales the displacement up/down; it does not move the resonance location
F0 = 1.0e6  # N (amplitude of forcing)

# sweep omega across a range around the natural frequency
omegas = np.linspace(0.1 * omega_n, 3.0 * omega_n, 800)

# steady-state amplitude for harmonic forcing
# note: this is the textbook frequency response for a damped sdof oscillator
A = (F0 / m) / np.sqrt((omega_n**2 - omegas**2)**2 + (2*zeta*omega_n*omegas)**2)

plt.figure()
plt.plot(omegas / omega_n, A)
plt.axvline(1.0, linestyle="--")
plt.xlabel(r"Forcing frequency ratio $\omega/\omega_n$")
plt.ylabel("Steady-state amplitude A (m)")
plt.title("Frequency response of forced damped oscillator")
plt.grid(True)
plt.savefig("plots/frequency_response.png", dpi=300, bbox_inches="tight")
plt.close()


############################### ODE Solution ####################################

# -----------------------------
# 3) time-domain simulation (solve the ode)
# -----------------------------
# now we integrate the ode numerically to see transients + steady state in time
# we use solve_ivp on the 1st-order system [x, v] where v = x'

def simulate_forcing(F_func, t_end=200.0, x0=0.0, v0=0.0, n_points=6000):
    """simulate x(t) for a chosen forcing function f(t)."""
    def rhs(t, y):
        x, v = y  # state: displacement and velocity
        dxdt = v  # x' = v
        # v' from m v' + c v + k x = f(t)
        dvdt = (F_func(t) - c * v - k * x) / m
        return [dxdt, dvdt]

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), [x0, v0], t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]

# compare several forcing frequencies across the resonance region
cases = [
    (r"$\omega/\omega_n=0.3$", 0.3 * omega_n),
    (r"$\omega/\omega_n=0.7$", 0.7 * omega_n),
    (r"$\omega/\omega_n=1.0$ (resonance)", 1.0 * omega_n),
    (r"$\omega/\omega_n=1.3$", 1.3 * omega_n),
    (r"$\omega/\omega_n=1.7$", 1.7 * omega_n),
]

plt.figure()
for label, omega in cases:
    F = lambda t, om=omega: F0 * np.cos(om * t)
    t, x, v = simulate_forcing(F, t_end=200.0)
    # we keep the full signal here; you could also crop t to show only steady state
    plt.plot(t, x, label=f"{label}")

plt.xlabel("Time t (s)")
plt.ylabel("Displacement x(t) (m)")
plt.title("Time response for different forcing frequencies")
plt.grid(True)
plt.legend()
plt.savefig("plots/time_response_harmonic.png", dpi=300, bbox_inches="tight")
plt.close()


############################### Fourier Series Forcing ####################################

# -----------------------------
# 4) fourier-series forcing (multiple harmonics)
# -----------------------------
# a periodic load can be written as a sum of sines/cosines
# f(t) = a0 + sum_n [a_n cos(n omega t) + b_n sin(n omega t)]
# here omega_base is the fundamental, and n * omega_base are the harmonics

# sweep omega_base = m_base * omega_n
# - fine sweep for CSV/table (0.01 increments)
# - coarse sweep for saving only a few illustrative time-series plots (0.10 increments)
m_values_fine = np.arange(0.10, 1.001, 0.01)
m_values_plot = np.arange(0.10, 1.001, 0.10)

# choose ONE value for the detailed time-series plots below
# (we keep these plots so you still have illustrative figures in the report)
m_base_demo = 0.1
omega_base = m_base_demo * omega_n

# If you want a *time-series plot for every m_base* in the sweep, set this to True.
# Warning: this will create ~91 PNG files.
PLOT_ALL_M_TIME_SERIES = True

# transient crop for the time-series plots (seconds)
plot_transient_cut = 50.0

# fourier coefficients (newtons)
# larger coefficients mean that harmonic contributes more forcing energy
a0 = 0.0
a = {1: 1.0e6, 2: 0.4e6, 3: 0.2e6}  # cosine coefficients (N)
b = {1: 0.2e6, 2: 0.0, 3: 0.1e6}    # sine coefficients (N)

def F_fourier(t):
    val = a0
    for n, an in a.items():
        val += an * np.cos(n * omega_base * t)
    for n, bn in b.items():
        val += bn * np.sin(n * omega_base * t)
    return val

t, x, v = simulate_forcing(F_fourier, t_end=200.0)

# total response when all harmonics act together (this can look complex)
plt.figure()
steady_plot = t > plot_transient_cut
plt.plot(t[steady_plot], x[steady_plot])
plt.xlabel("Time t (s)")
plt.ylabel("Displacement x(t) (m)")
plt.title(rf"Time response with Fourier-series forcing (combined), $\omega_{{base}}/\omega_n={m_base_demo:.2f}$")
plt.grid(True)
plt.savefig("plots/time_response_fourier.png", dpi=300, bbox_inches="tight")
plt.close()

# to make it readable, also simulate each harmonic on its own and plot it in a different color
# this helps show which harmonic is closest to resonance and dominates the response
N_max = 10
harmonics = sorted(set(list(a.keys()) + list(b.keys())))
harmonics = [n for n in harmonics if 1 <= n <= N_max]

colors = cm.viridis(np.linspace(0, 1, max(1, len(harmonics))))

plt.figure(figsize=(10, 6))
for i, n in enumerate(harmonics):
    an = a.get(n, 0.0)
    bn = b.get(n, 0.0)

    def F_n(t, n=n, an=an, bn=bn):
        return an * np.cos(n * omega_base * t) + bn * np.sin(n * omega_base * t)

    t_n, x_n, _ = simulate_forcing(F_n, t_end=200.0)

    # only plot the later part so the start-up transient does not clutter the figure
    steady = t_n > plot_transient_cut

    plt.plot(
        t_n[steady],
        x_n[steady],
        color=colors[i],
        label=rf"$n={n}$ ($n\omega/\omega_n$={(n*omega_base)/omega_n:.2f})"
    )

plt.xlabel("Time t (s)")
plt.ylabel("Displacement x(t) (m)")
plt.title(rf"Responses to individual Fourier harmonics (first 10), $\omega_{{base}}/\omega_n={m_base_demo:.2f}$")
plt.grid(True)
plt.legend(fontsize=9)
plt.savefig("plots/time_response_fourier_harmonics.png", dpi=300, bbox_inches="tight")
plt.close()

###############################
# 4a-extra) Optional: generate a time-series plot for every m_base in the sweep
###############################

if PLOT_ALL_M_TIME_SERIES:
    for m_base in m_values_plot:
        omega_base_loop = m_base * omega_n

        def F_fourier_loop(t, omega_base_loop=omega_base_loop):
            val = a0
            for n, an in a.items():
                val += an * np.cos(n * omega_base_loop * t)
            for n, bn in b.items():
                val += bn * np.sin(n * omega_base_loop * t)
            return val

        t_loop, x_loop, _ = simulate_forcing(F_fourier_loop, t_end=200.0, n_points=3000)
        steady_loop = t_loop > plot_transient_cut

        plt.figure()
        plt.plot(t_loop[steady_loop], x_loop[steady_loop])
        plt.xlabel("Time t (s)")
        plt.ylabel("Displacement x(t) (m)")
        plt.title(rf"Fourier forcing response, $\omega_{{base}}/\omega_n={m_base:.2f}$")
        plt.grid(True)
        plt.savefig(f"plots/time_response_fourier_m_{m_base:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # also save the decomposed responses (each harmonic separately) for this m_base
        harmonics_plot = sorted(set(list(a.keys()) + list(b.keys())))
        harmonics_plot = [n for n in harmonics_plot if 1 <= n <= N_max]

        colors_plot = cm.viridis(np.linspace(0, 1, max(1, len(harmonics_plot))))

        plt.figure(figsize=(10, 6))
        for i, n in enumerate(harmonics_plot):
            an = a.get(n, 0.0)
            bn = b.get(n, 0.0)

            def F_n_loop(t, n=n, an=an, bn=bn, omega_base_loop=omega_base_loop):
                return an * np.cos(n * omega_base_loop * t) + bn * np.sin(n * omega_base_loop * t)

            t_n, x_n, _ = simulate_forcing(F_n_loop, t_end=200.0, n_points=3000)
            steady_n = t_n > plot_transient_cut

            plt.plot(
                t_n[steady_n],
                x_n[steady_n],
                color=colors_plot[i],
                label=rf"$n={n}$ ($n\omega/\omega_n$={(n*omega_base_loop)/omega_n:.2f})"
            )

        plt.xlabel("Time t (s)")
        plt.ylabel("Displacement x(t) (m)")
        plt.title(rf"Decomposed harmonic responses, $\omega_{{base}}/\omega_n={m_base:.2f}$")
        plt.grid(True)
        plt.legend(fontsize=9)
        plt.savefig(f"plots/time_response_fourier_harmonics_m_{m_base:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(
        f"\nSaved {len(m_values_plot)} time-series plots to plots/time_response_fourier_m_*.png "
        f"and decomposed plots to plots/time_response_fourier_harmonics_m_*.png"
    )

# quick printout (for the demo plots): harmonic frequency ratios relative to omega_n
# values near 1.0 mean that harmonic is close to resonance
print("\nHarmonics relative to omega_n:")
for n in sorted(set(list(a.keys()) + list(b.keys()))):
    print(f"n={n}: n*omega_base/omega_n = {(n*omega_base)/omega_n:.3f}")


###############################
# 4b) Sweep omega_base and track steady-state amplitudes (for a table later)
###############################

# which harmonics to track (only those present in the coefficients, capped by N_max)
tracked_harmonics = sorted(set(list(a.keys()) + list(b.keys())))
tracked_harmonics = [n for n in tracked_harmonics if 1 <= n <= N_max]

# transient cutoff for amplitude estimation (seconds)
# increase this if you still see startup effects for low damping
transient_cut = 50.0

results = []  # each entry is a dict, one row per m_base

for m_base in m_values_fine:
    omega_base_sweep = m_base * omega_n

    def F_fourier_sweep(t, omega_base_sweep=omega_base_sweep):
        val = a0
        for n, an in a.items():
            val += an * np.cos(n * omega_base_sweep * t)
        for n, bn in b.items():
            val += bn * np.sin(n * omega_base_sweep * t)
        return val

    # simulate combined forcing (use fewer points to keep the sweep manageable)
    t_s, x_s, _ = simulate_forcing(F_fourier_sweep, t_end=200.0, n_points=3000)
    steady = t_s > transient_cut
    amp_combined = float(np.max(np.abs(x_s[steady])))

    row = {
        "m_base": float(m_base),
        "omega_base": float(omega_base_sweep),
        "amp_combined_max": amp_combined,
    }

    # also simulate each tracked harmonic alone and store its steady-state max amplitude
    amp_harmonics = {}
    for n in tracked_harmonics:
        an = a.get(n, 0.0)
        bn = b.get(n, 0.0)

        def F_n_sweep(t, n=n, an=an, bn=bn, omega_base_sweep=omega_base_sweep):
            return an * np.cos(n * omega_base_sweep * t) + bn * np.sin(n * omega_base_sweep * t)

        t_n, x_n, _ = simulate_forcing(F_n_sweep, t_end=200.0, n_points=3000)
        steady_n = t_n > transient_cut
        amp_n = float(np.max(np.abs(x_n[steady_n])))
        amp_harmonics[n] = amp_n
        row[f"amp_h{n}_max"] = amp_n
        row[f"ratio_h{n}"] = float((n * omega_base_sweep) / omega_n)

    # which harmonic dominates (largest max amplitude)
    if len(amp_harmonics) > 0:
        dom_n = max(amp_harmonics, key=lambda kk: amp_harmonics[kk])
        row["dominant_harmonic"] = int(dom_n)
        row["dominant_amp_max"] = float(amp_harmonics[dom_n])
        row["dominant_ratio"] = float((dom_n * omega_base_sweep) / omega_n)
    else:
        row["dominant_harmonic"] = ""
        row["dominant_amp_max"] = ""
        row["dominant_ratio"] = ""

    results.append(row)

# write CSV (one row per m_base) so you can paste it into a table later
csv_path = "plots/fourier_sweep_amplitudes.csv"

# header order: base info first, then harmonic columns
fieldnames = ["m_base", "omega_base", "amp_combined_max"]
for n in tracked_harmonics:
    fieldnames += [f"ratio_h{n}", f"amp_h{n}_max"]
fieldnames += ["dominant_harmonic", "dominant_ratio", "dominant_amp_max"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved sweep amplitude table to: {csv_path}")

# quick summary plot: max steady-state amplitude vs omega_base/omega_n
plt.figure()
plt.plot([r["m_base"] for r in results], [r["amp_combined_max"] for r in results])
plt.xlabel(r"$\omega_{base}/\omega_n$")
plt.ylabel("Max steady-state amplitude (m)")
plt.title("Maximum steady-state amplitude vs base forcing frequency")
plt.grid(True)
plt.savefig("plots/max_amplitude_vs_omega_base.png", dpi=300, bbox_inches="tight")
plt.close()



###############################
# Amplitude vs frequency for different damping ratios
###############################

zeta_values = [0.002, 0.005, 0.01, 0.02]

plt.figure()

for z in zeta_values:
    A_z = (F0 / m) / np.sqrt(
        (omega_n**2 - omegas**2)**2 +
        (2 * z * omega_n * omegas)**2
    )
    plt.plot(
        omegas / omega_n,
        A_z,
        label=rf"$\zeta = {z}$"
    )

plt.axvline(1.0, color="k", linestyle="--", alpha=0.6)
plt.xlabel(r"Forcing frequency ratio $\omega/\omega_n$")
plt.ylabel("Steady-state amplitude A (m)")
plt.title("Effect of damping on steady-state amplitude")
plt.grid(True)
plt.legend()
plt.ylim(0, 6)
plt.savefig("plots/amplitude_vs_damping.png", dpi=300, bbox_inches="tight")
plt.close()