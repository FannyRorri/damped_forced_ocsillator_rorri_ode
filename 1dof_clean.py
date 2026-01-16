"""
Forced Damped Oscillator Simulation
===================================
Demonstrates resonance behaviour and Fourier harmonic contributions
for a single-degree-of-freedom bridge model.

Outputs: plots saved to ./plots/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
import os

os.makedirs("plots", exist_ok=True)


# =============================================================================
# Model Parameters
# =============================================================================

# Target natural frequency (based on Tacoma Narrows, 1940)
# Note: actual failure involved torsional flutter, not simple resonance
f_n_target = 0.2  # Hz
omega_n_target = 2.0 * np.pi * f_n_target

# Effective modal mass and stiffness
m = 6.7e6  # kg
k = m * omega_n_target**2  # N/m (chosen to match target frequency)

# Damping ratio (smaller = sharper resonance peak)
zeta = 0.005

# Derived quantities
omega_n = np.sqrt(k / m)  # natural angular frequency (rad/s)
c = 2.0 * zeta * m * omega_n  # viscous damping coefficient (N·s/m)
f_n = omega_n / (2.0 * np.pi)  # natural frequency (Hz)

print(f"omega_n = {omega_n:.4f} rad/s, f_n = {f_n:.4f} Hz, c = {c:.3e} N·s/m")


# =============================================================================
# Frequency Response (Steady-State Amplitude)
# =============================================================================

F0 = 1.0e6  # forcing amplitude (N)
omegas = np.linspace(0.1 * omega_n, 3.0 * omega_n, 800)

# Analytical steady-state amplitude
A = (F0 / m) / np.sqrt((omega_n**2 - omegas**2)**2 + (2 * zeta * omega_n * omegas)**2)

plt.figure()
plt.plot(omegas / omega_n, A)
plt.axvline(1.0, linestyle="--")
plt.xlabel(r"Forcing frequency ratio $\omega/\omega_n$")
plt.ylabel("Steady-state amplitude (m)")
plt.title("Frequency response of forced damped oscillator")
plt.grid(True)
plt.savefig("plots/frequency_response.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# Time-Domain Simulation
# =============================================================================

def simulate_forcing(F_func, t_end=200.0, x0=0.0, v0=0.0, n_points=6000):
    """
    Integrate the equation of motion: m x'' + c x' + k x = F(t)

    Parameters
    ----------
    F_func : callable
        Forcing function F(t)
    t_end : float
        Simulation duration (s)
    x0, v0 : float
        Initial displacement and velocity
    n_points : int
        Number of output time points

    Returns
    -------
    t, x, v : arrays
        Time, displacement, and velocity
    """
    def rhs(t, y):
        x, v = y
        dxdt = v
        dvdt = (F_func(t) - c * v - k * x) / m
        return [dxdt, dvdt]

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), [x0, v0], t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]


# Compare responses at different forcing frequencies
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
    plt.plot(t, x, label=label)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Time response for different forcing frequencies")
plt.grid(True)
plt.legend()
plt.savefig("plots/time_response_harmonic.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# Fourier Series Forcing
# =============================================================================

# Sweep parameters
m_values_fine = np.arange(0.10, 1.001, 0.01)  # for CSV output
m_values_plot = np.arange(0.10, 1.001, 0.10)  # for time-series plots

# Demo value for detailed plots
m_base_demo = 0.1
omega_base = m_base_demo * omega_n

# Set True to generate plots for all m_base values
PLOT_ALL_M_TIME_SERIES = True

# Transient cutoff for plotting (s)
plot_transient_cut = 50.0

# Fourier coefficients (N)
a0 = 0.0
a = {1: 1.0e6, 2: 0.4e6, 3: 0.2e6}  # cosine coefficients
b = {1: 0.2e6, 2: 0.0, 3: 0.1e6}    # sine coefficients


def F_fourier(t):
    """Fourier series forcing function."""
    val = a0
    for n, an in a.items():
        val += an * np.cos(n * omega_base * t)
    for n, bn in b.items():
        val += bn * np.sin(n * omega_base * t)
    return val


# Combined response
t, x, v = simulate_forcing(F_fourier, t_end=200.0)
steady = t > plot_transient_cut

plt.figure()
plt.plot(t[steady], x[steady])
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title(rf"Fourier forcing response (combined), $\omega_{{base}}/\omega_n={m_base_demo:.2f}$")
plt.grid(True)
plt.savefig("plots/time_response_fourier.png", dpi=300, bbox_inches="tight")
plt.close()

# Individual harmonic responses
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
    steady = t_n > plot_transient_cut

    plt.plot(
        t_n[steady], x_n[steady],
        color=colors[i],
        label=rf"$n={n}$ ($n\omega/\omega_n={(n * omega_base) / omega_n:.2f}$)"
    )

plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title(rf"Individual harmonic responses, $\omega_{{base}}/\omega_n={m_base_demo:.2f}$")
plt.grid(True)
plt.legend(fontsize=9)
plt.savefig("plots/time_response_fourier_harmonics.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# Generate Plots for All Base Frequencies (Optional)
# =============================================================================

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

        # Combined response plot
        plt.figure()
        plt.plot(t_loop[steady_loop], x_loop[steady_loop])
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.title(rf"Fourier forcing response, $\omega_{{base}}/\omega_n={m_base:.2f}$")
        plt.grid(True)
        plt.savefig(f"plots/time_response_fourier_m_{m_base:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Decomposed harmonic responses
        harmonics_plot = [n for n in sorted(set(list(a.keys()) + list(b.keys()))) if 1 <= n <= N_max]
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
                t_n[steady_n], x_n[steady_n],
                color=colors_plot[i],
                label=rf"$n={n}$ ($n\omega/\omega_n={(n * omega_base_loop) / omega_n:.2f}$)"
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.title(rf"Decomposed harmonic responses, $\omega_{{base}}/\omega_n={m_base:.2f}$")
        plt.grid(True)
        plt.legend(fontsize=9)
        plt.savefig(f"plots/time_response_fourier_harmonics_m_{m_base:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"\nSaved {len(m_values_plot)} time-series plots to plots/")

# Print harmonic frequency ratios
print("\nHarmonic frequency ratios:")
for n in sorted(set(list(a.keys()) + list(b.keys()))):
    print(f"  n={n}: n·omega_base/omega_n = {(n * omega_base) / omega_n:.3f}")


# =============================================================================
# Summary plot: maximum steady-state amplitude vs base forcing frequency
# =============================================================================

amp_max = []
for m_base in m_values_fine:
    omega_base_sweep = m_base * omega_n

    def F_fourier_sweep(t, omega_base_sweep=omega_base_sweep):
        val = a0
        for n, an in a.items():
            val += an * np.cos(n * omega_base_sweep * t)
        for n, bn in b.items():
            val += bn * np.sin(n * omega_base_sweep * t)
        return val

    t_s, x_s, _ = simulate_forcing(F_fourier_sweep, t_end=200.0, n_points=3000)
    steady = t_s > plot_transient_cut
    amp_max.append(np.max(np.abs(x_s[steady])))

plt.figure()
plt.plot(m_values_fine, amp_max)
plt.xlabel(r"$\omega_{base}/\omega_n$")
plt.ylabel("Max steady-state amplitude (m)")
plt.title("Maximum amplitude vs base forcing frequency")
plt.grid(True)
plt.savefig("plots/max_amplitude_vs_omega_base.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# Effect of Damping Ratio
# =============================================================================

zeta_values = [0.002, 0.005, 0.01, 0.02]

plt.figure()
for z in zeta_values:
    A_z = (F0 / m) / np.sqrt(
        (omega_n**2 - omegas**2)**2 + (2 * z * omega_n * omegas)**2
    )
    plt.plot(omegas / omega_n, A_z, label=rf"$\zeta = {z}$")

plt.axvline(1.0, color="k", linestyle="--", alpha=0.6)
plt.xlabel(r"Forcing frequency ratio $\omega/\omega_n$")
plt.ylabel("Steady-state amplitude (m)")
plt.title("Effect of damping ratio on frequency response")
plt.grid(True)
plt.legend()
plt.ylim(0, 6)
plt.savefig("plots/amplitude_vs_damping.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nDone.")
