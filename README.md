# Forced Damped Oscillator – ODE Project

This repository contains a numerical study of a **forced damped harmonic oscillator**, used as a simplified model for the vertical motion of a bridge subjected to periodic loading. The project investigates resonance phenomena, steady-state behavior, and the effect of damping using both analytical expressions and time-domain simulations.

The model is intentionally simplified and is intended for **educational and exploratory purposes**, rather than for design-level structural analysis.

---

## Model Description

The bridge is modeled as a **single-degree-of-freedom (SDoF) linear oscillator** governed by the equation

\[
m x''(t) + c x'(t) + k x(t) = F(t),
\]

where:
- \(x(t)\) is the vertical displacement,
- \(m\) is the effective (modal) mass,
- \(c\) is the viscous damping coefficient,
- \(k\) is the stiffness,
- \(F(t)\) is an external periodic forcing.

The system parameters are chosen so that the natural frequency matches a target value inspired by the Tacoma Narrows Bridge, while explicitly noting that the historical failure involved torsional flutter rather than simple resonance.

---

## Features

The script `1dof_clean.py` performs the following analyses:

### 1. Frequency Response (Analytical)
- Computes the steady-state amplitude as a function of forcing frequency.
- Plots the frequency response curve and highlights resonance.

### 2. Time-Domain Response to Harmonic Forcing
- Simulates the transient and steady-state response for several forcing frequency ratios:
  - below resonance,
  - near resonance,
  - above resonance.

### 3. Fourier-Series Forcing
- Models periodic loading using a truncated Fourier series with multiple harmonics.
- Demonstrates how higher harmonics can independently excite resonance.
- Produces:
  - combined response plots,
  - individual harmonic response plots.

### 4. Base Frequency Sweep
- Sweeps the base forcing frequency ratio \(\omega_{\text{base}}/\omega_n\).
- Computes and plots the maximum steady-state displacement amplitude.

### 5. Effect of Damping
- Investigates how different damping ratios affect the resonant amplitude.
- Generates a comparison plot for multiple values of \(\zeta\).

All plots are automatically saved in the `plots/` directory.

---

## Requirements

The project uses standard scientific Python libraries:

- Python 3.x
- NumPy
- SciPy
- Matplotlib

You can install the required packages with:

```bash
pip install numpy scipy matplotlib

---

To run it: 

```bash
python 1dof_clean.py



