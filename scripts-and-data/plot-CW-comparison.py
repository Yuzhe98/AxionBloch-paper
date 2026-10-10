"""Compare saved CW simulation quadratures with the envelope-carrier model.

Run from the repository root with:
    python scripts-and-data/plot-CW-comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).with_name("CW_thermally_polarized_simu.npz")
OUTPUT_PATH = Path(__file__).with_name("CW_simulation_theory_comparison.png")


def main():
    data = np.load(DATA_PATH)
    time = data["timeStamp_s"]
    # The trajectory includes the final sample, while B_vec has one fewer.
    n = min(len(time), data["trjry"].shape[1])
    time = time[:n]
    simulation = data["trjry"][0, :n, :2]

    t2 = float(data["T2_s"])
    tdelta = float(data["Tdelta_s"])
    t2star = 1.0 / (1.0 / t2 + 1.0 / tdelta)
    frequency = float(data["signalFreqRot_Hz"])
    phase0 = float(data["init_phase_rad"])
    gamma = float(data["gamma_rad_s_T"])
    b1 = float(data["B1_rot_T"])

    # Match the CW theory expression in plot-SpinEcho_and_CW.py exactly.
    envelope = (
        (1.0 - np.exp(-time / t2star))
        * np.sin(gamma * b1 * t2star)
        * (1.0 + 1e-5)
    )
    phase = 2.0 * np.pi * frequency * time + phase0 + np.pi / 2.0
    theory = np.column_stack((envelope * np.cos(phase), envelope * np.sin(phase)))

    # One joint least-squares gain tests whether the discrepancy is just a
    # constant multiplicative scale, without dividing near zero crossings.
    gain = np.sum(simulation * theory) / np.sum(theory * theory)
    residual = simulation - theory
    scaled_residual = simulation - gain * theory
    relative_rms = np.linalg.norm(residual) / np.linalg.norm(simulation)
    scaled_relative_rms = np.linalg.norm(scaled_residual) / np.linalg.norm(simulation)

    # CW steady-state response is set by T2. Compare that envelope with the
    # T2* version used in the main plot, which includes inhomogeneous decay.
    t2_envelope = (
        (1.0 - np.exp(-time / t2))
        * np.sin(gamma * b1 * t2)
        * (1.0 + 1e-5)
    )
    t2_theory = np.column_stack(
        (t2_envelope * np.cos(phase), t2_envelope * np.sin(phase))
    )
    t2_residual = simulation - t2_theory

    # Fit a general quadrature rotation and gain. A good fit here with a poor
    # scalar fit indicates phase convention mismatch; a poor fit indicates
    # additional frequency or envelope mismatch.
    coeff, *_ = np.linalg.lstsq(theory, simulation, rcond=None)
    fitted = theory @ coeff
    fitted_relative_rms = np.linalg.norm(simulation - fitted) / np.linalg.norm(simulation)
    fitted_gain = np.sqrt(np.linalg.det(coeff)) if np.linalg.det(coeff) > 0 else np.nan
    fitted_phase_deg = np.degrees(np.arctan2(coeff[0, 1] - coeff[1, 0], coeff[0, 0] + coeff[1, 1]))

    print(f"Data: {DATA_PATH}")
    print(f"Samples: {n}; duration: {time[-1]:.6g} s")
    print(f"T2* = {t2star:.8g} s; drive = {frequency:.8g} Hz")
    print(f"gamma B1 T2* = {gamma * b1 * t2star:.8g}")
    print(f"Joint best-fit scalar simulation/theory gain = {gain:.8g}")
    print(f"Raw residual RMS / simulation RMS = {relative_rms:.3%}")
    print(
        "Using T2 instead of T2* gives residual RMS "
        f"{np.sqrt(np.mean(t2_residual**2)):.8g} M_eqb "
        f"(max abs {np.max(np.abs(t2_residual)):.8g} M_eqb)."
    )
    print(f"After scalar gain correction = {scaled_relative_rms:.3%}")
    print(f"After quadrature rotation/gain fit = {fitted_relative_rms:.3%}")
    print(f"Approx. fitted quadrature phase shift = {fitted_phase_deg:.5g} deg")
    print(f"Approx. fitted quadrature gain = {fitted_gain:.8g}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, simulation[:, 0], color="tab:red", label=r"$M_x$ simulation")
    axes[0].plot(time, simulation[:, 1], color="tab:green", label=r"$M_y$ simulation")
    axes[0].plot(time, theory[:, 0], "--", color="tab:blue", label=r"$M_x$ theory")
    axes[0].plot(time, theory[:, 1], "--", color="tab:purple", label=r"$M_y$ theory")
    axes[0].set_ylabel(r"$M/M_{eqb}$")
    axes[0].legend(ncol=2, frameon=False)

    axes[1].plot(time, residual[:, 0], color="tab:orange", label=r"$M_x^{sim}-M_x^{theo}$")
    axes[1].plot(time, residual[:, 1], color="tab:brown", label=r"$M_y^{sim}-M_y^{theo}$")
    axes[1].set_ylabel("Raw residual")
    axes[1].legend(frameon=False)

    axes[2].plot(time, scaled_residual[:, 0], color="tab:orange", label=r"$M_x$ after scalar fit")
    axes[2].plot(time, scaled_residual[:, 1], color="tab:brown", label=r"$M_y$ after scalar fit")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Residual after gain fit")
    axes[2].legend(frameon=False)

    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved comparison figure: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
