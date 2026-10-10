"""Compare a Bloch simulation with the measured T1 inversion-recovery data.

Run from the repository root with the project environment active::

    python scripts-and-data/compare-T1-inversion-recovery.py

The complex measurement is phase-rotated and linearly normalized to the
simulated signed readout. The fitted complex scale and offset account for
receiver phase, gain, and a constant baseline without removing the inversion
recovery sign change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_FILE = ROOT / "scripts-and-data" / "T1-data" / "T1_100%"
FIGURE_FILE = ROOT / "tex" / "figures" / "T1_100pct_simulation_comparison.pdf"
RESULTS_FILE = ROOT / "scripts-and-data" / "T1-data" / "T1_100pct_simulation_comparison.csv"

# Values transcribed from the measurement-file header.
RCF_FREQ = 7.81 * unit.MHz
T1 = 72e-3 * unit.s
T2 = 72e-3 * unit.s  # T2 is not reported; use T2=T1 for this comparison.
P90_DURATION_S = 18e-6
P180_DURATION_S = 36e-6
RATE = 500e3 * unit.Hz  # gives integer 9- and 18-step hard pulses
FWHM = 1.0 * ppm
N_FWHM = 0.0


def read_measurement(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return inversion times and complex integrals from the instrument file."""
    values = np.loadtxt(path, comments="%", encoding="latin1")
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Expected time, real, imag columns in {path}")
    wait_s = values[:, 0]
    signal = values[:, 1] + 1j * values[:, 2]
    return wait_s, signal


def simulate_signed_readout(wait_s: float) -> float:
    """Simulate pi - wait - pi/2 and return signed My after readout."""
    sample = Sample(
        name="T1_100pct_measurement_comparison",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=T1,
        T2=T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300.0 * unit.K,
        verbose=False,
    )
    B0 = RCF_FREQ / (sample.gamma / (2 * PI))
    magnet = Magnet(
        name="T1_100pct_measurement_comparison_magnet",
        B0=B0,
        FWHM=FWHM,
        nFWHM=N_FWHM,
    )

    dt_s = (1.0 / RATE).to_value(unit.s)
    p90_steps = int(round(P90_DURATION_S / dt_s))
    p180_steps = int(round(P180_DURATION_S / dt_s))
    wait_steps = int(round(wait_s / dt_s))
    readout_start = p180_steps + wait_steps
    readout_index = readout_start + p90_steps

    # Include one point after the readout pulse so readout_index is stored.
    duration = (readout_index + 1) / RATE
    simu = Simulation(
        name=f"T1_wait_{wait_s:.6g}s",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="measured_pi_pi_over_2_sequence"),
        RCF_freq=RCF_FREQ,
        rate=RATE,
        duration=duration,
        verbose=False,
    )

    b90 = PI / (simu.sample.gamma * (p90_steps * simu.timeStep))
    B_vec = np.zeros((1, simu.timeLen - 1, 3)) * unit.T
    # The measured pi pulse is twice as long as the pi/2 pulse, so both use
    # the same B1 amplitude to produce flip angles differing by a factor two.
    B_vec[0, :p180_steps, 0] = 0.5 * b90
    B_vec[0, readout_start:readout_index, 0] = 0.5 * b90
    simu.excField.B_vec = B_vec
    simu.generateTrajectories(integrator="RK4")
    return float(simu.trjry[0, readout_index, 1])


def fit_complex_gain_and_offset(
    simulated: np.ndarray, measured: np.ndarray
) -> tuple[complex, complex, np.ndarray]:
    """Fit measured ~= offset + complex_gain * simulated by linear least squares."""
    design = np.column_stack((np.ones(simulated.size), simulated))
    offset, gain = np.linalg.lstsq(design, measured, rcond=None)[0]
    fitted = offset + gain * simulated
    return complex(offset), complex(gain), fitted


def main() -> None:
    wait_s, measured = read_measurement(MEASUREMENT_FILE)
    simulated = np.array([simulate_signed_readout(t) for t in wait_s])

    # One complex affine calibration aligns receiver phase, gain, and constant
    # baseline while retaining a signed, dimensionless inversion-recovery curve.
    offset, gain, fitted_complex = fit_complex_gain_and_offset(simulated, measured)
    measured_normalized = (measured - offset) / gain
    in_phase_residual = measured_normalized.real - simulated
    quadrature_residual = measured_normalized.imag
    complex_residual_rms = float(
        np.sqrt(np.mean(np.abs(measured_normalized - simulated) ** 2))
    )
    relative_rms = complex_residual_rms / float(np.sqrt(np.mean(simulated**2)))

    print(f"Measurement points: {len(wait_s)}")
    print(f"Simulation T1: {T1.to_value(unit.ms):.1f} ms")
    print(f"Fitted receiver phase: {np.degrees(np.angle(gain)):.2f} deg")
    print(f"Fitted complex gain magnitude: {abs(gain):.6g} integral units per M/M_eqb")
    print(f"Fitted complex baseline: {offset.real:.6g} + {offset.imag:.6g}j")
    print(f"Normalized complex residual RMS: {complex_residual_rms:.5f} M_eqb")
    print(f"Residual RMS / simulated RMS: {100 * relative_rms:.2f}%")
    print("Note: T2 is assumed equal to T1 because the measurement header does not report it.")
    print("Note: O1=-5800 Hz is not applied as a detuning; its instrument reference is ambiguous.")

    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        RESULTS_FILE,
        np.column_stack(
            (
                wait_s,
                measured.real,
                measured.imag,
                simulated,
                measured_normalized.real,
                in_phase_residual,
                quadrature_residual,
            )
        ),
        delimiter=",",
        header=(
            "wait_s,measured_real,measured_imag,simulated_signed_My,"
            "measured_normalized_in_phase,residual_in_phase,residual_quadrature"
        ),
        comments="",
    )

    plt.rc("font", size=9)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, (signal_ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(8.5 / 2.54, 10 / 2.54),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(left=0.2, right=0.96, top=0.97, bottom=0.13, hspace=0.08)

    signal_ax.plot(wait_s, simulated, color="#1f77b4", linewidth=1.0, label="Simulation")
    signal_ax.scatter(
        wait_s,
        measured_normalized.real,
        color="#d62728",
        s=12,
        label="Measurement (phase-corrected, normalized)",
        zorder=3,
    )
    signal_ax.set_ylabel(r"Signed readout ($M/M_{\rm eqb}$)")
    signal_ax.legend(frameon=False, loc="lower right")

    residual_ax.axhline(0, color="0.35", linewidth=0.8)
    residual_ax.plot(
        wait_s,
        in_phase_residual,
        color="#d62728",
        marker="o",
        markersize=2.5,
        linewidth=0.8,
        label="In-phase residual",
    )
    residual_ax.plot(
        wait_s,
        quadrature_residual,
        color="#9467bd",
        marker="s",
        markersize=2.2,
        linewidth=0.8,
        label="Quadrature residual",
    )
    residual_ax.set_xlabel(r"Inversion time $\tau$ (s)")
    residual_ax.set_ylabel("Residual\n($M/M_{\\rm eqb}$)")
    residual_ax.legend(frameon=False, loc="best", fontsize=7)
    fig.align_ylabels((signal_ax, residual_ax))
    fig.savefig(FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {FIGURE_FILE}")
    print(f"Saved normalized comparison data: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
