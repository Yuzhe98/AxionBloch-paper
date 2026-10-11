# $env:PYTHONPATH = "C:\Users\zhenf\D\Yu0702\Axionbloch-paper;$env:PYTHONPATH”
# This file is for plotting the free decay of the spin system under a RF pulse.
from tkinter import font
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

from src.utils import high_contrast_extended
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

data_dir = "scripts-and-data/"

# Fit the measured I and Q channels together to an exponentially decaying
# rotating signal. The fitted I/Q amplitude sets the arbitrary plotting scale.
echo_file = Path(data_dir) / "CPMG_data_echo1.txt"
echo_metadata = {}
with echo_file.open(encoding="utf-8") as echo_handle:
    for line in echo_handle:
        if line.startswith("#") and "=" in line:
            key, value = line[1:].split("=", 1)
            echo_metadata[key.strip()] = value.strip()
echo_measurement = np.loadtxt(echo_file, comments="#")
echo_data_time_s = echo_measurement[:, 0]
echo_data_I = echo_measurement[:, 1]
echo_data_Q = echo_measurement[:, 2]
echo_center_s = float(echo_metadata["echo_center_s"])
echo_window_half_width_s = float(echo_metadata["window_half_width_s"])

echo_iq = echo_data_I + 1j * echo_data_Q
echo_edge_mask = np.abs(echo_data_time_s) > 0.75 * echo_window_half_width_s
echo_edge_I = echo_data_I[echo_edge_mask]
echo_edge_Q = echo_data_Q[echo_edge_mask]
echo_baseline_I_guess = float(np.median(echo_edge_I))
echo_baseline_Q_guess = float(np.median(echo_edge_Q))
echo_edge_noise = np.concatenate(
    (echo_edge_I - echo_baseline_I_guess, echo_edge_Q - echo_baseline_Q_guess)
)
echo_noise_scale = 1.4826 * np.median(np.abs(echo_edge_noise))
echo_iq_fft = np.fft.fft((echo_iq - np.mean(echo_iq)) * np.hanning(len(echo_iq)))
echo_iq_freqs = np.fft.fftfreq(
    len(echo_iq), d=float(np.median(np.diff(echo_data_time_s)))
)
echo_iq_freq_guess = float(echo_iq_freqs[np.argmax(np.abs(echo_iq_fft[1:])) + 1])


def _echo_iq_residual(parameters):
    amplitude = np.exp(parameters[0])
    T2star_s = np.exp(parameters[1])
    center_offset_s, frequency_hz, phase_rad, baseline_i, baseline_q = parameters[2:]
    envelope = amplitude * np.exp(
        -np.abs(echo_data_time_s - center_offset_s) / T2star_s
    )
    model = envelope * np.exp(
        1j * (2 * np.pi * frequency_hz * echo_data_time_s + phase_rad)
    ) + baseline_i + 1j * baseline_q
    residual = model - echo_iq
    return np.concatenate((residual.real, residual.imag))


echo_iq_fit = least_squares(
    _echo_iq_residual,
    x0=[
        np.log(max(np.max(np.abs(echo_iq)) - np.abs(echo_baseline_I_guess + 1j * echo_baseline_Q_guess), 1e-6)),
        np.log(0.004),
        0.0,
        echo_iq_freq_guess,
        0.0,
        echo_baseline_I_guess,
        echo_baseline_Q_guess,
    ],
    bounds=(
        [np.log(1e-6), np.log(1e-4), -0.002, -5000.0, -4 * np.pi, -1.0, -1.0],
        [np.log(1.0), np.log(0.1), 0.002, 5000.0, 4 * np.pi, 1.0, 1.0],
    ),
    loss="soft_l1",
    f_scale=max(echo_noise_scale, 1e-6),
)
echo_amplitude = float(np.exp(echo_iq_fit.x[0]))
echo_T2star = float(np.exp(echo_iq_fit.x[1]))
echo_center_offset_s = float(echo_iq_fit.x[2])
echo_center_fit_s = echo_center_s + echo_center_offset_s
echo_signal_freq_Hz = float(echo_iq_fit.x[3])
echo_phase_rad = float(echo_iq_fit.x[4])
echo_baseline_I = float(echo_iq_fit.x[5])
echo_baseline_Q = float(echo_iq_fit.x[6])

# Use the fitted effective T2* to set the inhomogeneous linewidth. T2=10 s
# follows the existing CPMG example; T2 is not identifiable from this short
# echo window alone.
echo_T2 = 10.0
echo_Tdelta = 1.0 / (1.0 / echo_T2star - 1.0 / echo_T2)
echo_RCF_Freq = 1.0 * unit.MHz
echo_t90_s = 2.0e-4
echo_rate = 100000.0 * unit.Hz
# Bring the first echo into the body of the panel; its center is 2*tau.
echo_simulated_center_s = 0.040
echo_tau_s = echo_simulated_center_s / 2.0
echo_duration_s = echo_simulated_center_s + echo_window_half_width_s + 2 * echo_t90_s

echo_sample = Sample(
    name="CPMG_echo1_comparison",
    gamma=gamma_p,
    massDensity=0.78945 * unit.g / unit.cm**3,
    molarMass=46.069 * unit.g / unit.mol,
    numOfSpinsPerMolecule=6 * unit.one,
    T2=echo_T2 * unit.s,
    T1=1e6 * unit.s,
    vol=1.0 * unit.cm**3,
    mu=mu_p,
    temp=300.0 * unit.K,
    verbose=False,
)
echo_FWHM = (1.0 / (np.pi * echo_Tdelta * unit.s) / echo_RCF_Freq) * unit.one
echo_magnet = Magnet(
    name="CPMG_echo1_comparison_magnet",
    B0=(echo_RCF_Freq - echo_signal_freq_Hz * unit.Hz)
    / (echo_sample.gamma / (2 * PI)),
    FWHM=echo_FWHM,
    nFWHM=20.0,
)
echo_magnet.setHomogeneity(numPt=500)
echo_simu = Simulation(
    name="Single_Hahn_echo_fitted_to_CPMG_data",
    sample=echo_sample,
    magnet=echo_magnet,
    excField=MagField(name="single_Hahn_echo"),
    rate=echo_rate,
    RCF_freq=echo_RCF_Freq,
    duration=echo_duration_s * unit.s,
    verbose=False,
)
echo_t90 = echo_t90_s * unit.s
echo_simu.excField.setCPMGPulseTrain(
    timeStep=echo_simu.timeStep,
    timeLen=echo_simu.timeLen,
    gamma=echo_simu.sample.gamma,
    t90=echo_t90,
    tau=echo_tau_s * unit.s,
    numEcho=1,
    nu_rot=echo_signal_freq_Hz * unit.Hz,
)
echo_simu.generateTrajectories(integrator="RK4")
echo_timeStamp_s = echo_simu.timeStep.to_value(unit.s) * np.arange(
    echo_simu.trjry.shape[1]
)
echo_B_vec = echo_simu.excField.B_vec[0].to_value(unit.T)
echo_B_timeStamp_s = echo_simu.timeStep.to_value(unit.s) * np.arange(
    echo_B_vec.shape[0]
)
echo_M = echo_simu.trjry[0]
echo_data_time_absolute_s = (
    echo_simulated_center_s + echo_data_time_s - echo_center_offset_s
)
echo_data_I_scaled = (echo_data_I - echo_baseline_I) / echo_amplitude
echo_data_Q_scaled = (echo_data_Q - echo_baseline_Q) / echo_amplitude
echo_sim_complex = echo_M[:, 0] + 1j * echo_M[:, 1]
echo_sim_at_data = np.interp(
    echo_data_time_absolute_s, echo_timeStamp_s, echo_sim_complex.real
) + 1j * np.interp(
    echo_data_time_absolute_s, echo_timeStamp_s, echo_sim_complex.imag
)
echo_sim_scale_mask = np.abs(echo_data_time_s - echo_center_offset_s) <= echo_window_half_width_s
echo_sim_scale = np.vdot(
    echo_sim_at_data[echo_sim_scale_mask],
    (echo_data_I_scaled + 1j * echo_data_Q_scaled)[echo_sim_scale_mask],
) / np.vdot(echo_sim_at_data[echo_sim_scale_mask], echo_sim_at_data[echo_sim_scale_mask])
echo_sim_complex_scaled = echo_sim_complex * echo_sim_scale

print(
    f"Echo fit: T2*={echo_T2star * 1e3:.3f} ms, "
    f"amplitude={echo_amplitude:.6g} Phi_0, "
    f"I/Q baselines=({echo_baseline_I:.3g}, {echo_baseline_Q:.3g}) Phi_0, "
    f"center offset={echo_center_offset_s * 1e3:+.3f} ms, "
    f"I/Q frequency={echo_signal_freq_Hz:.2f} Hz, tau={echo_tau_s * 1e3:.1f} ms"
)


linewidth = 1

plt.rc("font", size=10)  # font size for all figures
plt.rcParams["font.family"] = "Times New Roman"

# Make math text match Times New Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"
cm = 1 / 2.54  # convert cm to inch
fig = plt.figure(
    figsize=(8.5 * cm, 9 * cm), dpi=300
)  # initialize a figure following APS journal requirements
# #############################################################################
# to specify heights and widths of subfigures
width_ratios = [1]
height_ratios = [0.5, 1]
gs = gridspec.GridSpec(
    nrows=2, ncols=1, width_ratios=width_ratios, height_ratios=height_ratios
)  # create grid for multiple figures
# #############################################################################
# fix the margins
left = 0.16
bottom = 0.12
right = 0.97
top = 0.96
wspace = 0.0
hspace = 0.12
fig.subplots_adjust(
    left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
)
# #############################################################################
echo_pulse_ax = fig.add_subplot(gs[0, 0])
echo_M_ax = fig.add_subplot(gs[1, 0])

echo_pulse_ax.plot(
    echo_B_timeStamp_s,
    echo_B_vec[:, 0] * 1e9,
    label="$B_x$",
    color=high_contrast_extended[0],
    linewidth=linewidth,
)
echo_pulse_ax.text(0.015, 0.78, r"$\pi/2$", transform=echo_pulse_ax.transAxes)
echo_pulse_ax.text(0.50, 0.78, r"$\pi$", transform=echo_pulse_ax.transAxes)
echo_pulse_ax.plot(
    echo_B_timeStamp_s,
    echo_B_vec[:, 1] * 1e9,
    label="$B_y$",
    color=high_contrast_extended[1],
    linewidth=linewidth,
)

echo_M_ax.plot(
    echo_timeStamp_s,
    echo_sim_complex_scaled.real,
    label="$I$ (simulation)",
    color=high_contrast_extended[4],
    linewidth=linewidth,
)
echo_M_ax.plot(
    echo_timeStamp_s,
    echo_sim_complex_scaled.imag,
    label="$Q$ (simulation)",
    color=high_contrast_extended[5],
    linewidth=linewidth,
)
echo_M_ax.scatter(
    echo_data_time_absolute_s,
    echo_data_I_scaled,
    label="$I$ (measurement)",
    color=high_contrast_extended[0],
    s=5,
    linewidths=0,
    zorder=3,
)
echo_M_ax.scatter(
    echo_data_time_absolute_s,
    echo_data_Q_scaled,
    label="$Q$ (measurement)",
    color=high_contrast_extended[1],
    s=5,
    linewidths=0,
    zorder=3,
)
echo_M_ax.legend(
    loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2,
    frameon=False, fontsize=6.5,
)

# Axis labels and limits are kept together here for easy figure-wide editing.
echo_xlim = (-0.005, echo_simulated_center_s + echo_window_half_width_s)
axis_settings = (
    (echo_pulse_ax, "", "$B$ (arb. units)", echo_xlim, None),
    (echo_M_ax, "Time (s)", "$I, Q$ (arb. units)", echo_xlim, (-1.5, 1.5)),
)
for ax, xlabel, ylabel, xlim, ylim in axis_settings:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

# hide x-axis tick labels for the upper plot
echo_pulse_ax.set_xticklabels([])

# set legends for the subplots
echo_pulse_ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)


# put figure index
letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
for i, ax in enumerate([echo_pulse_ax]):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.text(-0.25, 1.2, letters[i], transform=ax.transAxes)
fig.align_ylabels([echo_pulse_ax, echo_M_ax])
# ha = 'left' or 'right'
# va = 'top' or 'bottom'
# plt.tight_layout()
plt.savefig("tex/figures/SpinEcho_real_data.pdf")
plt.show()
