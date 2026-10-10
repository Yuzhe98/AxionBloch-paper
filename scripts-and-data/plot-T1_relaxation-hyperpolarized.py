# $env:PYTHONPATH = "C:\Users\zhenf\D\Yu0702\Axionbloch-paper;$env:PYTHONPATH”
# This file is for plotting the free decay of the spin system under a RF pulse.
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

from src.utils import high_contrast_extended
data_dir = "scripts-and-data/"
HP_data = np.load(data_dir + "T1_relaxation-hyperpolarized.npz")
HP_Tdelta = HP_data["Tdelta_s"]
HP_T2 = HP_data["T2_s"]
HP_T2star = (HP_T2 ** (-1) + HP_Tdelta ** (-1)) ** (-1)
HP_T1 = HP_data["T_1_s"]

HP_size = min(
    len(HP_data["timeStamp_s"]),
    HP_data["B_vec"].shape[1],
    HP_data["trjry"].shape[1],
)
HP_plot_stride = max(1, HP_size // 20000)
HP_plot_indices = np.arange(0, HP_size, HP_plot_stride)
HP_timeStamp_s = HP_data["timeStamp_s"][HP_plot_indices]
HP_B_vec = HP_data["B_vec"][0, HP_plot_indices]
HP_M = HP_data["trjry"][0, HP_plot_indices]
HP_init_M = float(HP_M[0, 2])

linewidth = 1

plt.rc("font", size=10)  # font size for all figures
plt.rcParams["font.family"] = "Times New Roman"

# Make math text match Times New Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"
cm = 1 / 2.56  # convert cm to inch
fig = plt.figure(
    figsize=(8.5 * cm, 10 * cm), dpi=300
)  # initialize a figure following APS journal requirements
# #############################################################################
# to specify heights and widths of subfigures
width_ratios = [1]
height_ratios = [0.5, 1, 0.5]
gs = gridspec.GridSpec(
    nrows=3, ncols=1, width_ratios=width_ratios, height_ratios=height_ratios
)  # create grid for multiple figures
# #############################################################################
# fix the margins
left = 0.19
bottom = 0.16
right = 0.95
top = 0.96
wspace = 0.2
hspace = 0.14
fig.subplots_adjust(
    left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
)
# #############################################################################
# HP_pulse_ax = fig.add_subplot(gs[0, 0])
HP_B_ax = fig.add_subplot(gs[0, 0])
HP_M_ax = fig.add_subplot(gs[1, 0], sharex=HP_B_ax)
HP_residual_ax = fig.add_subplot(gs[2, 0], sharex=HP_B_ax)

HP_B_ax.plot(
    HP_timeStamp_s,
    HP_B_vec[:, 0],
    label="$B_x$",
    color=high_contrast_extended[0],
    linewidth=linewidth,
)
HP_B_ax.plot(
    HP_timeStamp_s,
    HP_B_vec[:, 1],
    label="$B_y$",
    color=high_contrast_extended[1],
    linestyle="--",
    linewidth=linewidth,
)

# HP_pulse_ax.plot(
#     HP_timeStamp_s,
#     HP_B_vec[:, 0] * 1e12,
#     label="$B_{x}$",
#     color=high_contrast_extended[0],
#     linewidth = linewidth ,
# )
# HP_pulse_ax.plot(
#     HP_timeStamp_s,
#     HP_B_vec[:, 1] * 1e12,
#     label="$B_{y}$",
#     color=high_contrast_extended[1],
#     linewidth = linewidth ,
# )
# HP_pulse_ax.set_xlabel("")
# HP_pulse_ax.set_ylabel("$B\\,(\\mathrm{p T})$")
# # pulse_ax.legend(loc="upper right", ncol=2, frameon=False)

HP_M_ax.plot(
    HP_timeStamp_s,
    HP_M[:, 2],
    label="$M_z \\, (M_\\mathrm{eqb})$",
    color=high_contrast_extended[-1],
    linewidth=linewidth,
)
# HP_M_ax.plot(
#     HP_timeStamp_s,
#     HP_M[:, 1],
#     label="$M_{y}$",
#     color=high_contrast_extended[3],
#     linewidth = linewidth ,
# )
# M_ax.plot(
#     timeStamp_s,
#     (M[:, 0] ** 2 + M[:, 1] ** 2) ** 0.5,
#     label="$(M_{x}^2 + M_{y}^2)^{1/2}$",
#     linestyle="dashed",
#     color=high_contrast_extended[5],
# )
gamma = 2 * np.pi * 42.57747892e6  # gyromagnetic ratio of proton in Hz/T
B1_T = 0.5 * 1e-11  # magnetic field strength in Tesla
decay_envelope = HP_init_M *  np.exp(-HP_timeStamp_s / HP_T1)
HP_M_ax.plot(
    HP_timeStamp_s[: len(HP_timeStamp_s) // 1],
    decay_envelope[: len(HP_timeStamp_s) // 1],
    label="$e^{-t/T_1}$",
    linestyle="dotted",
    color=high_contrast_extended[-6],
    linewidth=linewidth,
)
HP_M_ax.axhline(
    y=1, color=high_contrast_extended[-2], linestyle="dashed", linewidth=linewidth
)
HP_M_theory = 1 + (HP_init_M - 1) * np.exp(-HP_timeStamp_s / HP_T1)
HP_residual_ax.plot(
    HP_timeStamp_s,
    HP_M[:, 2] - HP_M_theory,
    label="$M_z^{\\mathrm{sim}}$",
    color=high_contrast_extended[3],
    linewidth=linewidth,
)
# HP_M_ax.axvline(
#     x=HP_T1, color=high_contrast_extended[-3], linestyle="dashed", linewidth=linewidth
# )
# HP_M_ax.text(
#     3.5,
#     0.0016,
#     "$e^{\\frac{-t}{T_1}}$",
#     transform=HP_M_ax.transAxes,
#     # fontsize=8,
#     color="#393b79",
#     ha="left",
# )

# Keep axis labels and limits together for easy figure-wide adjustment.
axis_settings = (
    (HP_B_ax, "", "$B\\,(\\mathrm{T})$", None, None),
    (HP_M_ax, "", "$M \\, (M_\\mathrm{eqb})$", None, (0.1, 4e5)),
    (HP_residual_ax, "Time (s)", "$M_z^{\\mathrm{sim}} - M_z^{\\mathrm{theory}}$", None, None),
)
for ax, xlabel, ylabel, xlim, ylim in axis_settings:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

HP_M_ax.set_yscale("log")
HP_B_ax.tick_params(labelbottom=False)
HP_M_ax.tick_params(labelbottom=False)

HP_M_ax.set_yticks([1, 1e2, 1e4])
HP_B_ax.legend(loc="upper right", frameon=False, ncol=2)
# HP_pulse_ax.legend(
#     loc="upper left",
#     bbox_to_anchor=(1.0, 1.0),
#     frameon=False,
# )
HP_M_ax.legend(
    loc="upper right",
    # bbox_to_anchor=(1.0, 1.0),
    frameon=False,
)
HP_residual_ax.legend(loc="upper right", frameon=False)
fig.align_ylabels([HP_B_ax, HP_M_ax, HP_residual_ax])
# M_ax.legend(loc="upper right", ncol=2, frameon=False)

# # put figure index
# letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
# for i, ax in enumerate([HP_pulse_ax]):
#     xleft, xright = ax.get_xlim()
#     ybottom, ytop = ax.get_ylim()
#     ax.text(-0.4, 1.2, letters[1], transform=ax.transAxes)
# # ha = 'left' or 'right'
# # va = 'top' or 'bottom'
# # plt.tight_layout()
plt.savefig("tex/figures/T1_relaxation-hyperpolarized.pdf")
plt.show()
