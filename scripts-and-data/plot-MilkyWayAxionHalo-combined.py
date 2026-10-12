# Plot the Milky Way axion-induced magnetic field and spin dynamics for
# simulations with 1 and 1000 realizations in a single two-panel figure.
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

from src.utils import high_contrast_extended, linestyles


linewidth = 2
err_alpha = 0.5
plt.rc("font", size=10)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"
cm = 1 / 2.54

fig = plt.figure(figsize=(17 * cm, 7.5 * cm), dpi=300)
subfigs = fig.subfigures(1, 2, wspace=0.0)


def plot_panel(subfig, simulation_count, panel_letter, show_ylabels):
    data = np.load(f"scripts-and-data/Axion-Xe_NMR-simulations-{simulation_count}.npz")
    size = len(data["timeStamp_s"]) - 1
    time = data["timeStamp_s"][:size]
    B_vec_mean = data["B_vec_mean"][:size]
    B_vec_std = data["B_vec_std"][:size]
    Mxy_mean = data["Mxy_mrs"][:size]
    Mxy_std = data["Mxy_srs"][:size]

    subfig.subplots_adjust(
        left=0.193,
        bottom=0.15,
        right=0.955,
        top=0.912,
        wspace=0.1,
        hspace=0.264,
    )
    gs = subfig.add_gridspec(
        nrows=3,
        ncols=1,
        width_ratios=[1],
        height_ratios=[1, 1, 2.5],
    )
    Bx_ax = subfig.add_subplot(gs[0, 0])
    By_ax = subfig.add_subplot(gs[1, 0], sharey=Bx_ax)
    M_ax = subfig.add_subplot(gs[2, 0])

    xyz = ["x", "y", "z"]
    B_scale = 1e18  # convert T to aT
    for i, ax in enumerate([Bx_ax, By_ax]):
        if simulation_count == 1000:
            ax.fill_between(
                time,
                (B_vec_mean[:, i] - B_vec_std[:, i]) * B_scale,
                (B_vec_mean[:, i] + B_vec_std[:, i]) * B_scale,
                color=high_contrast_extended[-2],
                alpha=err_alpha,
            )
        ax.plot(
            time,
            B_vec_mean[:, i] * B_scale,
            label="$B_" + xyz[i] + "$ mean $\\pm 1 \\sigma$",
            color=high_contrast_extended[i],
            alpha=1,
            zorder=3,
        )

    M_start = 1
    if simulation_count == 1000:
        M_ax.fill_between(
            time[M_start:],
            Mxy_mean[M_start:] - Mxy_std[M_start:],
            Mxy_mean[M_start:] + Mxy_std[M_start:],
            color=high_contrast_extended[-2],
            alpha=err_alpha,
        )
    M_ax.plot(
        time[M_start:],
        Mxy_mean[M_start:],
        label="$ M_{xy} \\, (M_\\mathrm{eqb})$ mean $\\pm 1 \\sigma$",
        color=high_contrast_extended[5],
        linewidth=linewidth,
    )

    M_ax.set_xlabel("Time (s)")
    M_ax.set_yscale("log")
    Bx_ax.set_xticklabels([])
    By_ax.set_xticklabels([])
    Bx_ax.set_yticks([-30, 0, 30])
    M_ax.set_yticks([1, 1e1, 1e2, 1e3])

    xleft, xright = Bx_ax.get_xlim()
    By_ax.set_xlim(xleft, xright)
    M_ax.set_xlim(xleft, xright)
    Bx_ax.set_ylim(-35, 35)
    M_ax.set_ylim(0.8, 1.2e3)

    if show_ylabels:
        Bx_ax.set_ylabel("$B_x\\,(\\mathrm{a T})$")
        By_ax.set_ylabel("$B_y\\,(\\mathrm{a T})$")
        M_ax.set_ylabel("$M_{xy} \\, (M_\\mathrm{eqb})$")
    else:
        Bx_ax.set_ylabel("")
        By_ax.set_ylabel("")
        M_ax.set_ylabel("")

    # Match the panel-letter placement used in plot-SpinEcho_and_CW.py.
    Bx_ax.text(-0.15, 1.2, panel_letter, transform=Bx_ax.transAxes)


plot_panel(subfigs[0], 1, "(a)", show_ylabels=True)
plot_panel(subfigs[1], 1000, "(b)", show_ylabels=False)


# # put figure index
# letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
# for i, ax in enumerate([echo_pulse_ax, CW_pulse_ax]):
#     xleft, xright = ax.get_xlim()
#     ybottom, ytop = ax.get_ylim()
    
plt.savefig("tex/figures/MilkyWayAxionHalo-1-and-1000.pdf")
plt.show()
