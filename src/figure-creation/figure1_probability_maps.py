"""
A figure to visualise the probabililty maps constructed from MESSENGER region
observations.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hermpy.utils import Constants
from matplotlib.collections import LineCollection

PROBABILITIY_MAPS_FILE = (
    Path(__file__).parent.parent.parent / "resources/region_probability_maps.nc"
)

FIG_OUTPUT = (
    Path(__file__).parent.parent.parent / "figures/figure1_probability_maps.pdf"
)


def main():

    # Insure output dir exists, if not, create it
    if not os.path.isdir(FIG_OUTPUT.parent):
        os.makedirs(FIG_OUTPUT.parent)

    # Load probabillity maps
    probability_maps = xr.load_dataset(PROBABILITIY_MAPS_FILE)

    # Create figure
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 4))

    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere"]
    for i, ax in enumerate(axes):

        map_data = probability_maps[regions[i]]

        mesh = ax.pcolormesh(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            map_data.values.T,
            vmin=0,
            vmax=1,
            cmap="magma",
        )

        ax.set_title(regions[i])
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 8)

        ax.set_facecolor("lightgrey")

        ax.set_xlabel(r"$X_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

        if i == 0:
            ax.set_ylabel(
                r"$\left(Y_{\rm MSM'}^2 + Z_{\rm MSM'}^2 \right)^{0.5}\quad \left[ R_{\rm M} \right]$"
            )

        # Add colorbar
        if i == 1:
            cbar_bounds = [-1.2, -0.5, 3.4, 0.1]
            cbar_ax = ax.inset_axes(cbar_bounds)

            plt.colorbar(
                mesh, cax=cbar_ax, location="bottom", label="Region Probability"
            )

        # Add Mercury
        mercury_params = {
            "segments": 40,
            "linewidth": 1,
        }
        draw_alternating_circle(
            (0, Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS),
            1,
            ax,
            **mercury_params,
        )
        draw_alternating_circle(
            (0, -(Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS)),
            1,
            ax,
            **mercury_params,
        )

    fig.savefig(FIG_OUTPUT, format="pdf")


def draw_alternating_circle(center, radius, ax, segments=200, linewidth=3):

    x0, y0 = center

    # circle points
    theta = np.linspace(0, 2 * np.pi, segments)
    x = x0 + radius * np.cos(theta)
    y = y0 + radius * np.sin(theta)

    # list of line segments
    points = np.column_stack((x, y))
    segments_list = np.stack([points[:-1], points[1:]], axis=1)

    # alternating colors
    colors = ["black" if i % 2 == 0 else "white" for i in range(len(segments_list))]

    # create LineCollection
    lc = LineCollection(segments_list, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)


if __name__ == "__main__":
    main()
