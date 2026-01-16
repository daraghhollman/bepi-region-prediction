"""
A figure to visualise the probabililty maps constructed from MESSENGER region
observations.
"""

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hermpy.utils import Constants
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 4))

    # Make residence plot
    residence_data = probability_maps["Minutes In Bin"] / 60  # Convert to hours
    residence_mesh = axes[0].pcolormesh(
        residence_data.coords["X MSM'"],
        residence_data.coords["CYL MSM'"],
        residence_data.values.T,
        cmap="viridis",
        norm="log",
    )

    residence_mask = residence_data.values.T != 0

    # Residience plot config
    cbar_bounds = [0, -0.6, 1, 0.1]
    cbar_ax = axes[0].inset_axes(cbar_bounds)

    axes[0].set_title("MESSENGER\nResidence")

    plt.colorbar(
        residence_mesh, cax=cbar_ax, location="bottom", label="Time Spent [hours]"
    )
    axes[0].set_ylabel(
        r"$\left(Y_{\rm MSM'}^2 + Z_{\rm MSM'}^2 \right)^{0.5}\quad \left[ R_{\rm M} \right]$"
    )

    # Regions plots config
    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere"]
    for i, ax in enumerate(axes[1:]):
        map_data = probability_maps[regions[i]]

        # We also don't want to show 0 counts for areas where the spacecraft never
        # observed that region as this obsured areas with near-zero counts.
        map_data.values[np.where(map_data.values == 0)] = np.nan

        mesh = ax.pcolormesh(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            map_data.values.T,
            vmax=1,
            cmap="magma",
        )

        # Add mask outline
        ax.contour(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            residence_mask,
            levels=[0.5],
            antialiased=False,
            colors="grey",
            zorder=-1,
        )
        ax.contourf(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            residence_mask,
            levels=[0, 0.5, 1],
            colors=["white", "lightgrey"],
            zorder=-2,
        )

        ax.set_title(regions[i])

        # Add colorbar
        if i == 1:
            cbar_bounds = [-1.2, -0.6, 3.4, 0.1]
            cbar_ax = ax.inset_axes(cbar_bounds)

            plt.colorbar(
                mesh, cax=cbar_ax, location="bottom", label="Relative Region Occurence"
            )

    # Config to apply to all axes
    for i, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 8)

        ax.set_xlabel(r"$X_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

        # Add Mercury
        mercury_params = {
            "segments": 30,
            "linewidth": 1.5,
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

        # Add panel label
        labels = ["a", "b", "c", "d"]
        ax.text(0.02, 0.9, f"({labels[i]})", transform=ax.transAxes)

    # Add custom legend
    custom_handles = [
        CurvedLegendHandle(angle=180),
        Rectangle((0, 0), 1, 1, facecolor="lightgrey", edgecolor="grey"),
    ]

    fig.legend(
        custom_handles,
        ["Mercury (northern and southern extent)", "MESSENGER Residence Bounds"],
        loc="upper center",
        ncol=1,
        frameon=False,
        handler_map={CurvedLegendHandle: custom_handles[0]},
        bbox_to_anchor=(0.6, 0.95),
    )

    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.2)
    fig.savefig(FIG_OUTPUT, format="pdf")


def draw_alternating_circle(center, radius, ax, segments=200, linewidth=3, **kwargs):
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
    lc = LineCollection(segments_list, colors=colors, linewidths=linewidth, **kwargs)
    ax.add_collection(lc)


class CurvedLegendHandle:
    def __init__(self, angle=180):
        self.angle = angle

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        w, h = handlebox.width, handlebox.height
        x, y = handlebox.xdescent, handlebox.ydescent

        # An arc centered inside the legend box
        arc = mpatches.Arc(
            (x + w / 2, y + h / 2),
            w,
            h * 2,
            angle=0,
            theta1=0,
            theta2=180,
            lw=2,
            ls=(0, (2, 2)),
            color="black",
            transform=handlebox.get_transform(),
        )

        handlebox.add_artist(arc)
        return arc


if __name__ == "__main__":
    main()
