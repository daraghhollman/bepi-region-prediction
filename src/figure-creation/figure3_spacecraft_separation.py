import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.table import QTable
from astropy.time import Time
from hermpy.data import rotate_to_aberrated_coordinates
from hermpy.net import ClientSPICE
from hermpy.plotting import plot_magnetospheric_boundaries
from hermpy.utils import Constants
from matplotlib.dates import DateFormatter, MinuteLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Patch
from planetary_coverage import MetaKernel
from sunpy.time import TimeRange

# A 7-colour colourblind friendly palette. Wong et al. Nature Methods.
wong_colours = [
    "#F0E442",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from get_probabilities import get_probability_at_position, load_probability_maps

START_TIME = dt.datetime(2027, 5, 2, 4)
END_TIME = dt.datetime(2027, 5, 2, 14)

TIME_RESOLUTION = 1000  # Number of spice measurements in that time.

PROBABILITY_MAP = load_probability_maps(
    Path(__file__).parent.parent.parent / "resources/region_probability_maps.nc"
)

FIG_OUTPUT = Path(__file__).parent.parent.parent / "figures/figure3.pdf"

LINE_PARAMS = {
    "lw": 2,
    "path_effects": [  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2.5, foreground="black"),
        matplotlib.patheffects.Normal(),
    ],
}


def main():
    # Insure directory exists
    if not os.path.isdir(FIG_OUTPUT.parent):
        os.makedirs(FIG_OUTPUT.parent)

    # Load SPICE data for these times. With the setup script, we downloaded spice
    # kernels which we must load.
    spice_client = ClientSPICE()

    # Planetary coverage has some nice code to handle updating path values for
    # spice metakernels at runtime.
    # We need the planning kernel for times in the future.
    bepi_metakernel = MetaKernel(
        Path(__file__).parent.parent.parent
        / "resources/bepicolombo-spice/kernels/mk/bc_plan.tm",
        kernels=Path(__file__).parent.parent.parent
        / "resources/bepicolombo-spice/kernels/",
    )

    spice_client.add_local_kernels(bepi_metakernel.kernels)

    with spice_client.KernelPool():
        # Get trajectory
        time_range = TimeRange(START_TIME, END_TIME)
        query_times = [
            t.center.to_datetime() for t in time_range.split(TIME_RESOLUTION)
        ]

        ets = spice.datetime2et(query_times)

        spacecraft = ["MPO", "MMO"]
        spacecraft_data: dict[str, dict[str, Any]] = {}

        for id in spacecraft:
            positions, _ = spice.spkpos(id, ets, "BC_MSM", "NONE", "Mercury")

            # We need to aberrate these positions. The
            # quickest way would be to convert to an Astropy
            # QTable and use hermpy's built-in aberration.
            positions_table: QTable = QTable(
                [
                    Time(query_times),
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                ],
                names=["UTC", "X MSM", "Y MSM", "Z MSM"],
                units=["", "km", "km", "km"],
            )
            positions_table = rotate_to_aberrated_coordinates(positions_table)

            # region_values contains three items, the probabilitiy of that region along the trajectory, along with a lower and upper bound for 95% CI.
            probabilities, lower, upper = get_probability_at_position(
                positions_table, PROBABILITY_MAP
            )

            spacecraft_data[id] = {
                "Times": query_times,
                "Positions": positions_table,
                "Probabilities": probabilities,
                "95% Lower": lower,
                "95% Upper": upper,
            }

    # Plotting
    fig = plt.figure(figsize=(8, 4))

    grid = GridSpec(2, 2, width_ratios=[3, 1], figure=fig)

    spacecraft_data["MPO"]["Axis"] = fig.add_subplot(grid[0, 0])
    spacecraft_data["MMO"]["Axis"] = fig.add_subplot(grid[1, 0])

    trajectory_ax = fig.add_subplot(grid[:, 1])

    axes = [
        spacecraft_data["MPO"]["Axis"],
        spacecraft_data["MMO"]["Axis"],
        trajectory_ax,
    ]

    for i, id in enumerate(spacecraft):
        ax = spacecraft_data[id]["Axis"]

        region_names = ["Solar Wind", "Magnetosheath", "Magnetosphere"]
        for j, region in enumerate(region_names):

            ax.plot(
                spacecraft_data[id]["Times"],
                spacecraft_data[id]["Probabilities"][j],
                color=wong_colours[j],
                label=region,
                **LINE_PARAMS,
            )

            # Confidence intervals
            ax.fill_between(
                spacecraft_data[id]["Times"],
                spacecraft_data[id]["95% Lower"][j],
                spacecraft_data[id]["95% Upper"][j],
                color=wong_colours[j],
                alpha=0.3,
                label=r"$\quad$95% CI",
            )

        x = spacecraft_data[id]["Positions"]["X MSM'"] / Constants.MERCURY_RADIUS.to(
            u.km
        )
        y = spacecraft_data[id]["Positions"]["Y MSM'"] / Constants.MERCURY_RADIUS.to(
            u.km
        )
        z = spacecraft_data[id]["Positions"]["Z MSM'"] / Constants.MERCURY_RADIUS.to(
            u.km
        )
        cyl = np.sqrt(y**2 + z**2)

        trajectory_ax.plot(x, cyl, color=wong_colours[3 + i * 2], label=id, zorder=5)

    # Time series ax formatting
    for id in spacecraft:
        ax = spacecraft_data[id]["Axis"]

        ax.margins(x=0)
        ax.set_ylabel(f"{id}\nRegion Probability")

        ax.set_xlim(START_TIME, END_TIME)

    spacecraft_data["MMO"]["Axis"].xaxis.set_major_formatter(DateFormatter("%H:%M"))
    spacecraft_data["MMO"]["Axis"].text(
        -0.09, -0.3, "2027-05-02", transform=spacecraft_data["MMO"]["Axis"].transAxes
    )
    spacecraft_data["MMO"]["Axis"].xaxis.set_major_locator(MinuteLocator(interval=120))
    spacecraft_data["MPO"]["Axis"].xaxis.set_major_locator(MinuteLocator(interval=120))
    plt.setp(spacecraft_data["MPO"]["Axis"].get_xticklabels(), visible=False)

    # Trajectory ax formatting
    trajectory_handles = []
    trajectory_labels = []
    trajectory_ax.set_aspect("equal")
    trajectory_ax.set_xlim(-2, 6)
    trajectory_ax.set_ylim(0, 8)
    trajectory_ax.set_xlabel(r"$X_{\rm MSM'}\,\left[ R_{\rm M} \right]$")
    trajectory_ax.set_ylabel(
        r"$\left( Y_{\rm MSM'}^2 + Z_{\rm MSM'}^2 \right)^{0.5} \,\left[ R_{\rm M} \right]$"
    )

    circle = Circle(
        (0, Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS),
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
        zorder=5,
    )
    trajectory_ax.add_patch(circle)
    circle = Circle(
        (0, -1 * Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS),
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
        zorder=5,
    )
    trajectory_ax.add_patch(circle)

    plot_magnetospheric_boundaries(trajectory_ax)

    residence_data = PROBABILITY_MAP["Minutes In Bin"] / 60  # Convert to hours
    residence_mask = residence_data.values.T != 0

    # Add MESSENGER outline
    trajectory_ax.contourf(
        residence_data.coords["X MSM'"],
        residence_data.coords["CYL MSM'"],
        residence_mask,
        levels=[0, 0.5, 1],
        colors=["white", "lightgrey"],
        zorder=-2,
    )

    ax_handles, ax_labels = trajectory_ax.get_legend_handles_labels()
    trajectory_handles.extend(ax_handles)
    trajectory_labels.extend(ax_labels)

    # Contour doesn't apply in legends, so we need to fake it
    proxy = Patch(facecolor="lightgrey", edgecolor="none")
    trajectory_handles.append(proxy)
    trajectory_labels.append("MESSENGER's Coverage")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Legends
    spacecraft_data["MPO"]["Axis"].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3
    )
    fig.legend(
        trajectory_handles,
        trajectory_labels,
        loc="lower center",
        bbox_to_anchor=(0.85, 0.75),
    )

    # Add panel labels
    panel_labels = "abc"
    for i, ax in enumerate(axes):
        label_text = ax.text(
            0.01,
            0.9,
            f"({panel_labels[i]})",
            transform=ax.transAxes,
        )

        label_text.set_path_effects(
            [
                matplotlib.patheffects.Stroke(
                    linewidth=3, foreground="white", alpha=0.7
                ),
                matplotlib.patheffects.Normal(),
            ]
        )

    plt.savefig(FIG_OUTPUT, format="pdf")


if __name__ == "__main__":
    main()
