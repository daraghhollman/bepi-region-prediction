"""
We want a figure to display all 6 flybys of Mercury as probability timeseries.
They will be subplots stacked on top of eachother, all centered on closest
approach.
"""

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
from astropy.time import Time, TimeDelta
from hermpy.data import rotate_to_aberrated_coordinates
from hermpy.net import ClientSPICE
from hermpy.utils import Constants
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, Rectangle
from planetary_coverage import MetaKernel
from sunpy.time import TimeRange

# A 7-colour colourblind friendly palette. Wong et al. Nature Methods.
wong_colours = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

LINE_PARAMS = {
    "lw": 2,
    "path_effects": [  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2.5, foreground="black"),
        matplotlib.patheffects.Normal(),
    ],
}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from get_probabilities import (get_probability_at_position,
                               load_probability_maps)

# These were yoinked from ESA press:
# https://www.cosmos.esa.int/web/bepicolombo/home
CLOSEST_APPROACH_TIMES: list[Time] = [
    Time("2021-10-01T23:34"),  # 1
    Time("2022-06-23T09:44"),  # 2
    Time("2023-06-19T19:34"),  # 3
    Time("2024-09-04T21:48"),  # 4
    Time("2024-12-01T14:23"),  # 5
    Time("2025-01-08T05:59"),  # 6
]

# Define some time buffer to plot around these times
TIME_BUFFER = TimeDelta("2hr")
TIME_RESOLUTION = 1000  # Number of spice measurements in that time.

PROBABILITY_MAP = load_probability_maps(
    Path(__file__).parent.parent.parent / "resources/region_probability_maps.nc"
)

FIG_OUTPUT = Path(__file__).parent.parent.parent / "figures/figure2_flybys.pdf"


def main():

    # Insure directory exists
    if not os.path.isdir(FIG_OUTPUT.parent):
        os.makedirs(FIG_OUTPUT.parent)

    # Load SPICE data for these times. With the setup script, we downloaded spice
    # kernels which we must load.
    spice_client = ClientSPICE()

    # Planetary coverage has some nice code to handle updating path values for
    # spice metakernels at runtime.
    bepi_metakernel = MetaKernel(
        Path(__file__).parent.parent.parent
        / "resources/bepicolombo-spice/kernels/mk/bc_ops.tm",
        kernels=Path(__file__).parent.parent.parent
        / "resources/bepicolombo-spice/kernels/",
    )

    spice_client.add_local_kernels(bepi_metakernel.kernels)

    axis_data: list[dict[str, Any]] = []

    with spice_client.KernelPool():

        # Both MPO and MMO were in functionally the same position for these flybys,
        # so it doesn't matter for which we query.
        for ca in CLOSEST_APPROACH_TIMES:
            ca_time_range = TimeRange(ca - TIME_BUFFER, ca + TIME_BUFFER)
            query_times = [
                t.center.to_datetime() for t in ca_time_range.split(TIME_RESOLUTION)
            ]

            ets = spice.datetime2et(query_times)
            positions, _ = spice.spkpos("MPO", ets, "BC_MSM", "NONE", "MERCURY")
            dipole_distance = np.linalg.norm(positions, axis=1)

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

            axis_data.append(
                {
                    "Minutes Since Closest Approach": (Time(query_times) - ca).to(
                        "minute"
                    ),
                    "Positions": positions_table,
                    "Distance": dipole_distance,
                    "Probabilities": probabilities,
                    "95% Lower": lower,
                    "95% Upper": upper,
                }
            )

    fig = plt.figure(figsize=(8, 8))

    fig.text(0.02, 0.41, "Region Probability", fontsize="large", rotation=90)

    # Some fancy nested grid logic to have panels with different ratios
    outer_grid = GridSpec(1, 2, width_ratios=[5, 2])
    timeseries_grid = GridSpecFromSubplotSpec(6, 1, subplot_spec=outer_grid[0])
    trajectory_grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[1])

    timeseries_axes = []
    trajectory_axes = []

    for i in range(len(axis_data)):

        previous_ax = timeseries_axes[i - 1] if i != 0 else None

        ax = fig.add_subplot(timeseries_grid[i], sharex=previous_ax, sharey=previous_ax)
        timeseries_axes.append(ax)

    for i in range(3):
        ax = fig.add_subplot(trajectory_grid[i])
        trajectory_axes.append(ax)

    timeseries_labels = "abcdef"
    for i, ax in enumerate(timeseries_axes):

        data = axis_data[i]

        # Fixup the nan values for the sake of plotting. Keep all nans where
        # all three regions give nan, otherwise set to 0.
        probabiltiies_array = np.array(data["Probabilities"], dtype=float)
        all_nan_mask = np.all(np.isnan(probabiltiies_array), axis=0)
        data["Probabilities"] = np.where(
            np.isnan(probabiltiies_array) & ~all_nan_mask, 0, probabiltiies_array
        )

        # Probability time series
        for j, (region_name, colour) in enumerate(
            zip(["Solar Wind", "Magnetosheath", "Magnetosphere"], [3, 0, 1])
        ):
            ax.plot(
                data["Minutes Since Closest Approach"].value,
                data["Probabilities"][j],
                color=wong_colours[colour],
                label=region_name,
                **LINE_PARAMS,
            )

            # Confidence intervals
            ax.fill_between(
                data["Minutes Since Closest Approach"].value,
                data["95% Lower"][j],
                data["95% Upper"][j],
                color=wong_colours[colour],
                alpha=0.5,
            )

        # Add panel label
        ax.text(0.01, 0.85, f"({timeseries_labels[i]})", transform=ax.transAxes)

        # Add an identifier for each trajectory to the flyby timeseries panel
        position = (0.95, 0.8)
        square = Rectangle(
            position,
            0.04,
            0.16,
            transform=ax.transAxes,
            facecolor=wong_colours[i],
            edgecolor="black",
        )
        ax.add_patch(square)

        ax.text(
            position[0] - 0.01,
            position[1] + 0.08,
            f"Flyby {i + 1}",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

        if ax == timeseries_axes[0]:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=3, fancybox=True
            )

        if ax == timeseries_axes[-1]:
            ax.set_xlabel("Minutes After Closest Approach")

        for ax in timeseries_axes[:-1]:
            ax.tick_params(labelbottom=False)

        x = data["Positions"]["X MSM'"] / Constants.MERCURY_RADIUS.to(u.km)
        y = data["Positions"]["Y MSM'"] / Constants.MERCURY_RADIUS.to(u.km)
        z = data["Positions"]["Z MSM'"] / Constants.MERCURY_RADIUS.to(u.km)

        trajectory_axes[0].plot(x, y, color=wong_colours[i], **LINE_PARAMS)
        trajectory_axes[1].plot(x, z, color=wong_colours[i], **LINE_PARAMS)
        trajectory_axes[2].plot(y, z, color=wong_colours[i], **LINE_PARAMS)

    trajectory_axes[0].set_xlabel(r"$X_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")
    trajectory_axes[0].set_ylabel(r"$Y_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

    trajectory_axes[1].set_xlabel(r"$X_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")
    trajectory_axes[1].set_ylabel(r"$Z_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

    trajectory_axes[2].set_xlabel(r"$Y_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")
    trajectory_axes[2].set_ylabel(r"$Z_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

    trajectory_labels = "ghi"
    for i, ax in enumerate(trajectory_axes):
        ax.text(0.02, 0.9, f"({trajectory_labels[i]})", transform=ax.transAxes)
        ax.set_aspect("equal")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        # Add Mercury
        circle = Circle(
            (0, 0), 1, edgecolor="black", facecolor="none", linewidth=2, zorder=5
        )
        ax.add_patch(circle)

    plt.tight_layout(pad=2.5)
    plt.savefig(FIG_OUTPUT, format="pdf")


if __name__ == "__main__":
    main()
