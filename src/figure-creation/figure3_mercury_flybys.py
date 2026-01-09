"""
We want a figure to display all 6 flybys of Mercury as probability timeseries.
They will be subplots stacked on top of eachother, all centered on closest
approach.
"""

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.table import QTable
from astropy.time import Time, TimeDelta
from hermpy.data import rotate_to_aberrated_coordinates
from hermpy.net import ClientSPICE
from planetary_coverage import MetaKernel
from sunpy.time import TimeRange

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from get_probabilities import get_probability_at_position, load_probability_maps

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


def main():
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

            axis_data.append(
                {
                    "Minutes Since Closest Approach": (Time(query_times) - ca).to(
                        "minute"
                    ),
                    "Positions": positions_table,
                    "Distance": dipole_distance,
                    "Probabilities": get_probability_at_position(
                        positions_table, PROBABILITY_MAP
                    ),
                }
            )

    fig, axes = plt.subplots(len(axis_data), 1, sharex=True, sharey=True)

    for i, ax in enumerate(axes):

        data = axis_data[i]

        ax.plot(
            data["Minutes Since Closest Approach"],
            data["Probabilities"][0],
            color="yellow",
        )
        ax.plot(
            data["Minutes Since Closest Approach"],
            data["Probabilities"][1],
            color="orange",
        )
        ax.plot(
            data["Minutes Since Closest Approach"],
            data["Probabilities"][2],
            color="blue",
        )

    plt.show()


if __name__ == "__main__":
    main()
