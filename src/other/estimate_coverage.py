"""
We want to estimate the ratio of time BepiColombo spacecraft will spend within
MESSENGER's coverage, and outside it. To do this, we determine predictions over
the course of 1 Mercury year, and calculate the ratio of np.nan measurements to
data.
"""

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice
from astropy.table import QTable
from astropy.time import Time
from hermpy.data import rotate_to_aberrated_coordinates
from hermpy.net import ClientSPICE
from planetary_coverage import MetaKernel
from sunpy.time import TimeRange

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from get_probabilities import get_probability_at_position, load_probability_maps

START_TIME = dt.datetime(2027, 5, 2, 4)

MERCURY_YEAR = dt.timedelta(days=88)
END_TIME = START_TIME + MERCURY_YEAR

# Number of spice measurements in that time. Thi only needs to be an estimate,
# so we can be pretty liberal with the cadence.
CADENCE = dt.timedelta(minutes=10)
TIME_RESOLUTION = int((END_TIME - START_TIME) / CADENCE)

PROBABILITY_MAP = load_probability_maps(
    Path(__file__).parent.parent.parent / "resources/region_probability_maps.nc"
)


def main():
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
            probabilities, _, _ = get_probability_at_position(
                positions_table, PROBABILITY_MAP
            )

            spacecraft_data[id] = {
                "Times": query_times,
                "Positions": positions_table,
                "Probabilities": probabilities,
            }

    cadence_minutes = CADENCE.total_seconds() / 60
    num_spacecraft = len(spacecraft)

    total_time_out_of_coverage = 0

    total_time_per_spacecraft = len(spacecraft_data["MMO"]["Times"]) * cadence_minutes

    for id in spacecraft:
        num_samples_outside_coverage = np.isnan(
            spacecraft_data[id]["Probabilities"][0]
        ).sum()

        time_out = num_samples_outside_coverage * cadence_minutes
        time_in = total_time_per_spacecraft - time_out
        fraction_in = time_in / total_time_per_spacecraft

        spacecraft_data[id]["Time out of coverage"] = time_out
        spacecraft_data[id]["Time in coverage"] = time_in
        spacecraft_data[id]["Fraction in coverage"] = fraction_in

        total_time_out_of_coverage += time_out

        print(fraction_in)

    print(1 - total_time_out_of_coverage / (total_time_per_spacecraft * num_spacecraft))


if __name__ == "__main__":
    main()
