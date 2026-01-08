"""
We start by making a dataset of region observations from the MESSENGER mission.
This script takes an input list of bow shock and magnetopause crossings and
comparse data timestamps to these to determine which magnetospheric region
MESSENGER was in at that time and position.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import spiceypy as spice
from astropy.table import QTable
from astropy.time import Time
from hermpy.data import CrossingList, InstantEventList, rotate_to_aberrated_coordinates
from hermpy.net import ClientSPICE
from sunpy.time import TimeRange

OUTPUT_FILE = Path(__file__).parent.parent / "resources/messenger_regions.ecsv"


def main():
    print("Fetching SPICE kernels")
    SPICE_CLIENT = ClientSPICE()

    with SPICE_CLIENT.KernelPool():
        print("Fetching MESSENGER positions")
        messenger_positions: QTable = get_messenger_positions()
        print("Fetching MESSENGER boundary crossings")
        messenger_boundary_crossings: InstantEventList = get_messenger_crossings()

    # For each position (row) in messenger_positions, we want to compare the
    # timestamp to the crossings list. A quick way to do this is to first merge
    # the tables by adding two new columns to messenger_posisons: 'Previous
    # Crossing' and 'Next Crossing'. From these, we can work out with a lookup
    # table which region MESSENGER was in at that time.
    print("Merging tables")
    merged_table = merge_crossings_to_positions_table(
        messenger_positions, messenger_boundary_crossings.table
    )

    print("Determining MESSENGER regions")
    region_table = determine_messenger_regions(merged_table)

    # The last think we will do to make our life easier is adding an additional
    # position column CYL MSM', which is the distance from the x-axis.
    region_table["CYL MSM'"] = np.sqrt(
        region_table["Y MSM'"] ** 2 + region_table["Z MSM'"] ** 2
    )

    # Reorder the columns for my sanity
    region_table = region_table[
        [
            "UTC",
            "X MSM'",
            "Y MSM'",
            "Z MSM'",
            "CYL MSM'",
            "Region",
        ]
    ]

    # Save this
    region_table.write(OUTPUT_FILE, format="ascii.ecsv", overwrite=True)
    print(f"MESSENGER region data saved to: {OUTPUT_FILE}")


def determine_messenger_regions(table: QTable) -> QTable:

    previous_crossing_table = {
        "BS_OUT": "Solar Wind",
        "BS_IN": "Magnetosheath",
        "MP_OUT": "Magnetosheath",
        "MP_IN": "Magnetosphere",
        "UNPHYSICAL (MSp -> SW)": "Solar Wind",
        "UNPHYSICAL (SW -> MSp)": "Magnetosphere",
    }
    next_crossing_table = {
        "BS_OUT": "Magnetosheath",
        "BS_IN": "Solar Wind",
        "MP_OUT": "Magnetosphere",
        "MP_IN": "Magnetosheath",
        "UNPHYSICAL (MSp -> SW)": "Magnteosphere",
        "UNPHYSICAL (SW -> MSp)": "Solar Wind",
    }

    table["Region (from previous crossing)"] = [
        previous_crossing_table[crossing]
        for crossing in table["Previous Crossing Label"]
    ]

    table["Region (from next crossing)"] = [
        next_crossing_table[crossing] for crossing in table["Next Crossing Label"]
    ]

    # Remove rows where the crossings before and after disagree on the region.
    table = table[
        table["Region (from next crossing)"] == table["Region (from previous crossing)"]
    ]

    # Then we are safe to assign.
    table["Region"] = table["Region (from next crossing)"]

    # Only keep columns we want
    table.remove_columns(
        [
            "Previous Crossing Label",
            "Next Crossing Label",
            "Region (from previous crossing)",
            "Region (from next crossing)",
        ]
    )

    return table


def merge_crossings_to_positions_table(
    positions_table: QTable, crossings_table: QTable, cache_file: Path | None = None
) -> QTable:
    """
    positions_tabke and crossings_table must be sorted in time.
    """

    if cache_file == None:
        cache_file = (
            Path(__file__).parent.parent
            / "resources/messenger_positions_crossings_merged_cache.ecsv"
        )

    if cache_file.exists():
        return QTable.read(cache_file, format="ascii.ecsv")

    merged_table = QTable()

    # Add the columns we want to keep from positions_table
    for column in ["UTC", "X MSM'", "Y MSM'", "Z MSM'"]:
        merged_table[column] = positions_table[column]

    # Find the closest crossing before each time.
    previous_crossing_indices = (
        np.searchsorted(crossings_table["UTC"], positions_table["UTC"], side="right")
        - 1
    )
    next_crossing_indices = previous_crossing_indices + 1

    # Add associated crossing labels
    merged_table["Previous Crossing Label"] = crossings_table["Label"][
        previous_crossing_indices
    ]
    merged_table["Next Crossing Label"] = crossings_table["Label"][
        next_crossing_indices
    ]

    # Cache so we don't need to recaculate each run
    merged_table.write(cache_file, format="ascii.ecsv", overwrite=True)

    return merged_table


def get_messenger_crossings() -> InstantEventList:

    # Load crossing list from file
    messenger_crossing_list_location = (
        Path(__file__).parent.parent / "resources/hollman_2025_crossing_list.csv"
    )
    crossing_list = CrossingList.from_csv(
        messenger_crossing_list_location, time_column="Time"
    )
    crossing_list.table.rename_column("Time", "UTC")

    # Add positions
    ets = spice.datetime2et(
        [t.to_datetime(leap_second_strict="warn") for t in crossing_list.table["UTC"]]
    )

    # Get the position data for these times in MSM coordinates.
    positions, _ = spice.spkpos("MESSENGER", ets, "MSGR_MSM", "NONE", "MERCURY")

    # Add these positions to the crossing list table
    crossing_list.table.add_columns(
        [
            positions[:, 0] * u.km,
            positions[:, 1] * u.km,
            (positions[:, 2] * u.km),
        ],
        names=["X MSM", "Y MSM", "Z MSM"],
    )

    # Rotate into aberrated coordinate system
    crossing_list.table = rotate_to_aberrated_coordinates(crossing_list.table)

    # Remove unaberrated columns
    crossing_list.table.remove_columns(["X MSM", "Y MSM", "Z MSM", "Aberration Angle"])

    return crossing_list


def get_messenger_positions(cache_file: Path | None = None) -> QTable:

    if cache_file == None:
        cache_file = (
            Path(__file__).parent.parent / "resources/messenger_position_cache.ecsv"
        )

    if cache_file.exists():
        return QTable.read(cache_file, format="ascii.ecsv")

    # It can be slow to keep calculating this for the whole mission, so we
    # cache to disk and check if it exists already before recalculating.

    # We first define the length of time to look at. These are roughly the start
    # and end dates for the mission (with a safe margin on the ends).

    # full_mission_time_range = TimeRange("2011-04-01", "2015-03-01")
    full_mission_time_range = TimeRange("2011-04-01", "2015-03-01")

    # Find a list of times between the bounds of the time range and
    resolution = 1 * u.minute
    query_times: list[Time] = [
        t.start
        for t in full_mission_time_range.window(cadence=resolution, window=resolution)
    ]

    ets = spice.datetime2et(
        [t.to_datetime(leap_second_strict="warn") for t in query_times]
    )

    # Get the position data for these times in MSM coordinates.
    positions, _ = spice.spkpos("MESSENGER", ets, "MSGR_MSM", "NONE", "MERCURY")

    # Convert these positions into an astropy table
    messenger_positions: QTable = QTable(
        [
            query_times,
            positions[:, 0] * u.km,
            positions[:, 1] * u.km,
            (positions[:, 2] * u.km),
        ],
        names=["UTC", "X MSM", "Y MSM", "Z MSM"],
    )

    # Rotate into aberrated coordinate system
    messenger_positions = rotate_to_aberrated_coordinates(messenger_positions)

    # Remove unaberrated columns
    messenger_positions.remove_columns(["X MSM", "Y MSM", "Z MSM", "Aberration Angle"])

    # Cache the file for quicker loading in future
    messenger_positions.write(cache_file, format="ascii.ecsv", overwrite=True)

    return messenger_positions


if __name__ == "__main__":
    main()
