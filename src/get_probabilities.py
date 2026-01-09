from pathlib import Path

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.table import QTable
from hermpy.utils import Constants


def load_probability_maps(file: Path) -> xr.Dataset:
    return xr.load_dataset(file)


def get_probability_at_position(
    positions: QTable, probability_map: xr.Dataset
) -> tuple[list[float], list[float], list[float]]:

    # For each position, we need to:
    # - convert to cylindrical coordinates
    # - compare position with map
    # - record a probability value for solar wind, magnetosheath, and
    # magneosphere.

    # Convert to X-CYL
    positions["CYL MSM'"] = np.sqrt(positions["Y MSM'"] ** 2 + positions["Z MSM'"] ** 2)

    # The maps are in units of radii, so we must convert.
    positions["X MSM'"] = positions["X MSM'"] / Constants.MERCURY_RADIUS.to(u.km)
    positions["CYL MSM'"] = positions["CYL MSM'"] / Constants.MERCURY_RADIUS.to(u.km)

    # Compare positions with map.
    x_coords = probability_map.coords["X MSM'"].values
    cyl_coords = probability_map.coords["CYL MSM'"].values
    bin_size = x_coords[1] - x_coords[0]

    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    cyl_bins = np.arange(0, 8 + bin_size, bin_size)

    # Digitize the trajectory data into bin indices
    x_indices = np.digitize(positions["X MSM'"], x_bins) - 1
    cyl_indices = np.digitize(positions["CYL MSM'"], cyl_bins) - 1

    # Iterrate through position indices and assign probabilities
    solar_wind_values: list[float] = []
    magnetosheath_values: list[float] = []
    magnetosphere_values: list[float] = []

    for ix, ic in zip(x_indices, cyl_indices):

        # Ensure the index is within the valid histogram range
        if 0 <= ix < len(x_bins) - 1 and 0 <= ic < len(cyl_bins) - 1:

            solar_wind_values.append(probability_map["Solar Wind"][ix, ic].item())
            magnetosheath_values.append(probability_map["Magnetosheath"][ix, ic].item())
            magnetosphere_values.append(probability_map["Magnetosphere"][ix, ic].item())

        else:

            solar_wind_values.append(np.nan)
            magnetosheath_values.append(np.nan)
            magnetosphere_values.append(np.nan)

    return (solar_wind_values, magnetosheath_values, magnetosphere_values)
