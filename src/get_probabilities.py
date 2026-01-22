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
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    # For each position, we need to:
    # - convert to cylindrical coordinates
    # - compare position with map
    # - record a probability value for solar wind, magnetosheath, and
    # magneosphere.

    positions = positions.copy()

    # Convert to X-CYL
    positions["CYL MSM'"] = np.sqrt(positions["Y MSM'"] ** 2 + positions["Z MSM'"] ** 2)

    # The maps are in units of radii, so we must convert.
    positions["X MSM'"] = positions["X MSM'"] / Constants.MERCURY_RADIUS.to(u.km)
    positions["CYL MSM'"] = positions["CYL MSM'"] / Constants.MERCURY_RADIUS.to(u.km)

    # Compare positions with map.
    x_coords = probability_map.coords["X MSM'"].values
    bin_size = x_coords[1] - x_coords[0]

    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    cyl_bins = np.arange(0, 8 + bin_size, bin_size)

    # Digitize the trajectory data into bin indices
    x_indices = np.digitize(positions["X MSM'"], x_bins) - 1
    cyl_indices = np.digitize(positions["CYL MSM'"], cyl_bins) - 1

    # Iterrate through position indices and assign probabilities
    solar_wind_prob, magnetosheath_prob, magnetosphere_prob = [], [], []
    solar_wind_lower, magnetosheath_lower, magnetosphere_lower = [], [], []
    solar_wind_upper, magnetosheath_upper, magnetosphere_upper = [], [], []

    for ix, ic in zip(x_indices, cyl_indices):
        if 0 <= ix < len(x_bins) - 1 and 0 <= ic < len(cyl_bins) - 1:
            # Solar wind
            solar_wind_lower.append(
                probability_map["Solar Wind 95% Lower"][ix, ic].item()
            )
            solar_wind_prob.append(probability_map["Solar Wind"][ix, ic].item())
            solar_wind_upper.append(
                probability_map["Solar Wind 95% Upper"][ix, ic].item()
            )

            # Magnetosheath
            magnetosheath_lower.append(
                probability_map["Magnetosheath 95% Lower"][ix, ic].item()
            )
            magnetosheath_prob.append(probability_map["Magnetosheath"][ix, ic].item())
            magnetosheath_upper.append(
                probability_map["Magnetosheath 95% Upper"][ix, ic].item()
            )

            # Magnetosphere
            magnetosphere_lower.append(
                probability_map["Magnetosphere 95% Lower"][ix, ic].item()
            )
            magnetosphere_prob.append(probability_map["Magnetosphere"][ix, ic].item())
            magnetosphere_upper.append(
                probability_map["Magnetosphere 95% Upper"][ix, ic].item()
            )
        else:
            # Out-of-range positions
            solar_wind_lower.append(np.nan)
            solar_wind_prob.append(np.nan)
            solar_wind_upper.append(np.nan)

            magnetosheath_lower.append(np.nan)
            magnetosheath_prob.append(np.nan)
            magnetosheath_upper.append(np.nan)

            magnetosphere_lower.append(np.nan)
            magnetosphere_prob.append(np.nan)
            magnetosphere_upper.append(np.nan)

    probabilities = [solar_wind_prob, magnetosheath_prob, magnetosphere_prob]
    lower = [solar_wind_lower, magnetosheath_lower, magnetosphere_lower]
    upper = [solar_wind_upper, magnetosheath_upper, magnetosphere_upper]

    return probabilities, lower, upper
