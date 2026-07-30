from typing import List
from typing import Tuple
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
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
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

    x_indices = np.digitize(positions["X MSM'"], x_bins) - 1
    cyl_indices = np.digitize(positions["CYL MSM'"], cyl_bins) - 1

    in_range = (
        (x_indices >= 0) & (x_indices < len(x_bins) - 1) &
        (cyl_indices >= 0) & (cyl_indices < len(cyl_bins) - 1)
    )

    # Clip so out-of-range indices don't raise
    safe_x_indices = np.clip(x_indices, 0, len(x_bins) - 2)
    safe_cyl_indices = np.clip(cyl_indices, 0, len(cyl_bins) - 2)

    regions = [
        "Solar Wind", "Magnetosheath", "Magnetosphere",
    ]

    probabilities, lower, upper = [], [], []
    for name in regions:
        prob_vals = probability_map[name].values
        lower_vals = probability_map[f"{name} 95% Lower"].values
        upper_vals = probability_map[f"{name} 95% Upper"].values

        p = prob_vals[safe_x_indices, safe_cyl_indices]
        l = lower_vals[safe_x_indices, safe_cyl_indices]
        u_ = upper_vals[safe_x_indices, safe_cyl_indices]

        # Mask out-of-range points as NaN
        p = np.where(in_range, p, np.nan)
        l = np.where(in_range, l, np.nan)
        u_ = np.where(in_range, u_, np.nan)

        probabilities.append(p.tolist())
        lower.append(l.tolist())
        upper.append(u_.tolist())

    return probabilities, lower, upper
