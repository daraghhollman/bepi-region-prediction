"""
We can visually pick out the average bow shock and magnetopuase as the 0.5
contour in the probability maps determined. We want to compare those contours
to the average postions as determined by Winslow et al. (2013).
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hermpy.plotting import plot_magnetospheric_boundaries
from scipy.spatial import distance

PROBABILITIY_MAPS_FILE = (
    Path(__file__).parent.parent.parent / "resources/region_probability_maps.nc"
)

FIG_OUTPUT = (
    Path(__file__).parent.parent.parent / "figures/compare-boundary-positions.pdf"
)


def main():

    # Ensure output dir exists, if not, create it
    if not os.path.isdir(FIG_OUTPUT.parent):
        os.makedirs(FIG_OUTPUT.parent)

    # Load probabillity maps
    probability_maps = xr.load_dataset(PROBABILITIY_MAPS_FILE)

    # We interpolate to avoid large discrete jumps
    cell_size = 0.05
    interpolated_coords = {
        "X MSM'": np.arange(-5, 5, cell_size),
        "CYL MSM'": np.arange(0, 10, cell_size),
    }
    probability_maps = probability_maps.interp(interpolated_coords)

    contour_bounds = (0.45, 0.55)
    bow_shock_contour = (probability_maps["Solar Wind"] >= contour_bounds[0]) & (
        probability_maps["Solar Wind"] <= contour_bounds[1]
    )
    magnetopause_contour = (probability_maps["Magnetosphere"] >= contour_bounds[0]) & (
        probability_maps["Magnetosphere"] <= contour_bounds[1]
    )

    bs_points = (
        bow_shock_contour.where(bow_shock_contour)
        .stack(points=("X MSM'", "CYL MSM'"))
        .dropna("points")
    )
    mp_points = (
        magnetopause_contour.where(magnetopause_contour)
        .stack(points=("X MSM'", "CYL MSM'"))
        .dropna("points")
    )

    bs_contour = np.column_stack(
        (bs_points["X MSM'"].values, bs_points["CYL MSM'"].values)
    )
    mp_contour = np.column_stack(
        (mp_points["X MSM'"].values, mp_points["CYL MSM'"].values)
    )

    (bs_model_x, bs_model_y), (mp_model_x, mp_model_y) = get_boundary_points()
    bs_model = np.column_stack((bs_model_x, bs_model_y))
    mp_model = np.column_stack((mp_model_x, mp_model_y))

    # GET MINIMUM DISTANCE
    # This is directional: this ordering means that we determine a distance for
    # each contour point.
    bs_distances = distance.cdist(bs_contour, bs_model)
    mp_distances = distance.cdist(mp_contour, mp_model)

    bs_min_distance = bs_distances.min(axis=1)
    mp_min_distance = mp_distances.min(axis=1)

    print(
        f"Average bow shock min distance: {bs_min_distance.mean():.3f} +/- {bs_min_distance.std():.3f}"
    )
    print(
        f"Average magnetopause min distance: {mp_min_distance.mean():.3f} +/- {mp_min_distance.std():.3f}"
    )

    # PLOTTING
    _, ax = plt.subplots()

    ax.scatter(
        bs_points["X MSM'"].values, bs_points["CYL MSM'"].values, s=1, color="#0072B2"
    )
    ax.scatter(
        mp_points["X MSM'"].values, mp_points["CYL MSM'"].values, s=1, color="#D55E00"
    )

    plot_magnetospheric_boundaries(ax)

    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)

    plt.show()


def get_boundary_points():

    sub_solar_magnetopause: float = 1.45
    alpha: float = 0.5
    psi: float = 1.04
    p: float = 2.75
    initial_x: float = 0.5

    phi = np.linspace(0, 2 * np.pi, 1000)
    rho = sub_solar_magnetopause * (2 / (1 + np.cos(phi))) ** alpha

    magnetopause_x_coords = rho * np.cos(phi)
    magnetopause_y_coords = rho * np.sin(phi)

    L = psi * p

    rho = L / (1 + psi * np.cos(phi))

    bowshock_x_coords = initial_x + rho * np.cos(phi)
    bowshock_y_coords = rho * np.sin(phi)

    # Bow shock functional form creates non-physical points far sunward of Mercury.
    # These are incorrect and must be removed.
    bowshock_y_coords = bowshock_y_coords[bowshock_x_coords < 2]
    bowshock_x_coords = bowshock_x_coords[bowshock_x_coords < 2]

    return (
        (bowshock_x_coords, bowshock_y_coords),
        (magnetopause_x_coords, magnetopause_y_coords),
    )


if __name__ == "__main__":
    main()
