"""
There are some observations of magnetosheath along an orbit which would be
expected to be excluively solar wind in nominal conditions. This script serves
to explore those orbits.
"""

from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.time import TimeDelta
from hermpy.utils import Constants

X_MIN = 3 * Constants.MERCURY_RADIUS
X_MAX = 4 * Constants.MERCURY_RADIUS
CYL_MIN = 2 * Constants.MERCURY_RADIUS
CYL_MAX = 4 * Constants.MERCURY_RADIUS

MESSENGER_REGIONS_FILE = (
    Path(__file__).parent.parent.parent / "resources/messenger_regions.ecsv"
)

# Load the observations dataset
messenger_regions = QTable.read(MESSENGER_REGIONS_FILE, format="ascii.ecsv")

# Filter by location
x_condition = (messenger_regions["X MSM'"] > X_MIN) & (
    messenger_regions["X MSM'"] < X_MAX
)
cyl_condition = (messenger_regions["CYL MSM'"] > CYL_MIN) & (
    messenger_regions["CYL MSM'"] < CYL_MAX
)
messenger_regions = messenger_regions[x_condition & cyl_condition]

# We can count the orbits by removing all rows that differ from the previous by 20 minutes
times = messenger_regions["UTC"]
time_differences = times[1:] - times[:-1]

orbit_breaks = time_differences > TimeDelta(20 * 60, format="sec")

# Indices where a new orbit starts
break_indices = np.where(orbit_breaks)[0] + 1
orbit_start_indices = np.concatenate(([0], break_indices))
orbit_end_indices = np.concatenate((break_indices - 1, [len(times) - 1]))

orbit_durations = times[orbit_end_indices] - times[orbit_start_indices]

longest_orbit_idx = np.argmax(orbit_durations)

longest_start = orbit_start_indices[longest_orbit_idx]
longest_end = orbit_end_indices[longest_orbit_idx]
longest_duration = orbit_durations[longest_orbit_idx]

longest_orbit_regions = messenger_regions[longest_start : longest_end + 1]

print(f"Longest time in region: {longest_duration.to("min")}")
print(f"Orbit index: {longest_orbit_idx}")

messenger_regions_longest_orbit = messenger_regions[longest_start : longest_end + 1]

print(messenger_regions_longest_orbit)
