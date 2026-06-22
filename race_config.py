"""
Race configuration when simulating a race using Pitwall.
"""

# RACE SELECTOR
YEAR = 2024
EVENT = "São Paulo"
SESSION_TYPE = "R"
DRIVER_CODE = "NOR"

# LAP WINDOW
START_LAP = 1
END_LAP = 69
TOTAL_LAPS = 71 # Full Grand Prix length (used for SC probability and laps_remaining)

# REPLAY TUNING
REPLAY_SPEED = 3.0

DISPLAY_NAME = f"{YEAR} {EVENT} Grand Prix"
