"""
Race configuration when simulating a race using Pitwall.
"""

# RACE SELECTOR
YEAR = 2025
EVENT = "Qatar"
SESSION_TYPE = "R"
DRIVER_CODE = "PIA"

# LAP WINDOW
START_LAP = 1
END_LAP = 57
TOTAL_LAPS = 57 # Full Grand Prix length (used for SC probability and laps_remaining)

# REPLAY TUNING
REPLAY_SPEED = 3.0

DISPLAY_NAME = f"{YEAR} {EVENT} Grand Prix"
