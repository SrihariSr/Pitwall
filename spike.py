"""
Phase 0 spike: prove we can pull real F1 data through FastF1.
Loads the 2022 Monaco GP and prints Leclerc's lap times + pit stops.
"""
import fastf1
from pathlib import Path

# Set up a cache folder so we don't re-download data every run.
cache_dir = Path("data/fastf1_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Load the 2022 Monaco GP race session.
print("Loading 2022 Monaco GP race session...")
session = fastf1.get_session(2022, "Monaco", "R")
session.load()

# Pull Leclerc's laps.
leclerc_laps = session.laps.pick_drivers("LEC")

print(f"\nLeclerc completed {len(leclerc_laps)} laps")
print(f"Fastest lap: {leclerc_laps['LapTime'].min()}")

print("\nFirst 10 laps:")
print(leclerc_laps[["LapNumber", "LapTime", "Compound", "TyreLife", "PitInTime", "PitOutTime"]].head(10))

# Identify pit stops
pit_laps = leclerc_laps[leclerc_laps["PitInTime"].notna()]
print(f"\nPit stops on laps: {pit_laps['LapNumber'].tolist()}")