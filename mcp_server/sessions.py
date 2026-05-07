import fastf1
from pathlib import Path
from functools import lru_cache

# One-time setup: tell FastF1 where to cache its raw API responses on disk.
_cache_dir = Path("data/fastf1_cache")
_cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(_cache_dir))

@lru_cache(maxsize=8)
def load_session(year: int, event: str, session_type: str):
    """
    Load and cache a FastF1 session.

    Parameters:
        year: e.g. 2022
        event: e.g. "Monaco" or "Hungarian Grand Prix" or round number as string
        session_type: "R" (race), "Q" (quali), "FP1"/"FP2"/"FP3", "S" (sprint)

    Returns:
        A loaded fastf1.core.Session object.

    The result is cached in memory by argument tuple, so calling this
    function multiple times with the same args is instant after the first.
    """
    session = fastf1.get_session(year, event, session_type)
    session.load()
    return session