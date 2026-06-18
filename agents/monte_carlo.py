"""
Monte Carlo race-end simulator.

Given current race state, the subagent runs 5000 simulations of the remaining race for
each of two strategies (BOX_NOW vs STAY_OUT), tracking the focal driver's
finishing position in each. Outputs a distribution over finishing positions
per strategy.

The simulation is deliberately coarse: lap pace = base + (tyre_age * deg_rate) + noise,
with pit stops costing around 22s and safety cars compressing the field.
It's enough to capture the main factors of pit-stop decisions. it is not a 
replacement for a real F1 team's simulator.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from mcp_server.live_state import get_active_state
from mcp_server.server import (
    get_driver_lap_history,
    get_tyre_stints,
    get_gaps_to_rivals,
    get_current_race_state,
    historical_sc_rate,
)
from llm.client import LLMClient
from agents.schemas import MonteCarloAssessment, StrategyOutcome

# Degradation rates in seconds of lap-time added per lap of stint age
# Degradation rates are hard-coded approximations. Actual degradation rates vary by track and stint length
_DEGRADATION_RATES = {
    "SOFT": 0.08,
    "MEDIUM": 0.05,
    "HARD": 0.03,
    "INTERMEDIATE": 0.10,
    "WET": 0.15,
    "UNKNOWN": 0.06
}

# Total time taken in the pit-stop and the pit-lane
_PIT_LOSS_SECONDS = 22.0

# Standard distribution of the Gaussian noise term (noise ~ N(0, _LAP_TIME_NOISE^2))
_LAP_TIME_NOISE = 1.3

NUM_OF_SIMULATIONS = 5000

@dataclass
class _DriverSimState:
    """
    Per-driver state carried through one simulation.
    """
    code: str
    cumulative_time: float # seconds from session start
    base_pace: float       # seconds per "neutral" lap on fresh tyres
    tyre_age: int          # laps on current compound
    compound: str
    has_pitted_in_strategy: bool

def _simulate_one(
    drivers: list[_DriverSimState],
    focal_code: str,
    strategy: str,
    laps_to_race: int,
    sc_prob_per_lap: float,
    new_compound: str,
    rng: np.random.Generator,
) -> int:
    sim = [
    _DriverSimState(
        code=d.code,
        cumulative_time=d.cumulative_time,
        base_pace=d.base_pace,
        tyre_age=d.tyre_age,
        compound=d.compound,
        has_pitted_in_strategy=d.has_pitted_in_strategy,
        )
        for d in drivers
    ]

    for _ in range(laps_to_race):
        sc_this_lap = rng.random() < sc_prob_per_lap

        for d in sim:
            should_pit = d.code == focal_code and not d.has_pitted_in_strategy and strategy == "BOX_NOW"

            deg = _DEGRADATION_RATES.get(d.compound, _DEGRADATION_RATES.get("UNKNOWN"))
            lap_time = d.base_pace + (d.tyre_age * deg) + rng.normal(0.0, _LAP_TIME_NOISE)

            if should_pit:
                lap_time += _PIT_LOSS_SECONDS
                d.compound = new_compound
                d.tyre_age = 0
                d.has_pitted_in_strategy = True
            else:
                d.tyre_age += 1
            
            if sc_this_lap:
                lap_time *= 1.3 # Safety car slows everyone down
            
            d.cumulative_time += lap_time
        
    sim.sort(key=lambda d: d.cumulative_time)
    for position, d in enumerate(sim, start=1):
        if d.code == focal_code:
            return position

async def _summarise(
    client: LLMClient,
    model: str,
    driver_code: str,
    current_lap: int,
    box: StrategyOutcome,
    stay: StrategyOutcome,
) -> str:
    """One LLM call to turn the raw distributions into a sentence."""
    from pydantic import BaseModel, Field

    class _Interpretation(BaseModel):
        interpretation: str = Field(description="One-sentence plain English summary of the strategic difference between BOX_NOW and STAY_OUT.")

    system = """You are summarising a Monte Carlo race-end simulation for the orchestrator. Be neutral. Do not recommend a strategy. State the strategically meaningful difference in one sentence, citing the podium probability or expected position. If the difference is small (under 5% on podium probability), say so explicitly."""

    user = f"""Driver {driver_code} at lap {current_lap}.

BOX_NOW:  expected finish P{box.expected_position:.1f}, podium chance {box.p_podium*100:.0f}%, points chance {box.p_points*100:.0f}%
STAY_OUT: expected finish P{stay.expected_position:.1f}, podium chance {stay.p_podium*100:.0f}%, points chance {stay.p_points*100:.0f}%

Summarise the strategic difference in one neutral sentence."""

    result = await client.generate_structured(
        model=model,
        system_prompt=system,
        user_prompt=user,
        response_schema=_Interpretation,
        temperature=0.2,
    )
    return result.interpretation

def _build_distribution(positions: list[int], grid_size: int) -> dict[int, float]:
    """
    Turns a list of finishing positions into a {position: probability} dictionary.
    """
    counts = np.bincount(positions, minlength=grid_size + 1)[1:grid_size+1]
    probs = counts / counts.sum()
    
    return {pos:float(p) for pos, p in enumerate(probs, start=1) if p > 0.001}


def _estimate_base_pace(history) -> float:
    """
    Take the median of recent green-flag lap times as the base pace.
    """
    times = [lap.lap_time_seconds for lap in history.laps[-10:] if lap.lap_time_seconds]
    if not times:
        return 95.0
    return float(np.median(times))

async def assess_monte_carlo(
    client: LLMClient,
    driver_code: str,
    year: int,
    event: str,
    session_type: str,
    model: str = "gemini-2.5-flash-lite",
    new_compound: str = "MEDIUM",
) -> MonteCarloAssessment:
    """
    Runs Monte Carlo race-end simulation comparing BOX_NOW vs STAY_OUT.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active.")
    
    current_lap = state.current_lap()
    race = get_current_race_state()

    gaps = get_gaps_to_rivals(
        year=year,
        event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap
    )
    history = get_driver_lap_history(
        year=year,
        event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap
    )
    stints = get_tyre_stints(
        year=year,
        event=event,
        session_type=session_type,
        driver_code=driver_code,
        current_lap=current_lap
    )

    # Build initial drivers from the current gap picture.
    # The focal driver's cumulative time is the anchor; rivals' times are
    # derived from their gaps.
    focal_time = 0.0
    base_pace = _estimate_base_pace(history)


    # If the focal driver's current stint is very young (≤1 lap), they just
    # pitted in reality. Treat them as already-pitted in the simulation so
    # BOX_NOW doesn't pit them a second time.
    current_stint_age = (
        current_lap - stints.stints[-1].start_lap
        if stints.stints else current_lap
    )
    already_pitted = current_stint_age <= 1

    drivers: list[_DriverSimState] = [
        _DriverSimState(
            code=driver_code,
            cumulative_time=focal_time,
            base_pace=base_pace,
            tyre_age=current_stint_age,
            compound=stints.stints[-1].compound if stints.stints else "UNKNOWN",
            has_pitted_in_strategy=already_pitted,
        ),
    ]

    relevant_rivals = [
        r for r in gaps.rivals
        if r.gap_seconds is not None and abs(r.gap_seconds) <= 60.0
    ]

    for rival in relevant_rivals:
        if rival.gap_seconds is None:
            continue
        try:
            rival_history = get_driver_lap_history(
                year=year,
                event=event,
                session_type=session_type,
                driver_code=rival.rival_driver_code,
                current_lap=current_lap,
            )
            rival_stints = get_tyre_stints(
                year=year,
                event=event,
                session_type=session_type,
                driver_code=rival.rival_driver_code,
                current_lap=current_lap,
            )
            rival_pace = _estimate_base_pace(rival_history)
            rival_compound = rival_stints.stints[-1].compound if rival_stints.stints else "MEDIUM"
            rival_tyre_age = current_lap - (rival_stints.stints[-1].start_lap if rival_stints.stints else current_lap)
        except Exception:
            # Fallback if MCP query fails for this rival
            rival_pace = base_pace
            rival_compound = "MEDIUM"
            rival_tyre_age = current_lap // 2

        drivers.append(_DriverSimState(
            code=rival.rival_driver_code,
            cumulative_time=focal_time - rival.gap_seconds,
            base_pace=rival_pace,
            tyre_age=rival_tyre_age,
            compound=rival_compound,
            has_pitted_in_strategy=False,
        ))
    
    total_laps = 64 if event == "Monaco" else 60
    laps_remaining = max(total_laps - current_lap, 1)

    sc_rate = historical_sc_rate(
        event=event,
        lap_from=current_lap + 1,
        lap_to=total_laps,
    )
    sc_prob_per_lap = sc_rate.combined_probability / laps_remaining

    # If you're reached here, good job reading this far!
    # This is an easter egg from 'a Hitchhiker's Guide to the Galaxy' :)
    rng = np.random.default_rng(seed=42)
    box_now_positions: list[int] = []
    stay_out_positions: list[int] = []

    for _ in range(NUM_OF_SIMULATIONS):
        box_now_positions.append(_simulate_one(
            drivers, driver_code, "BOX_NOW", laps_remaining, sc_prob_per_lap, new_compound, rng,
        ))
        stay_out_positions.append(_simulate_one(
            drivers, driver_code, "STAY_OUT", laps_remaining, sc_prob_per_lap, new_compound, rng,
        ))

    grid_size = len(drivers)
    box_now_dist = _build_distribution(box_now_positions, grid_size)
    stay_out_dist = _build_distribution(stay_out_positions, grid_size)

    box_now_outcome = StrategyOutcome(
        strategy="BOX_NOW",
        expected_position=float(np.mean(box_now_positions)),
        position_distribution=box_now_dist,
        p_podium=sum(p for pos, p in box_now_dist.items() if pos <= 3),
        p_points=sum(p for pos, p in box_now_dist.items() if pos <= 10),
    )
    stay_out_outcome = StrategyOutcome(
        strategy="STAY_OUT",
        expected_position=float(np.mean(stay_out_positions)),
        position_distribution=stay_out_dist,
        p_podium=sum(p for pos, p in stay_out_dist.items() if pos <= 3),
        p_points=sum(p for pos, p in stay_out_dist.items() if pos <= 10),
    )

    interpretation = await _summarise(
        client,
        model,
        driver_code,
        current_lap,
        box_now_outcome,
        stay_out_outcome
    )

    return MonteCarloAssessment(
        simulations_run=NUM_OF_SIMULATIONS,
        box_now=box_now_outcome,
        stay_out=stay_out_outcome,
        interpretation=interpretation,
        confidence=0.65, # Honest mid-tier confidence for the coarse model
    )