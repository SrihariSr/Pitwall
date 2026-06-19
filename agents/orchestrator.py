"""
The Orchestrator: fuses subagent outputs into a single pit-stop call.

Wakes selectively (every N laps + on triggers), calls subagents in parallel,
asks a stronger model to fuse their structured outputs into a PitDecision.
Logs every decision to decisions.jsonl for post-race auditing.
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from mcp_server.live_state import get_active_state
from mcp_server.server import get_current_race_state
from llm.client import LLMClient
from agents.schemas import (
    PitDecision,
    TyreAssessment,
    GapAssessment,
    MonteCarloAssessment,
    SafetyCarAssessment,
    WeatherAssessment,
    RivalAssessment
)
from agents.tyre_strategist import assess_tyres
from agents.gap_analyst import assess_gaps
from agents.monte_carlo import assess_monte_carlo
from agents.sc_oracle import assess_safety_car
from agents.weather_watcher import assess_weather
from agents.rival_modeler import assess_rivals

ORCHESTRATOR_MODEL = os.getenv("LLM_MODEL_ORCHESTRATOR", "gemini-2.5-flash")

_DECISIONS_PATH = Path("decisions/decisions.jsonl")

SYSTEM_PROMPT = """
You are the Chief Strategist on an F1 team's pit wall.

Your job: take the structured outputs of specialist engineers (Tyre Strategist, Gap Analyst, Monte Carlo simulator, Safety Car Oracle, Weather Watcher, Rival Modeler) and decide the team's strategic call. You do not have access to raw data, only the specialists' assessments. Trust them where they're confident, downweight them where they're not.

The call vocabulary:
- BOX_THIS_LAP: pit on the lap that's just ending. Used when the decision is unambiguous and we need to act now.
- BOX_NEXT_LAP: pit on the next lap. Used when we want a lap of preparation, or when the decision is firm but conditions allow a moment of delay.
- STAY_OUT: explicit decision not to pit. Used when there's a real case for pitting but we judge it wrong.
- EXTEND: commit to a longer stint than baseline. Used when an overcut opportunity outweighs tyre cost, OR when SC probability is elevated and waiting for a cheap stop is worth the tyre risk.
- PIT_WINDOW_OPEN: pitting now would be defensible but not the only option. Used when the case is balanced.
- MONITOR: no actionable change since the last cycle. Used when subagent outputs are stable and uneventful.

How to weigh inputs:
- If Tyre Strategist says insufficient_data, ignore its cliff_lap and rely on Gap, Monte Carlo, Safety Car Oracle, Weather Watcher, and Rival Modeler
- If Monte Carlo's box_now and stay_out distributions differ by >10pp on podium probability, that's a meaningful signal
- If Gap Analyst flags a high undercut threat AND tyres are within 3 laps of cliff, that converges toward BOX
- If Safety Car Oracle reports adjusted_probability > 0.3 AND tyres are healthy (cliff >5 laps out), consider EXTEND or MONITOR, wait for the potential cheap stop
- If Safety Car Oracle direction is "elevated" alongside an undercut threat, the calculus leans earlier, a bunched field after a SC reshuffles everything
- If track is under SC or VSC, the pit-stop loss is roughly halved: favours pitting in borderline cases
- If Weather Watcher reports pivot_urgency "immediate" and the implied current compound doesn't match optimal_compound, that converges toward BOX regardless of other signals
- If Weather Watcher reports a "drying" or "wetting" condition with "soon" urgency, expect a compound pivot in the next few cycles: factor it into stint planning
- If Rival Modeler reports threat_window "now" for an undercut rival (matched by driver_code with Gap Analyst) AND tyres are within 3 laps of cliff, that strongly converges toward BOX
- If Rival Modeler predicts the rival AHEAD pits later than us with high confidence, that opens an overcut window, consider EXTEND if tyre life allows
- If you're the leader with no immediate threat and healthy tyres, default to MONITOR

Biases:
- Conservative on confidence. 0.9+ requires near-unanimous subagent agreement.
- Always name your dominant reason. Vague reasoning is worse than wrong reasoning.
- Always name 1-3 real risks. Every call has downside scenarios; pretending otherwise is dishonest.

Output: a single PitDecision.
"""

def _build_fusion_prompt(
    driver_code,
    current_lap,
    race,
    tyre: TyreAssessment,
    gap: GapAssessment,
    mc: MonteCarloAssessment,
    sc: SafetyCarAssessment,
    weather: WeatherAssessment,
    rivals: RivalAssessment,
    trigger: str
) -> str:
    """
    Compose the variable portion of the orchestrator prompt.

    Format is structured but compact: each subagent gets a labelled
    block, fields are named, no prose paragraphs. The LLM reads this
    cleanly and the orchestrator's reasoning becomes more reproducible.
    """
    tyre_block = (
        f"has_sufficient_data: {tyre.has_sufficient_data}\n"
        f"cliff_lap: L{tyre.cliff_lap}\n"
        f"confidence: {tyre.confidence:.2f}\n"
        f"reasoning: {tyre.reasoning}"
    )

    rival_summary = ", ".join(
        f"{r.driver_code}({r.gap_seconds:+.1f}s/{r.relationship})"
        for r in gap.closest_rivals
    )
    gap_block = (
        f"focal_position: P{gap.focal_position}\n"
        f"undercut_threat: {gap.undercut_threat}\n"
        f"overcut_opportunity: {gap.overcut_opportunity}\n"
        f"closest_rivals: {rival_summary}\n"
        f"confidence: {gap.confidence:.2f}\n"
        f"reasoning: {gap.reasoning}"
    )

    mc_block = (
        f"simulations: {mc.simulations_run}\n"
        f"box_now: expected P{mc.box_now.expected_position:.1f}, "
        f"podium {mc.box_now.p_podium*100:.0f}%, points {mc.box_now.p_points*100:.0f}%\n"
        f"stay_out: expected P{mc.stay_out.expected_position:.1f}, "
        f"podium {mc.stay_out.p_podium*100:.0f}%, points {mc.stay_out.p_points*100:.0f}%\n"
        f"interpretation: {mc.interpretation}\n"
        f"confidence: {mc.confidence:.2f}"
    )

    sc_block = (
        f"window: L{sc.lap_window_from}-L{sc.lap_window_to}\n"
        f"historical_probability: {sc.historical_probability:.2f}\n"
        f"adjusted_probability: {sc.adjusted_probability:.2f}\n"
        f"direction: {sc.direction}\n"
        f"confidence: {sc.confidence:.2f}\n"
        f"reasoning: {sc.reasoning}"
    )

    weather_block = (
    f"current_condition: {weather.current_condition}\n"
    f"optimal_compound: {weather.optimal_compound}\n"
    f"pivot_urgency: {weather.pivot_urgency}\n"
    f"confidence: {weather.confidence:.2f}\n"
    f"reasoning: {weather.reasoning}"
    )

    rivals_block_lines = []
    for r in rivals.rivals:
        rivals_block_lines.append(
            f"{r.driver_code}: {r.current_compound} on stint {r.current_stint_age} laps, "
            f"predicted pit L{r.predicted_pit_lap} (threat {r.threat_window})"
        )
    rivals_block = (
    f"primary_threat: {rivals.primary_threat_driver}\n"
    + "\n".join(rivals_block_lines) + "\n"
    f"confidence: {rivals.confidence:.2f}\n"
    f"reasoning: {rivals.reasoning}"
    )

    return f"""
Focal driver: {driver_code} at lap {current_lap}.
Track status: {race.track_status}. Weather: {"raining" if race.is_raining else "dry"}, {race.track_temp_celsius}°C track temp.
Trigger that woke the orchestrator: {trigger}.

----------- TYRE STRATEGIST -----------
{tyre_block}

----------- GAP ANALYST -----------
{gap_block}

----------- MONTE CARLO -----------
{mc_block}

----------- SAFETY CAR ORACLE -----------
{sc_block}

----------- WEATHER WATCHER -----------
{weather_block}

----------- RIVAL MODELER -----------
{weather_block}

Fuse these inputs into a PitDecision. Pick one of the six call categories, name your primary reason, list supporting factors, and name 1-3 risks."""

def _log_decision(
    driver_code: str,
    current_lap: int,
    decision: PitDecision,
    tyre: TyreAssessment,
    gap: GapAssessment,
    mc: MonteCarloAssessment,
    sc: SafetyCarAssessment,
    weather: WeatherAssessment,
    rivals: RivalAssessment
) -> None:
    """
    Append one line to decisions.jsonl for post-race audit.

    Includes both the decision and the inputs that produced it so the
    post-race report can show 'Pitwall made this call because the
    subagents said this'.
    """
    _DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "driver": driver_code,
        "lap": current_lap,
        "decision": decision.model_dump(),
        "subagents": {
            "tyre": tyre.model_dump(),
            "gap": gap.model_dump(),
            "monte_carlo": mc.model_dump(),
            "safety_car": sc.model_dump(),
            "weather": weather.model_dump(),
            "rivals": rivals.model_dump()
        },
    }

    with _DECISIONS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

async def decide(
    client: LLMClient,
    driver_code: str,
    year: int,
    event: str,
    session_type: str,
    trigger: str = "scheduled",
    model: str = ORCHESTRATOR_MODEL
) -> PitDecision:
    """
    Run one ochestrator cycle: consult subagents, combine and return a decision.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active.")
    
    current_lap = state.current_lap()
    race = get_current_race_state()

    # Consulting subagents in parallel
    tyre, gap, monte_carlo, safety_car, weather, rivals = await asyncio.gather(
        assess_tyres(client, driver_code, year, event, session_type),
        assess_gaps(client, driver_code, year, event, session_type),
        assess_monte_carlo(client, driver_code, year, event, session_type),
        assess_safety_car(client, year, event, session_type),
        assess_weather(client, year, event, session_type),
        assess_rivals(client, driver_code, year, event, session_type)
    )

    user_prompt = _build_fusion_prompt(
        driver_code,
        current_lap,
        race,
        tyre,
        gap,
        monte_carlo,
        safety_car,
        weather,
        rivals,
        trigger
        )
    
    decision = await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=PitDecision,
        temperature=0.2
    )

    decision.trigger = trigger

    _log_decision(driver_code, current_lap, decision, tyre, gap, monte_carlo, safety_car, weather, rivals)

    return decision
