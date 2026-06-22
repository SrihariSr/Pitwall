"""
Pydantic schemas for subagent outputs.

Every subagent returns a small, tightly-scoped object. The orchestrator
reads all of them in one prompt to fuse into a single pit-stop call.

Schemas are intentionally minimal: fewer tokens means less hallucination surface,
easier for the orchestrator to reason about.
"""

from pydantic import BaseModel, Field

class TyreAssessment(BaseModel):
    """The Tyre Strategist's verdict on the focal driver's current stint."""

    has_sufficient_data: bool = Field(
        description=(
            "True if there's enough lap data to make a meaningful cliff "
            "projection. False if the stint is too young (under 3 laps) or "
            "data is missing. When False, cliff_lap is meaningless and "
            "should be ignored by consumers."
        )
    )
    cliff_lap: int = Field(
        description=(
            "The lap by which we expect the tyre to hit its performance "
            "cliff. Only meaningful when has_sufficient_data=True."
        )
    )
    confidence: float = Field(
        description=(
            "0.0-1.0 confidence in the cliff_lap estimate. Should be low "
            "(<0.4) when has_sufficient_data=False."
        ),
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="One sentence justifying the call, citing specific evidence from the lap data."
    )

class RivalGapEntry(BaseModel):
    """A single rival's gap as the Gap Analyst sees it.

    Minimal: only the rivals the orchestrator actually needs to know about.
    The full per-rival data is in MCP; this is the curated summary.
    """

    driver_code: str = Field(description="3-letter rival code")
    position: int = Field(description="Rival's track position")
    gap_seconds: float = Field(
        description=(
            "Time gap in seconds. Positive = rival is AHEAD on track. "
            "Negative = rival is behind."
        )
    )
    relationship: str = Field(
        description=(
            "How this rival relates strategically: "
            "'undercut_threat' (within pit-window behind), "
            "'overcut_target' (within pit-window ahead, pitting first might be exploitable), "
            "'direct_battle' (within DRS/overtaking range), "
            "'context' (worth knowing but not strategically active)."
        )
    )


class GapAssessment(BaseModel):
    """
    The Gap Analyst's verdict on the focal driver's strategic gap situation.
    """

    focal_position: int = Field(description="Focal driver's current track position")
    undercut_threat: str = Field(
        description=(
            "Severity of undercut threat from drivers behind: "
            "'high' (clear and imminent risk), 'medium' (plausible), "
            "'low' (theoretically possible but unlikely), 'none' (no threat)."
        )
    )
    overcut_opportunity: str = Field(
        description=(
            "Whether an overcut on rivals ahead is on the table: "
            "'high' (clear opportunity), 'medium' (situational), "
            "'low' (unlikely to pay off), 'none' (no opportunity)."
        )
    )
    closest_rivals: list[RivalGapEntry] = Field(
        description=(
            "The 2-4 most strategically relevant rivals, in track-position order. "
            "Includes immediate threats and immediate opportunities, not the whole field."
        )
    )
    reasoning: str = Field(
        description="One sentence citing specific gaps and dynamics, not generalities."
    )
    confidence: float = Field(
        description=(
            "0.0-1.0 confidence in the assessment. Lower when the field is "
            "in flux (recent pit stops shuffling order, SC just ended)."
        ),
        ge=0.0,
        le=1.0
    )

class StrategyOutcome(BaseModel):
    """
    Finishing-position distribution for one strategy option.
    """

    strategy: str = Field(description="Strategy label: 'BOX_NOW' or 'STAY_OUT'")
    expected_position: float = Field(
        description="Expected (mean) finishing position across the simulations"
    )
    position_distribution: dict[int, float] = Field(
        description=(
            "Probability of finishing at each position. Keys are positions "
            "(1 to grid size, typically 20-22 depending on the season), "
            "values are probabilities (0-1) summing to ~1.0."
        )
    )
    p_podium: float = Field(
        description="Probability of finishing P1-P3, 0-1",
        ge=0.0,
        le=1.0
    )
    p_points: float = Field(
        description="Probability of finishing P1-P10 (in the points), 0-1",
        ge=0.0,
        le=1.0
    )


class MonteCarloAssessment(BaseModel):
    """The Monte Carlo simulator's verdict comparing strategies.

    The simulator does NOT recommend a strategy — it presents the distributions.
    The Orchestrator decides which strategy to act on.
    """

    simulations_run: int = Field(description="Number of simulations per strategy")
    box_now: StrategyOutcome = Field(description="Outcome distribution for boxing this lap")
    stay_out: StrategyOutcome = Field(description="Outcome distribution for staying out")
    interpretation: str = Field(
        description=(
            "One-sentence plain-language interpretation of the distributions, "
            "highlighting the strategically meaningful difference (or lack of "
            "one)."
        )
    )
    confidence: float = Field(
        description=(
            "0-1 confidence in the simulation's relevance. Lower when the "
            "model's assumptions are likely violated (e.g. rapidly changing "
            "weather, anomalous SC density, novel circuit conditions)."
        ),
        ge=0.0,
        le=1.0
    )

class PitDecision(BaseModel):
    """
    The Orchestrator's pit-stop call, fused from all subagent inputs.

    This is the single output the system commits to and logs. Downstream
    consumers (dashboard, radio narrator, post-race analysis) read this.
    """

    call: str = Field(
        description=(
            "The strategic call. One of: "
            "'BOX_THIS_LAP' (commit to pit this lap), "
            "'BOX_NEXT_LAP' (prepare to pit, signal the crew), "
            "'STAY_OUT' (explicit decision not to pit), "
            "'EXTEND' (commit to a longer stint, e.g. for an overcut), "
            "'PIT_WINDOW_OPEN' (pitting now would be defensible but not "
            "the only option), "
            "'MONITOR' (no actionable change, keep watching)."
        )
    )
    confidence: float = Field(
        description=(
            "0-1 confidence in the call. Below 0.5 the call is exploratory; "
            "above 0.8 the data points clearly one way."
        ),
        ge=0.0,
        le=1.0
    )
    primary_reason: str = Field(
        description=(
            "One sentence stating the dominant reason for this call, "
            "citing which subagent(s) drove the decision."
        )
    )
    supporting_factors: list[str] = Field(
        description=(
            "Up to 4 secondary considerations that informed the call. "
            "Each a short phrase, e.g. 'Tyre Strategist: cliff at L18', "
            "'Gap Analyst: PER in undercut range'."
        )
    )
    risks: list[str] = Field(
        description=(
            "Up to 3 things that could make this call wrong. Honesty about "
            "downside scenarios. Example: 'If safety car deploys L22-25, "
            "BOX_NOW costs us the cheap stop.'"
        )
    )
    trigger: str = Field(
        description=(
            "What woke the orchestrator: 'scheduled', 'track_status', "
            "'rival_pit', 'manual'. Useful for post-race auditing."
        )
    )
    change_from_previous: str | None = Field(
    default=None,
    description=(
        "One sentence explaining how this decision relates to your most "
        "recent previous decision. Use null only for the first cycle of "
        "a race. Be explicit about whether you are maintaining a previous "
        "call, pivoting, or repeating a recommendation that was not executed."
    ),
)

class SafetyCarAssessment(BaseModel):
    """
    The Safety Car Oracle's verdict on SC/VSC probability in the upcoming window.

    Combines historical base rates with current race context to produce an
    adjusted estimate. Does not recommend strategy.
    """

    lap_window_from: int = Field(description="Start of the lap window assessed (typically current_lap + 1)")
    lap_window_to: int = Field(description="End of the lap window assessed")
    historical_probability: float = Field(
        description="Combined SC+VSC probability from historical races at this circuit and window",
        ge=0.0,
        le=1.0
    )
    adjusted_probability: float = Field(
        description="Probability adjusted for current race context (track status, weather)",
        ge=0.0,
        le=1.0
    )
    direction: str = Field(
        description=(
            "How current context shifts probability vs the historical baseline: "
            "'elevated' (>=1.3x baseline), 'normal' (within ~20% of baseline), "
            "'depressed' (<=0.7x baseline)."
        )
    )
    reasoning: str = Field(
        description="One sentence justifying the adjustment, citing specific signals from current race state."
    )
    confidence: float = Field(
        description="0-1 confidence in the assessment. Lower when historical sample size is small or context is ambiguous.",
        ge=0.0,
        le=1.0
    )

class WeatherAssessment(BaseModel):
    """
    The Weather Watcher's verdict on current conditions and optimal compound.

    Reports what the track is doing now and what compound it calls for.
    Does not recommend pit stops, that's the Orchestrator's job.
    """

    current_condition: str = Field(
        description=(
            "The current weather/track condition. One of: 'dry', 'damp', "
            "'light_wet', 'heavy_wet', 'drying' (rain stopped, track clearing), "
            "or 'wetting' (rain just started, track still mostly dry)."
        )
    )
    optimal_compound: str = Field(
        description="The optimal tyre compound family for current conditions: 'slicks', 'inters', or 'wets'."
    )
    pivot_urgency: str = Field(
        description=(
            "How urgently a compound change may be needed. One of: "
            "'immediate' (conditions warrant a pivot now), "
            "'soon' (pivot likely within 3-5 laps), "
            "'stable' (no pivot expected within the lookahead)."
        )
    )
    reasoning: str = Field(
        description="One sentence justifying the assessment, citing specific weather trends or sample data."
    )
    confidence: float = Field(
        description="0-1 confidence in the assessment. Lower during transitional or volatile conditions.",
        ge=0.0,
        le=1.0
    )

class RivalPrediction(BaseModel):
    """Prediction for a single rival's strategic state."""

    driver_code: str = Field(description="3-letter driver code, e.g. 'SAI'")
    current_compound: str = Field(
        description="Current tyre compound: 'SOFT', 'MEDIUM', 'HARD', 'INTER', 'WET', or 'UNKNOWN'."
    )
    current_stint_age: int = Field(
        description="Number of laps completed on the current set of tyres."
    )
    predicted_pit_lap: int = Field(
        description="The lap on which this rival is most likely to make their next pit stop."
    )
    threat_window: str = Field(
        description=(
            "When their next stop is expected. One of: 'now' (within 3 laps), "
            "'soon' (3-8 laps), 'later' (8+ laps)."
        )
    )


class RivalAssessment(BaseModel):
    """The Rival Modeler's view of opposing strategies in the immediate window."""

    rivals: list[RivalPrediction] = Field(
        description="Predictions for the closest rivals, ordered by strategic threat."
    )
    primary_threat_driver: str = Field(
        description=(
            "Driver code of the rival posing the biggest strategic threat: "
            "the one whose move would most disrupt our race position."
        )
    )
    reasoning: str = Field(
        description="One sentence justifying the primary threat assessment with specific evidence."
    )
    confidence: float = Field(
        description="0-1 confidence. Lower when stint data is sparse or conditions transitional.",
        ge=0.0,
        le=1.0
    )

class RadioMessage(BaseModel):
    """
    A race-engineer radio message generated from a PitDecision.

    Different from the other subagent outputs: this is post-processing,
    converting the Chief Strategist's structured decision into the language
    a race engineer would actually use on team radio to the driver.
    """

    urgency: str = Field(
        description=(
            "Message urgency. One of: 'critical' (immediate driver action), "
            "'info' (status update, no action change), "
            "'planning' (heads-up about future events)."
        )
    )
    primary_call: str = Field(
        description=(
            "Headline imperative, 2-8 words. E.g. 'Box this lap, box this lap', "
            "'Stay out, stay out', 'Window opens next lap'."
        )
    )
    full_message: str = Field(
        description=(
            "Complete radio message as a race engineer would deliver it. "
            "1-3 short sentences, calm and concise, with the WHY included briefly."
        )
    )
    