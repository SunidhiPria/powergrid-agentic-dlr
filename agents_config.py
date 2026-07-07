"""
agents_config.py
================
CrewAI multi-agent orchestration for the POWERGRID Agentic DLR System.

Three-agent sequential pipeline
---------------------------------
1. WeatherAnalyticsAgent   — reads sensor data, reports conditions
2. ConductorThermalAgent   — applies IEEE 738 physics, computes ampacity
3. GridDispatcherAgent     — executes DLR update, verifies grid safety

The agents share a structured context object passed between tasks.
The LLM backend is configurable: Google Gemini 1.5 Flash (cloud) or
a local Ollama model (offline / air-gapped POWERGRID environments).

Usage
-----
    from agents_config import run_dlr_crew
    result = run_dlr_crew(ambient_temp=35.0, wind_speed=18.0)
"""

import os
import logging
from typing import Any

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from physics_dlr import calculate_dynamic_ampacity, get_physics_breakdown
from grid_engine import (
    create_Kahalgaon_BiharSharif_network,
    run_base_load_flow,
    update_line_rating_tool,
    estimate_curtailment_mw,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

def _get_llm():
    """
    Resolve and return the LLM instance for agent reasoning.

    Priority order:
      1. Google Gemini 1.5 Flash  (if GOOGLE_API_KEY env var is set)
      2. Ollama local model        (offline fallback — "mistral" or any local)

    Returns
    -------
    LLM instance compatible with crewai.Agent's `llm` parameter.
    """
    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    if google_api_key:
        try:
            from crewai import LLM

            llm = LLM(
                model="gemini/gemini-2.0-flash",
                api_key=google_api_key,
                temperature=0.1,
            )

            logger.info("LLM backend: Google Gemini 1.5 Flash (cloud)")
            return llm
        except ImportError:
            logger.warning(
                "langchain_google_genai not installed. "
                "Falling back to Ollama. Run: pip install langchain-google-genai"
            )

        # Ollama offline fallback
    try:
        from crewai import LLM

        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")

        llm = LLM(
            model=f"ollama/{ollama_model}",
            base_url="http://localhost:11434"
        )

        logger.info("LLM backend: Ollama local model '%s'", ollama_model)
        return llm

    except ImportError:
        logger.warning(
            "Neither langchain_google_genai nor langchain_ollama available. "
            "Agents will run in tool-only mode with no LLM reasoning text."
        )
        return None


# ---------------------------------------------------------------------------
# CrewAI Tool wrappers
# ---------------------------------------------------------------------------

class WeatherSensorTool(BaseTool):
    """
    Simulates reading ambient temperature and wind speed from a regional
    IoT sensor network deployed along the Kahalgaon–Bihar Sharif corridor.

    In production, this would call a REST API (e.g., POWERGRID SCADA /
    IMD weather station). Here the values are injected via the tool's
    `_run` method at call time.
    """

    name: str = "WeatherSensorReadTool"
    description: str = (
        "Read current ambient temperature (°C) and perpendicular wind speed "
        "(km/h) from the Kahalgaon–Bihar Sharif corridor sensor network. "
        "Returns a structured report with sensor readings."
    )

    # Injected at construction time by run_dlr_crew()
    ambient_temp: float = 42.0
    wind_speed: float = 0.0

    def _run(self, query: str = "") -> str:
        report = (
            f"=== POWERGRID Corridor Weather Report ===\n"
            f"Corridor: Kahalgaon–Bihar Sharif 400 kV (95 km)\n"
            f"Sensor timestamp: Real-time simulation\n"
            f"Ambient Temperature : {self.ambient_temp:.1f} °C\n"
            f"Wind Speed (perp.)  : {self.wind_speed:.1f} km/h\n"
            f"Wind Angle to Line  : 90° (dominant crosswind)\n"
            f"Solar Irradiance    : 1000 W/m² (clear sky, Eastern India)\n"
            f"Conductor Type      : ACSR Twin Moose (400 kV class)\n"
            f"Status: All sensors nominal. Data transmitted.\n"
        )
        return report


class DLRPhysicsCalculatorTool(BaseTool):
    """
    Applies the IEEE 738 steady-state heat balance equation to calculate
    the real-time maximum safe conductor ampacity (Dynamic Line Rating).
    """

    name: str = "IEEE738DLRCalculatorTool"
    description: str = (
        "Calculate real-time conductor ampacity using IEEE 738 thermal physics. "
        "Input: 'ambient_temp=<value>, wind_speed=<value>' as a string. "
        "Returns full heat balance breakdown and safe ampacity in kA."
    )

    ambient_temp: float = 42.0
    wind_speed: float = 0.0

    def _run(self, query: str = "") -> str:
        breakdown = get_physics_breakdown(self.ambient_temp, self.wind_speed)

        report = (
            f"=== IEEE 738 Conductor Thermal Analysis ===\n"
            f"Conductor: ACSR Twin Moose (400 kV, 95 km)\n"
            f"Input Conditions:\n"
            f"  Ambient Temperature  : {self.ambient_temp:.1f} °C\n"
            f"  Wind Speed           : {self.wind_speed:.1f} km/h\n"
            f"Heat Balance Components (W/m):\n"
            f"  Q_c (Forced Convection Cooling) : {breakdown['q_c_w_per_m']:.3f}\n"
            f"  Q_r (Radiative Cooling)          : {breakdown['q_r_w_per_m']:.3f}\n"
            f"  Q_s (Solar Heat Gain)            : {breakdown['q_s_w_per_m']:.3f}\n"
            f"  Net Cooling (Q_c+Q_r-Q_s)        : {breakdown['net_cooling_w_per_m']:.3f}\n"
            f"  R(T_c=85°C)                      : {breakdown['r_tc_ohm_per_m']:.6e} Ω/m\n"
            f"IEEE 738 Result:\n"
            f"  Dynamic Ampacity (I_max)         : {breakdown['ampacity_a']:.1f} A\n"
            f"  Dynamic Ampacity (I_max)         : {breakdown['ampacity_ka']:.4f} kA\n"
            f"  Static Base Rating               : 500.0 A (0.5000 kA)\n"
            f"  DLR Uplift vs Static             : "
            f"{((breakdown['ampacity_ka']/0.5 - 1)*100):.1f}%\n"
            f"Recommendation: Update line max_i_ka to {breakdown['ampacity_ka']:.4f} kA\n"
        )
        return report


class GridDispatchTool(BaseTool):
    """
    Executes a Dynamic Line Rating update on the pandapower network model
    and reports the resulting line loading and safety status.

    The Grid Dispatcher Agent calls this tool to implement the DLR decision.
    """

    name: str = "GridDLRDispatchTool"
    description: str = (
        "Execute a Dynamic Line Rating update on the Kahalgaon–Bihar Sharif 400 kV line. "
        "Updates the thermal limit and re-runs power flow. "
        "Input: new ampacity in kA as a decimal string (e.g., '0.6200'). "
        "Returns: base loading %, DLR loading %, curtailment saved, financial impact."
    )

    ambient_temp: float = 42.0
    wind_speed: float = 0.0

    def _run(self, new_ampacity_ka_str: str = "0.5") -> str:
        try:
            new_ampacity_ka = float(new_ampacity_ka_str.strip())
        except ValueError:
            # Fallback: compute fresh from physics
            new_ampacity_ka = calculate_dynamic_ampacity(
                self.ambient_temp, self.wind_speed
            )

        # Fresh network for base case
        net_static = create_Kahalgaon_BiharSharif_network()
        loading_static = run_base_load_flow(net_static)

        # DLR network
        net_dlr = create_Kahalgaon_BiharSharif_network()
        loading_dlr, is_safe = update_line_rating_tool(net_dlr, new_ampacity_ka)

        # Curtailment economics
        curtailed_mw = estimate_curtailment_mw(loading_static, loading_dlr)

        # Wheeling revenue: assume ₹4.5/kWh average wind energy rate, 1-hour window
        revenue_saved_lakh = (curtailed_mw * 1000 * 4.5) / 1e5  # in ₹ Lakh

        safety_status = "✅ WITHIN THERMAL LIMITS" if is_safe else "⚠️  THERMAL VIOLATION"

        report = (
            f"=== Grid Dispatch Control Action ===\n"
            f"Corridor: Kahalgaon–Bihar Sharif 400 kV\n"
            f"Action: Dynamic Line Rating Update\n"
            f"New Thermal Ceiling: {new_ampacity_ka:.4f} kA ({new_ampacity_ka*1000:.0f} A)\n"
            f"\nLoad-Flow Results:\n"
            f"  Static Rating (500A)  → Line Loading: {loading_static:.2f}%\n"
            f"  DLR Rating ({new_ampacity_ka*1000:.0f}A)    → Line Loading: {loading_dlr:.2f}%\n"
            f"  Loading Delta: {loading_static - loading_dlr:.2f}% reduction\n"
            f"\nRenewable Energy Recovery:\n"
            f"  Wind Power Curtailment Avoided  : {curtailed_mw:.1f} MW\n"
            f"  Wheeling Revenue Saved (1 hr)   : ₹{revenue_saved_lakh:.2f} Lakh\n"
            f"\nSafety Assessment: {safety_status}\n"
            f"Decision: DLR update {'COMMITTED to SCADA' if is_safe else 'HELD PENDING REVIEW'}\n"
        )
        return report


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def _build_agents(
    ambient_temp: float,
    wind_speed: float,
    llm: Any,
) -> tuple:
    """
    Instantiate the three CrewAI agents with their tools and LLM.

    Returns
    -------
    tuple
        (weather_agent, thermal_agent, dispatcher_agent)
    """
    weather_tool = WeatherSensorTool(
        ambient_temp=ambient_temp,
        wind_speed=wind_speed,
    )
    dlr_tool = DLRPhysicsCalculatorTool(
        ambient_temp=ambient_temp,
        wind_speed=wind_speed,
    )
    dispatch_tool = GridDispatchTool(
        ambient_temp=ambient_temp,
        wind_speed=wind_speed,
    )

    # ------------------------------------------------------------------
    # Agent 1: Weather Analytics Agent
    # ------------------------------------------------------------------
    weather_agent = Agent(
        role="Senior Meteorological Analyst — POWERGRID Grid Monitoring Cell",
        goal=(
            "Continuously monitor and report real-time ambient temperature and "
            "localised perpendicular wind speed conditions along the "
            "Kahalgaon–Bihar Sharif 400 kV transmission corridor using distributed "
            "IoT sensor data. Provide precise, timestamped weather snapshots "
            "for downstream thermal analysis."
        ),
        backstory=(
            "You are a seasoned meteorologist embedded within the POWERGRID "
            "Eastern Regional Load Despatch Centre. You have 15 years "
            "of experience correlating IMD weather station data with transmission "
            "line performance. You know that even a 5 km/h increase in crosswind "
            "along the Bihar Sharif corridor can unlock tens of MW of stranded wind "
            "generation. Your reports are the first step in a physics-driven "
            "DLR decision chain."
        ),
        tools=[weather_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ------------------------------------------------------------------
    # Agent 2: Conductor Thermal Safety Agent
    # ------------------------------------------------------------------
    thermal_agent = Agent(
        role="High-Voltage Transmission Thermal Engineer — IEEE 738 Specialist",
        goal=(
            "Process real-time corridor weather data through the IEEE 738 "
            "steady-state conductor heat balance equation to compute the "
            "precise maximum allowable conductor ampacity (Dynamic Line Rating). "
            "Ensure every ampacity calculation is physically validated and "
            "within the 500 A–700 A ACSR Twin Moose operating band."
        ),
        backstory=(
            "You hold a PhD in Power Systems Engineering from IIT Madras and "
            "spent 10 years at POWERGRID's High Voltage Research Laboratory in "
            "Manesar. You authored POWERGRID's internal DLR implementation "
            "guidelines. You treat the IEEE 738 standard with religious precision. "
            "You know that the Twin Moose conductor can experience significant "
            "additional thermal capacity during favorable weather conditions "
            "over Eastern India—capacity that is currently being wasted due to "
            "static ratings. "
            "You are the scientific conscience of this system."
        ),
        tools=[dlr_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ------------------------------------------------------------------
    # Agent 3: Grid Dispatcher Agent
    # ------------------------------------------------------------------
    dispatcher_agent = Agent(
        role="Autonomous Grid Control Room Dispatcher — Southern Region",
        goal=(
            "Receive the physics-validated dynamic ampacity from the thermal "
            "engineer, execute the thermal limit upgrade on the Kahalgaon–Bihar Sharif "
            "line via the SCADA control interface, verify post-update grid safety "
            "through power-flow re-computation, and report the MW of wind energy "
            "recovered and financial wheeling revenue saved for POWERGRID records."
        ),
        backstory=(
            "You are the senior duty dispatcher at the Southern Regional Load "
            "Despatch Centre with 20 years on shift. You have watched thousands "
            "of MW of renewable energy being curtailed due to conservative static "
            "line ratings that were set decades ago. You are "
            "authorised under CERC Grid Code Regulation 5.7 to implement DLR "
            "updates when backed by certified thermal analysis. You act decisively "
            "but never compromise safety — every control action you take is logged, "
            "physics-backed, and reversible within 30 seconds."
        ),
        tools=[dispatch_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    return weather_agent, thermal_agent, dispatcher_agent


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _build_tasks(
    weather_agent: Agent,
    thermal_agent: Agent,
    dispatcher_agent: Agent,
    ambient_temp: float,
    wind_speed: float,
) -> list:
    """
    Define the three sequential CrewAI tasks.

    Returns
    -------
    list[Task]
        Tasks in execution order: weather → thermal → dispatch.
    """
    task_weather = Task(
        description=(
            f"Read the current meteorological conditions from the Kahalgaon–Bihar Sharif "
            f"corridor sensor network (ambient_temp={ambient_temp}°C, "
            f"wind_speed={wind_speed} km/h). "
            f"Use the WeatherSensorReadTool to retrieve and validate the reading. "
            f"Produce a structured weather report confirming sensor data quality. "
            f"Pass the confirmed temperature and wind speed to the next agent."
        ),
        expected_output=(
            "A structured weather report confirming: (1) corridor name, "
            "(2) ambient temperature in °C, (3) wind speed in km/h, "
            "(4) wind angle, (5) solar irradiance, and (6) sensor health status."
        ),
        agent=weather_agent,
    )

    task_thermal = Task(
        description=(
            f"Using the weather data from the previous task "
            f"(ambient_temp={ambient_temp}°C, wind_speed={wind_speed} km/h), "
            f"invoke the IEEE738DLRCalculatorTool to compute the real-time "
            f"dynamic ampacity of the ACSR Twin Moose conductor. "
            f"Document all four heat balance components (Q_c, Q_r, Q_s, R(T_c)). "
            f"State the final recommended max_i_ka value in kA with justification."
        ),
        expected_output=(
            "A complete IEEE 738 thermal analysis report containing: "
            "(1) all heat balance values in W/m, "
            "(2) conductor resistance at 85°C, "
            "(3) solved ampacity in both Amperes and kA, "
            "(4) percentage uplift over static rating, "
            "(5) explicit recommendation of the new max_i_ka value."
        ),
        agent=thermal_agent,
        context=[task_weather],
    )

    task_dispatch = Task(
        description=(
            "Read the recommended max_i_ka value from the thermal engineer's report. "
            "Invoke the GridDLRDispatchTool with that ampacity value to: "
            "(1) execute the line rating update on the Kahalgaon–Bihar Sharif 400 kV model, "
            "(2) compare line loading under static vs. dynamic rating, "
            "(3) compute MW of wind curtailment avoided, "
            "(4) calculate ₹ Lakh of wheeling revenue recovered. "
            "Confirm safety compliance and log the control action outcome."
        ),
        expected_output=(
            "A dispatch control record stating: "
            "(1) old vs. new thermal ceiling, "
            "(2) static line loading % vs. DLR line loading %, "
            "(3) wind MW recovered, "
            "(4) ₹ Lakh wheeling revenue saved, "
            "(5) safety compliance verdict (pass/fail), "
            "(6) SCADA commit status."
        ),
        agent=dispatcher_agent,
        context=[task_thermal],
    )

    return [task_weather, task_thermal, task_dispatch]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dlr_crew(
    ambient_temp: float,
    wind_speed: float,
) -> dict:
    """
    Execute the full three-agent DLR optimisation pipeline and return
    a structured result dictionary for the Streamlit dashboard.

    Parameters
    ----------
    ambient_temp : float
        Ambient corridor temperature [°C].
    wind_speed : float
        Perpendicular wind speed [km/h].

    Returns
    -------
    dict
        Keys:
          - crew_output : str — full agent narrative reasoning log
          - loading_static : float — line loading % at 500A static rating
          - loading_dlr : float — line loading % at DLR rating
          - ampacity_ka : float — IEEE 738 dynamic ampacity [kA]
          - curtailed_mw : float — wind MW curtailment prevented
          - revenue_lakh : float — ₹ wheeling revenue saved
          - is_safe : bool — DLR grid safety verification
    """
    logger.info(
        "Starting DLR CrewAI pipeline | T_a=%.1f°C | V=%.1f km/h",
        ambient_temp, wind_speed,
    )

    # Resolve LLM
    llm = _get_llm()

    # Build agents
    weather_agent, thermal_agent, dispatcher_agent = _build_agents(
        ambient_temp, wind_speed, llm
    )

    # Build tasks
    tasks = _build_tasks(
        weather_agent, thermal_agent, dispatcher_agent,
        ambient_temp, wind_speed,
    )

    # Assemble crew (sequential process)
    crew = Crew(
        agents=[weather_agent, thermal_agent, dispatcher_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,   # Stateless for prototype
    )

    # Execute
    crew_result = crew.kickoff()
    crew_output_text = str(crew_result)

    # --- Compute metrics independently (deterministic, LLM-agnostic) ---
    from physics_dlr import calculate_dynamic_ampacity, get_physics_breakdown
    from grid_engine import (
        create_Kahalgaon_BiharSharif_network,
        run_base_load_flow,
        update_line_rating_tool,
        estimate_curtailment_mw,
    )

    ampacity_ka = calculate_dynamic_ampacity(ambient_temp, wind_speed)

    net_static = create_Kahalgaon_BiharSharif_network()
    loading_static = run_base_load_flow(net_static)

    net_dlr = create_Kahalgaon_BiharSharif_network()
    loading_dlr, is_safe = update_line_rating_tool(net_dlr, ampacity_ka)

    curtailed_mw = estimate_curtailment_mw(loading_static, loading_dlr)
    revenue_lakh = (curtailed_mw * 1000 * 4.5) / 1e5

    return {
        "crew_output": crew_output_text,
        "loading_static": round(loading_static, 2),
        "loading_dlr": round(loading_dlr, 2),
        "ampacity_ka": round(ampacity_ka, 4),
        "curtailed_mw": round(curtailed_mw, 2),
        "revenue_lakh": round(revenue_lakh, 3),
        "is_safe": is_safe,
    }
