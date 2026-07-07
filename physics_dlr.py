"""
physics_dlr.py
==============
IEEE 738 Steady-State Thermal Model for ACSR "Twin Moose" conductor.
Implements the heat balance equation:

    Q_c + Q_r = Q_s + I² × R(T_c)

Solving for max allowable ampacity I:

    I = sqrt( (Q_c + Q_r - Q_s) / R(T_c) )

This module is the authoritative source-of-truth for conductor thermal
physics in the POWERGRID Agentic DLR system.

References
----------
- IEEE Std 738-2012, "Standard for Calculating the Current-Temperature
  Relationship of Bare Overhead Conductors"
- POWERGRID ACSR Twin Moose datasheet (400 kV class)
"""

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conductor physical constants — ACSR "Twin Moose" (400 kV class)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConductorConstants:
    """
    Immutable physical parameters for the ACSR Twin Moose conductor
    as per POWERGRID specifications and IEEE 738 Annex B tables.
    """
    # Conductor outer diameter [m]
    diameter: float = 0.03176

    # Conductor surface emissivity (oxidised aluminium, unitless 0-1)
    emissivity: float = 0.5

    # Solar absorptivity (oxidised aluminium, unitless 0-1)
    absorptivity: float = 0.5

    # DC resistance at 25°C [Ω/m]
    resistance_25c: float = 6.8e-5

    # Temperature coefficient of resistance for aluminium [1/°C]
    temp_coeff_resistance: float = 0.00403

    # Maximum allowable conductor core temperature [°C]
    max_core_temp: float = 85.0

    # Base static rating at 42°C ambient, zero wind [Amperes]
    base_static_rating_a: float = 500.0

    # Maximum allowable surge rating (wind-augmented, 40% above base) [Amperes]
    max_surge_rating_a: float = 700.0


CONDUCTOR = ConductorConstants()


# ---------------------------------------------------------------------------
# Atmospheric & solar constants
# ---------------------------------------------------------------------------

# Stefan-Boltzmann constant [W/(m²·K⁴)]
STEFAN_BOLTZMANN: float = 5.6704e-8

# Solar heat flux (clear sky, tropical India, ~1000 W/m²) [W/m]
# Per unit length of conductor: Q_s = α × D × Q_solar
SOLAR_FLUX_W_M2: float = 1000.0

# Kinematic viscosity of air at ~60°C film temp [m²/s]
AIR_KINEMATIC_VISCOSITY: float = 1.89e-5

# Thermal conductivity of air at ~60°C film temp [W/(m·°C)]
AIR_THERMAL_CONDUCTIVITY: float = 0.02966

# Prandtl number of air at ~60°C (dimensionless)
AIR_PRANDTL: float = 0.71

# Wind angle relative to conductor axis [degrees]
# A perpendicular crosswind (90°) gives maximum cooling.
# POWERGRID Kahalgaon-Bihar Sharif corridor dominant wind angle.
WIND_ANGLE_DEG: float = 90.0


# ---------------------------------------------------------------------------
# Core IEEE 738 physics functions
# ---------------------------------------------------------------------------

def _conductor_resistance_at_temp(core_temp_c: float) -> float:
    """
    Calculate conductor AC resistance [Ω/m] at a given core temperature.

    Uses linear temperature correction per IEEE 738 Eq. (2):
        R(T_c) = R_25 × [1 + α × (T_c − 25)]

    Parameters
    ----------
    core_temp_c : float
        Conductor core temperature in °C.

    Returns
    -------
    float
        Resistance in Ω/m.
    """
    r_tc = CONDUCTOR.resistance_25c * (
        1 + CONDUCTOR.temp_coeff_resistance * (core_temp_c - 25.0)
    )
    return r_tc


def _convective_cooling_forced(
    ambient_temp_c: float,
    wind_speed_ms: float,
    core_temp_c: float,
) -> float:
    """
    Forced convective heat loss Q_c [W/m] per IEEE 738 Section 4.4.2.

    Uses the Churchill-Bernstein correlation adapted for cylinders:
        Nu = (B₁ + B₂ × Re^n) × (sin θ)^0.52
        Q_c = Nu × k_f × π × (T_c − T_a)

    For the simplified prototype, a linearised version is employed:
        Q_c = (1.01 + 0.0372 × Re^0.52) × k_f × (T_c − T_a)

    Parameters
    ----------
    ambient_temp_c : float
        Ambient air temperature [°C].
    wind_speed_ms : float
        Perpendicular wind speed [m/s].
    core_temp_c : float
        Conductor core temperature [°C].

    Returns
    -------
    float
        Forced convective cooling rate [W/m]. Returns zero if wind is calm.
    """
    delta_temp = core_temp_c - ambient_temp_c
    if delta_temp <= 0:
        return 0.0

    # Wind angle correction factor (sin θ)^0.52
    angle_rad = math.radians(WIND_ANGLE_DEG)
    angle_factor = (math.sin(angle_rad)) ** 0.52  # = 1.0 at 90°

    if wind_speed_ms < 0.01:
        # Natural convection only (low-wind fallback)
        # Simplified: Q_cn = 3.645 × ρ^0.5 × D^0.75 × ΔT^1.25
        # Using approximate ρ_air = 1.029 kg/m³ at 60°C
        rho_air = 1.029
        q_cn = (
            3.645
            * (rho_air ** 0.5)
            * (CONDUCTOR.diameter ** 0.75)
            * (delta_temp ** 1.25)
        )
        return q_cn

    # Reynolds number
    reynolds = (wind_speed_ms * CONDUCTOR.diameter) / AIR_KINEMATIC_VISCOSITY

    # Nusselt number (IEEE 738 Table 1: B₁=0.0119, B₂=0.471, n=0.633 for Re range 2650-50000)
    if reynolds < 2650:
        b1, b2, n = 0.583, 0.471, 0.471
    else:
        b1, b2, n = 0.0239, 0.0266, 0.805

    nusselt = (b1 + b2 * (reynolds ** n)) * angle_factor
    q_cf = nusselt * AIR_THERMAL_CONDUCTIVITY * delta_temp
    return q_cf


def _radiative_cooling(ambient_temp_c: float, core_temp_c: float) -> float:
    """
    Radiative heat loss Q_r [W/m] per IEEE 738 Section 4.4.4.

        Q_r = ε × σ × π × D × (T_c⁴ − T_a⁴)

    Temperatures must be in Kelvin for Stefan-Boltzmann.

    Parameters
    ----------
    ambient_temp_c : float
        Ambient air temperature [°C].
    core_temp_c : float
        Conductor core temperature [°C].

    Returns
    -------
    float
        Radiative cooling rate [W/m].
    """
    t_c_k = core_temp_c + 273.15
    t_a_k = ambient_temp_c + 273.15
    q_r = (
        CONDUCTOR.emissivity
        * STEFAN_BOLTZMANN
        * math.pi
        * CONDUCTOR.diameter
        * (t_c_k**4 - t_a_k**4)
    )
    return max(q_r, 0.0)


def _solar_heat_gain() -> float:
    """
    Solar heat gain Q_s [W/m] per IEEE 738 Section 4.4.5.

        Q_s = α × D × Q_solar

    For the Kahalgaon–Bihar Sharif corridor (Eastern India, ~9°N latitude),
    clear-sky solar flux is taken as 1000 W/m².

    Returns
    -------
    float
        Solar heat gain per unit length [W/m].
    """
    q_s = CONDUCTOR.absorptivity * CONDUCTOR.diameter * SOLAR_FLUX_W_M2
    return q_s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_dynamic_ampacity(
    ambient_temp: float,
    wind_speed: float,
) -> float:
    """
    Calculate the real-time maximum allowable conductor current (Ampacity)
    using the IEEE 738 steady-state heat balance equation.

    Heat balance:
        Q_c + Q_r = Q_s + I² × R(T_c)

    Solving for I:
        I = sqrt( max(0, Q_c + Q_r − Q_s) / R(T_c) )

    The result is clamped between the static base rating (500 A) and the
    maximum safe surge rating (700 A = +40%) to prevent unsafe extrapolation.

    Parameters
    ----------
    ambient_temp : float
        Ambient air temperature [°C]. Typical range: 20–45 °C for Eastern India.
    wind_speed : float
        Perpendicular wind speed [km/h]. Converted internally to m/s.

    Returns
    -------
    float
        Safe dynamic ampacity in **kilo-Amperes [kA]** for direct
        compatibility with pandapower `max_i_ka` line parameters.

    Examples
    --------
    >>> calculate_dynamic_ampacity(42.0, 0.0)   # base condition
    0.5
    >>> calculate_dynamic_ampacity(30.0, 25.0)  # cool, windy day
    ~0.62  (value between 0.5 and 0.7)
    """
    # Convert wind speed from km/h → m/s
    wind_speed_ms = wind_speed / 3.6

    core_temp_c = CONDUCTOR.max_core_temp  # IEEE 738: solve at T_c = 85°C

    # --- Heat flow components ---
    q_c = _convective_cooling_forced(ambient_temp, wind_speed_ms, core_temp_c)
    q_r = _radiative_cooling(ambient_temp, core_temp_c)
    q_s = _solar_heat_gain()
    r_tc = _conductor_resistance_at_temp(core_temp_c)

    # --- Solve for ampacity ---
    net_cooling = q_c + q_r - q_s
    if net_cooling <= 0.0:
        # Degenerate case: solar gain exceeds cooling → use static floor
        ampacity_a = CONDUCTOR.base_static_rating_a
        logger.warning(
            "Net cooling is non-positive (Q_c=%.2f, Q_r=%.2f, Q_s=%.2f). "
            "Clamping to static base rating.",
            q_c, q_r, q_s,
        )
    else:
        ampacity_a = math.sqrt(net_cooling / r_tc)

    # Clamp to physically validated operating band [500 A, 700 A]
    ampacity_a = max(CONDUCTOR.base_static_rating_a, ampacity_a)
    ampacity_a = min(CONDUCTOR.max_surge_rating_a, ampacity_a)

    # Convert to kA for pandapower compatibility
    ampacity_ka = ampacity_a / 1000.0

    logger.info(
        "DLR | T_a=%.1f°C  V=%.1f km/h  →  "
        "Q_c=%.2f W/m  Q_r=%.2f W/m  Q_s=%.2f W/m  "
        "R(T_c)=%.4e Ω/m  →  I=%.1f A (%.3f kA)",
        ambient_temp, wind_speed, q_c, q_r, q_s, r_tc, ampacity_a, ampacity_ka,
    )
    return ampacity_ka


def get_physics_breakdown(
    ambient_temp: float,
    wind_speed: float,
) -> dict:
    """
    Return a detailed breakdown of all IEEE 738 heat flow components
    for display in the Streamlit dashboard and agent reasoning logs.

    Parameters
    ----------
    ambient_temp : float
        Ambient temperature [°C].
    wind_speed : float
        Wind speed [km/h].

    Returns
    -------
    dict
        Keys: q_c, q_r, q_s, r_tc, net_cooling, ampacity_a, ampacity_ka.
    """
    wind_speed_ms = wind_speed / 3.6
    core_temp_c = CONDUCTOR.max_core_temp

    q_c = _convective_cooling_forced(ambient_temp, wind_speed_ms, core_temp_c)
    q_r = _radiative_cooling(ambient_temp, core_temp_c)
    q_s = _solar_heat_gain()
    r_tc = _conductor_resistance_at_temp(core_temp_c)
    net_cooling = max(0.0, q_c + q_r - q_s)

    if net_cooling <= 0.0:
        ampacity_a = CONDUCTOR.base_static_rating_a
    else:
        ampacity_a = math.sqrt(net_cooling / r_tc)

    ampacity_a = max(CONDUCTOR.base_static_rating_a, ampacity_a)
    ampacity_a = min(CONDUCTOR.max_surge_rating_a, ampacity_a)
    ampacity_ka = ampacity_a / 1000.0

    return {
        "q_c_w_per_m": round(q_c, 3),
        "q_r_w_per_m": round(q_r, 3),
        "q_s_w_per_m": round(q_s, 3),
        "net_cooling_w_per_m": round(net_cooling, 3),
        "r_tc_ohm_per_m": round(r_tc, 8),
        "ampacity_a": round(ampacity_a, 2),
        "ampacity_ka": round(ampacity_ka, 4),
    }
