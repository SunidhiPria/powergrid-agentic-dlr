"""
grid_engine.py
==============
Pandapower physical network model for the POWERGRID 400 kV
Kahalgaon–Bihar Sharif transmission corridor.

Network topology
----------------
  [Bihar Sharif Wind Hub (Bus 0)]  ←——— 95 km ACSR Twin Moose ———→  [Kahalgaon Demand Hub (Bus 1)]
        |                                                                    |
  Wind Farm sgen                                                     External Grid (slack)
   450 MW injection                                                  (National Grid)

The model uses pandapower's Newton-Raphson power flow solver.
All impedance values are derived from ACSR Twin Moose datasheet.

Notes
-----
- Voltage base: 400 kV
- Apparent power base: 100 MVA (pandapower default)
- All per-unit quantities are computed internally by pandapower.
"""

import logging
from typing import Tuple

import numpy as np
import pandapower as pp
import pandapower.networks  # ensures optional submodule is loaded

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ACSR Twin Moose 400 kV line parameters (per-unit-length, SI)
# ---------------------------------------------------------------------------

# Positive sequence resistance [Ω/km] — twin-bundle (two conductors in parallel)
LINE_R_OHM_PER_KM: float = 0.0272

# Positive sequence reactance [Ω/km] — twin-bundle
LINE_X_OHM_PER_KM: float = 0.3120

# Positive sequence susceptance [μS/km] — twin-bundle (charging)
LINE_B_US_PER_KM: float = 3.16

# Line length [km]
LINE_LENGTH_KM: float = 210.0

# Nominal voltage [kV]
VNOM_KV: float = 400.0

# Static thermal rating [kA] — 500 A base case (no wind, 42°C)
STATIC_RATING_KA: float = 0.5

# Wind farm active power injection [MW]
WIND_FARM_MW: float = 500.0

# Wind farm reactive power (near unity power factor, slight capacitive)
WIND_FARM_MVAR: float = -25.0


# ---------------------------------------------------------------------------
# Network factory
# ---------------------------------------------------------------------------

def create_Kahalgaon_BiharSharif_network() -> pp.pandapowerNet:
    """
    Build and return a fresh pandapower network representing the
    Kahalgaon–Bihar Sharif 400 kV corridor.

    The network is created from scratch each call to ensure a clean
    simulation state, preventing result carry-over between agent runs.

    Returns
    -------
    pp.pandapowerNet
        Configured pandapower network ready for power-flow execution.
    """
    net = pp.create_empty_network(name="POWERGRID_Kahalgaon_BiharSharif_400kV")

    # --- Buses ---
    bus_BiharSharif = pp.create_bus(
        net,
        vn_kv=VNOM_KV,
        name="Kahalgaon Generation Hub",
        geodata=(8.8, 77.9),   # approximate GPS: Bihar Sharif, Bihar
    )

    bus_Kahalgaon = pp.create_bus(
        net,
        vn_kv=VNOM_KV,
        name="Bihar Sharif Load Hub",
        geodata=(9.9, 78.1),   # approximate GPS: Kahalgaon, Bihar
    )

    # --- External grid (slack / infinite bus) at Kahalgaon ---
    # Represents the National Grid interconnection point.
    pp.create_ext_grid(
        net,
        bus=bus_Kahalgaon,
        vm_pu=1.02,        # target voltage 1.02 pu (408 kV)
        va_degree=0.0,     # reference angle = 0°
        name="National Grid Slack",
    )

    # --- Wind farm static generator at Bihar Sharif ---
    # sgen models renewable generation (positive P injection into bus).
    pp.create_sgen(
        net,
        bus=bus_BiharSharif,
        p_mw=WIND_FARM_MW,
        q_mvar=WIND_FARM_MVAR,
        name="Eastern Renewable Energy Hub",
        type="renewable",
    )

    # --- Transmission line ---
    # Parameters: Twin Moose ACSR bundle, 95 km, initial static rating 500 A.
    pp.create_line_from_parameters(
        net,
        from_bus=bus_BiharSharif,
        to_bus=bus_Kahalgaon,
        length_km=LINE_LENGTH_KM,
        r_ohm_per_km=LINE_R_OHM_PER_KM,
        x_ohm_per_km=LINE_X_OHM_PER_KM,
        c_nf_per_km=LINE_B_US_PER_KM * 1e3 / (2 * np.pi * 50),  # convert B→C
        max_i_ka=STATIC_RATING_KA,
        name="Kahalgaon–Bihar Sharif 400kV ACSR Twin Moose",
    )

    logger.info(
        "Network '%s' created: 2 buses, 1 line (%.0f km), "
        "wind farm %.0f MW, static rating %.0f A.",
        net.name, LINE_LENGTH_KM, WIND_FARM_MW, STATIC_RATING_KA * 1000,
    )
    return net


# ---------------------------------------------------------------------------
# Power-flow execution helpers
# ---------------------------------------------------------------------------

def run_base_load_flow(net: pp.pandapowerNet) -> float:
    """
    Execute Newton-Raphson power flow under static line rating conditions
    and return the Kahalgaon–Bihar Sharif line loading percentage.

    Line loading % = (|I_actual| / I_max_rated) × 100

    Values > 100% indicate a thermal constraint violation.

    Parameters
    ----------
    net : pp.pandapowerNet
        Configured pandapower network (from `create_Kahalgaon_BiharSharif_network`).

    Returns
    -------
    float
        Line loading percentage [%]. E.g. 110.5 means 10.5% overloaded.

    Raises
    ------
    RuntimeError
        If the power-flow solver diverges (non-convergent case).
    """
    try:
        pp.runpp(
            net,
            algorithm="nr",          # Newton-Raphson
            max_iteration=50,
            tolerance_mva=1e-6,
            check_connectivity=True,
            numba=False,             # Numba JIT disabled for portability
        )
    except Exception as exc:
        raise RuntimeError(
            f"Power-flow (base case) did not converge: {exc}"
        ) from exc

    loading_pct: float = float(net.res_line["loading_percent"].iloc[0])
    logger.info("Base load-flow: line loading = %.2f %%", loading_pct)
    return loading_pct


def update_line_rating_tool(
    net: pp.pandapowerNet,
    new_ampacity_ka: float,
) -> Tuple[float, bool]:
    """
    Dynamically update the transmission line thermal rating and re-run
    the power flow to verify operational stability.

    This is the primary control action executed by the Grid Dispatcher Agent.
    It modifies `max_i_ka` on the pandapower line object in-place and
    re-solves the network equations.

    Parameters
    ----------
    net : pp.pandapowerNet
        The active network object (modified in-place).
    new_ampacity_ka : float
        New thermal limit in kilo-Amperes [kA], as calculated by the
        IEEE 738 physics engine (physics_dlr.calculate_dynamic_ampacity).

    Returns
    -------
    Tuple[float, bool]
        - loading_pct: Updated line loading percentage after DLR upgrade.
        - is_safe: True if loading_pct ≤ 100% (no thermal violation).

    Notes
    -----
    Safety is determined purely by the IEEE 738 ampacity — if the DLR
    engine says 620 A is safe, the dispatcher can raise the limit to 0.62 kA.
    """
    # Validate range: DLR ampacity must be between 0.5 kA (static floor)
    # and 0.7 kA (maximum 40% surge allowance).
    new_ampacity_ka = float(np.clip(new_ampacity_ka, 0.5, 0.7))

    # --- Update line rating in pandapower dataframe ---
    net.line.loc[0, "max_i_ka"] = new_ampacity_ka
    logger.info(
        "Line rating updated → %.3f kA (%.0f A)",
        new_ampacity_ka, new_ampacity_ka * 1000,
    )

    # --- Re-run power flow with new rating ---
    try:
        pp.runpp(
            net,
            algorithm="nr",
            max_iteration=50,
            tolerance_mva=1e-6,
            check_connectivity=True,
            numba=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Power-flow (DLR updated case) did not converge: {exc}"
        ) from exc

    loading_pct: float = float(net.res_line["loading_percent"].iloc[0])
    is_safe: bool = loading_pct <= 100.0

    logger.info(
        "DLR load-flow: line loading = %.2f %%  |  Safe = %s",
        loading_pct, is_safe,
    )
    return loading_pct, is_safe


def get_line_results(net: pp.pandapowerNet) -> dict:
    """
    Extract detailed line result metrics after a solved power flow.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network with a converged power-flow solution (`net.res_line` populated).

    Returns
    -------
    dict
        Detailed line metrics: current [kA], loading [%], active/reactive
        power flows from both ends, and maximum rated current [kA].
    """
    if net.res_line.empty:
        return {"error": "No converged power-flow results available."}

    row = net.res_line.iloc[0]
    return {
        "i_from_ka": round(float(row["i_from_ka"]), 4),
        "i_to_ka": round(float(row["i_to_ka"]), 4),
        "loading_percent": round(float(row["loading_percent"]), 2),
        "p_from_mw": round(float(row["p_from_mw"]), 2),
        "q_from_mvar": round(float(row["q_from_mvar"]), 2),
        "p_to_mw": round(float(row["p_to_mw"]), 2),
        "q_to_mvar": round(float(row["q_to_mvar"]), 2),
        "pl_mw": round(float(row["pl_mw"]), 2),   # active power loss
        "max_i_ka": round(float(net.line["max_i_ka"].iloc[0]), 4),
    }


def estimate_curtailment_mw(
    loading_pct_static: float,
    loading_pct_dlr: float,
    wind_farm_mw: float = WIND_FARM_MW,
) -> float:
    """
    Estimate how many MW of wind power are prevented from curtailment
    by upgrading from static to dynamic line rating.

    When the static rating causes loading > 100%, the grid operator must
    curtail wind generation. DLR raises the ceiling, recovering that energy.

    Parameters
    ----------
    loading_pct_static : float
        Line loading under static 500 A rating [%].
    loading_pct_dlr : float
        Line loading under dynamic (IEEE 738) rating [%].
    wind_farm_mw : float
        Total installed wind farm capacity [MW].

    Returns
    -------
    float
        Estimated MW recovered (prevented curtailment) [MW].
    """
    if loading_pct_static <= 100.0:
        return 0.0   # No curtailment was happening

    # Fraction of generation that was over-constrained
    overload_fraction = (loading_pct_static - 100.0) / loading_pct_static
    recovered_mw = overload_fraction * wind_farm_mw

    # If DLR also overloads, recovery is partial
    if loading_pct_dlr > 100.0:
        remaining_overload = (loading_pct_dlr - 100.0) / loading_pct_dlr
        recovered_mw -= remaining_overload * wind_farm_mw
        recovered_mw = max(0.0, recovered_mw)

    return round(recovered_mw, 2)
