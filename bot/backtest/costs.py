"""Trading costs. One simple model for every strategy, so results are comparable.

Interface stubs for Phase 3. See docs/design/backtest.md.

A cost is charged on every fill, buys and sells alike:

    cost = fill notional * bps_per_side / 10_000

where fill notional is shares times fill price. The default is 5 bps per side
(config.DEFAULT_COST_BPS_PER_SIDE). Every report also shows 0 and 20 bps
(config.COST_SENSITIVITY_BPS). The cost covers commission, spread and a little
slippage in one number. Dividends and the end-of-test valuation are not fills
for this purpose, except that positions still open at the end are valued net
of one sell cost (see engine.close_out_at_end).
"""

from dataclasses import dataclass

from bot.config import COST_SENSITIVITY_BPS, DEFAULT_COST_BPS_PER_SIDE

MAX_COST_BPS = 1_000.0


@dataclass(frozen=True)
class CostModel:
    """Cost in basis points per side. 5.0 means 0.05% of each fill's value."""

    bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE


def make_cost_model(bps_per_side: float) -> CostModel:
    """A CostModel, after checking 0 <= bps_per_side <= MAX_COST_BPS. Else raises ValueError."""
    if not 0.0 <= bps_per_side <= MAX_COST_BPS:
        raise ValueError(f"bps_per_side must be between 0 and {MAX_COST_BPS}, got {bps_per_side}.")
    return CostModel(bps_per_side=bps_per_side)


def fill_cost(notional: float, costs: CostModel) -> float:
    """The cost of one fill: abs(notional) * bps_per_side / 10_000."""
    return abs(notional) * costs.bps_per_side / 10_000.0


def affordable_notional(target_notional: float, cash: float, costs: CostModel) -> float:
    """The largest buy notional up to ``target_notional`` whose notional plus cost fits in ``cash``.

    ``min(target_notional, cash / (1 + bps / 10_000))``, and 0.0 when cash is not positive.
    No leverage: cash never goes below zero.
    """
    if cash <= 0.0:
        return 0.0
    return min(target_notional, cash / (1.0 + costs.bps_per_side / 10_000.0))


def sensitivity_cost_models() -> tuple[CostModel, ...]:
    """The cost levels every report shows (0, 5 and 20 bps by default)."""
    return tuple(CostModel(bps_per_side=bps) for bps in COST_SENSITIVITY_BPS)
