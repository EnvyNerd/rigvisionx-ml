"""ROI calculations for predictive maintenance"""


def calculate_roi(implementation_cost: float, annual_savings: float, years: int = 3) -> float:
    """
    Calculate ROI for predictive maintenance implementation.

    Args:
        implementation_cost: Cost to implement system
        annual_savings: Annual cost savings from avoiding failures
        years: Time period for ROI calculation

    Returns:
        ROI percentage
    """
    total_savings = annual_savings * years
    roi = ((total_savings - implementation_cost) / implementation_cost) * 100
    return roi


def calculate_break_even_period(implementation_cost: float, monthly_savings: float) -> float:
    """Calculate break-even period in months."""
    return implementation_cost / monthly_savings


def compare_scenarios(preventive_cost: float, failure_cost: float, failure_probability: float) -> dict:
    """Compare cost of preventive maintenance vs. risk of failure."""
    expected_failure_cost = failure_cost * failure_probability
    return {
        "preventive_cost": preventive_cost,
        "expected_failure_cost": expected_failure_cost,
        "recommendation": "Preventive" if preventive_cost < expected_failure_cost else "Monitor",
    }
