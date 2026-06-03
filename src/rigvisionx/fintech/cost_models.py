"""Cost modeling for maintenance decisions"""


def calculate_maintenance_cost(parts: list, labor_hours: float, labor_rate: float = 150) -> float:
    """Calculate total maintenance cost."""
    parts_cost = sum(part["cost"] * part["quantity"] for part in parts)
    labor_cost = labor_hours * labor_rate
    return parts_cost + labor_cost


def calculate_failure_cost(equipment_value: float, downtime_hours: float, revenue_per_hour: float) -> float:
    """Calculate cost of unexpected equipment failure."""
    downtime_cost = downtime_hours * revenue_per_hour
    replacement_cost = equipment_value
    return downtime_cost + replacement_cost


def calculate_preventive_maintenance_cost(maintenance_cost: float, maintenance_interval: int) -> float:
    """Calculate average cost per unit time for preventive maintenance."""
    return maintenance_cost / maintenance_interval
