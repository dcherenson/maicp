"""Resolve margins by class or robot while preserving individual robot values."""

from collections.abc import Mapping
import math

from maicp.configuration import ClassConfig, Platform


def resolve_robot_margins(
    margins: Mapping[Platform | str, float],
    platforms: Mapping[str, Platform],
    *,
    classes: Mapping[Platform, ClassConfig] | None = None,
) -> tuple[dict[str, float], dict[Platform, float] | None]:
    """Resolve robot and class margins. Check policy margins against the class bounds when classes are supplied."""
    if not isinstance(margins, Mapping) or not margins:
        raise ValueError("Tracking margins must be a nonempty mapping")
    class_margins: dict[Platform, float] = {}
    robot_margins: dict[str, float] = {}
    for key, value in margins.items():
        platform = next(
            (p for p in Platform if isinstance(key, str)
             and key.strip().lower() in {p.value, p.name.lower()}),
            None,
        )
        if isinstance(key, Platform):
            platform = key
        if platform is not None:
            class_margins[platform] = float(value)
        elif isinstance(key, str):
            robot_margins[key] = float(value)
        else:
            raise TypeError("Tracking margin keys must be classes or robot IDs")
    if class_margins and robot_margins:
        raise ValueError("Tracking margins cannot mix class and robot keys")
    if class_margins:
        if set(class_margins) != set(Platform):
            raise ValueError("Tracking class margins must contain both UAV and UGV")
        robot_margins = {robot: class_margins[p] for robot, p in platforms.items()}
    elif set(robot_margins) != set(platforms):
        raise ValueError(
            f"Missing robot margins: {sorted(set(platforms) - set(robot_margins))}; "
            f"unknown robot margins: {sorted(set(robot_margins) - set(platforms))}"
        )
    for robot, value in robot_margins.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Tracking margin for {robot} must be finite and nonnegative")
        if classes is not None:
            cls = classes[platforms[robot]]
            if not cls.margin_min <= value <= cls.margin_max:
                raise ValueError(f"Tracking margin for {robot} is outside its class operational box")
    return robot_margins, class_margins or None
