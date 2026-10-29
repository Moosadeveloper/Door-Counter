"""
geometry.py
-----------
The "door line": a straight line the user draws over the first video frame.

signed_distance(x, y) is POSITIVE on the inside side and NEGATIVE outside.
A person who goes from negative -> positive entered; positive -> negative left.
"""
import math
from typing import Tuple


class DoorLine:
    def __init__(self, line: Tuple[float, float, float, float], width: int, height: int,
                 flip: bool = False, zone_frac: float = 0.012):
        x1, y1, x2, y2 = line
        self.p1 = (x1 * width, y1 * height)
        self.p2 = (x2 * width, y2 * height)
        self.dx = self.p2[0] - self.p1[0]
        self.dy = self.p2[1] - self.p1[1]
        self.length = math.hypot(self.dx, self.dy)
        if self.length < 5:
            raise ValueError("Door line is too short - draw a longer line.")
        self.sign = -1 if flip else 1
        # half-width of the "ignore" band around the line (jitter protection)
        self.zone = max(4.0, zone_frac * max(width, height))

    def locate(self, x: float, y: float) -> Tuple[float, float]:
        """Return (d, t): d = signed pixel distance from the line (+ = inside),
        t = position along the line (0 at first point, 1 at second point)."""
        cross = self.dx * (y - self.p1[1]) - self.dy * (x - self.p1[0])
        d = self.sign * cross / self.length
        t = ((x - self.p1[0]) * self.dx + (y - self.p1[1]) * self.dy) / (self.length ** 2)
        return d, t

    def inside_normal(self) -> Tuple[float, float]:
        """Unit vector pointing toward the inside (used to draw the IN arrow)."""
        return (self.sign * -self.dy / self.length, self.sign * self.dx / self.length)
