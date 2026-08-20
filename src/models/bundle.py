"""
DataBundle = one unit of science data moving through the network (PDF section 7.4).

The key modelling idea: a bundle is the unit of MISSION VALUE, not just bytes.
Two files of identical size can be routed completely differently because one is
a rare transient with a 180 s deadline and the other is housekeeping data that
can wait six hours.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataBundle:
    bundle_id: str
    source_id: int
    size_bytes: int
    created_s: float = 0.0
    science_priority: float = 0.5        # 0.0 - 1.0
    deadline_s: Optional[float] = None   # absolute sim time it becomes worthless
    data_type: str = "STAR_FIELD"        # TRANSIENT | STAR_FIELD | CALIBRATION | HOUSEKEEPING
    destination_class: str = "ANY_GROUND"

    # mutable state as it travels
    remaining_bytes: int = 0
    current_holder: int = -1
    route_history: List[int] = field(default_factory=list)
    retries: int = 0
    status: str = "QUEUED"               # QUEUED|IN_TX|STORED|DELIVERED|EXPIRED|DROPPED

    def __post_init__(self):
        if not 0.0 <= self.science_priority <= 1.0:
            raise ValueError("science_priority must be in 0..1")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.remaining_bytes == 0:
            self.remaining_bytes = self.size_bytes
        if self.current_holder == -1:
            self.current_holder = self.source_id
        if not self.route_history:
            self.route_history = [self.source_id]

    def is_expired(self, now_s: float) -> bool:
        return self.deadline_s is not None and now_s > self.deadline_s
