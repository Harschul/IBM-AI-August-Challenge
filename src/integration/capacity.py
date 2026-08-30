"""Capacity reservation for temporal contacts.

The baseline already checks `residual_capacity_bytes`; this ledger supplies the
missing mutation/commit step so independently routed bundles cannot all consume
the same window at full capacity.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import Route


def contact_key(contact: Contact) -> tuple[int, int, float, float, str]:
    return (
        int(contact.source_id),
        int(contact.destination_id),
        float(contact.start_s),
        float(contact.end_s),
        str(contact.link_type),
    )


class CapacityLedger:
    def __init__(self, plan: ContactPlan):
        self._base = plan
        self._remaining = {
            contact_key(c): int(c.residual_capacity_bytes) for c in plan.contacts
        }

    def remaining(self, contact: Contact) -> int:
        return self._remaining[contact_key(contact)]

    def planning_plan(self) -> ContactPlan:
        """Fresh immutable-ish view with current residual capacities."""
        return ContactPlan(
            [
                replace(c, residual_capacity_bytes=self._remaining[contact_key(c)])
                for c in self._base.contacts
            ]
        )

    def reserve_contact(self, contact: Contact, size_bytes: int) -> None:
        key = contact_key(contact)
        remaining = self._remaining.get(key)
        if remaining is None:
            raise KeyError(f"contact not tracked by ledger: {key}")
        if size_bytes > remaining:
            raise ValueError(
                f"contact capacity exceeded for {key}: need {size_bytes}, have {remaining}"
            )
        self._remaining[key] = remaining - int(size_bytes)

    def reserve_route(self, route: Route, size_bytes: int) -> None:
        # Check first, then commit atomically from the caller's perspective.
        for c in route.hops:
            if size_bytes > self.remaining(c):
                raise ValueError("route no longer has enough residual capacity")
        for c in route.hops:
            self.reserve_contact(c, size_bytes)

    def utilization(self) -> float:
        initial = sum(int(c.residual_capacity_bytes) for c in self._base.contacts)
        remaining = sum(self._remaining.values())
        return 0.0 if initial == 0 else 1.0 - remaining / initial
