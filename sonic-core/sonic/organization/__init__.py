"""SONIC organizational architecture and employee ownership contracts."""

from sonic.organization.architecture import (
    DEPARTMENTS,
    SUBSYSTEM_OWNERSHIP,
    TEAMS,
    Department,
    Team,
    SubsystemOwnership,
    owner_for_path,
    ownership_for_path,
)

__all__ = [
    "DEPARTMENTS", "SUBSYSTEM_OWNERSHIP", "TEAMS",
    "Department", "Team", "SubsystemOwnership",
    "owner_for_path", "ownership_for_path",
]
