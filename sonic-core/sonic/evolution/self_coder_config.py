"""Validated environment configuration for the opt-in background self-coder."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SelfCoderConfig:
    enabled: bool
    every_ticks: int
    interval_seconds: int


def get_self_coder_config(environ: Mapping[str, str] | None = None) -> SelfCoderConfig:
    values = os.environ if environ is None else environ
    raw_enabled = values.get("SONIC_SELF_CODER_ENABLED", "true").strip().lower()
    bool_values = {
        "1": True, "true": True, "yes": True, "on": True,
        "0": False, "false": False, "no": False, "off": False,
    }
    if raw_enabled not in bool_values:
        raise ValueError(
            "SONIC_SELF_CODER_ENABLED must be one of: true/false, 1/0, yes/no, on/off"
        )

    raw_interval = values.get("SONIC_SELF_CODER_EVERY_TICKS", "20").strip()
    try:
        every_ticks = int(raw_interval)
    except ValueError as exc:
        raise ValueError("SONIC_SELF_CODER_EVERY_TICKS must be an integer from 1 to 10000") from exc
    if not 1 <= every_ticks <= 10000:
        raise ValueError("SONIC_SELF_CODER_EVERY_TICKS must be an integer from 1 to 10000")
    raw_interval_seconds = values.get("SONIC_SELF_CODER_INTERVAL_SECONDS", "3600").strip()
    try:
        interval_seconds = int(raw_interval_seconds)
    except ValueError as exc:
        raise ValueError(
            "SONIC_SELF_CODER_INTERVAL_SECONDS must be an integer from 60 to 86400"
        ) from exc
    if not 60 <= interval_seconds <= 86400:
        raise ValueError(
            "SONIC_SELF_CODER_INTERVAL_SECONDS must be an integer from 60 to 86400"
        )
    return SelfCoderConfig(
        enabled=bool_values[raw_enabled],
        every_ticks=every_ticks,
        interval_seconds=interval_seconds,
    )
