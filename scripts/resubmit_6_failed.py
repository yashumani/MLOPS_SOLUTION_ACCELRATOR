#!/usr/bin/env python3
"""Retired historical replay entrypoint."""

from __future__ import annotations

import sys


MESSAGE = (
    "The historical six-config replay set is retired. Use "
    "scripts/batch_submit_all.py --scenario <scenario-id> --execute for an "
    "explicit immutable qualification revision."
)


if __name__ == "__main__":
    print(MESSAGE, file=sys.stderr)
    raise SystemExit(2)
