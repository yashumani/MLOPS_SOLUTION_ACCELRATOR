#!/usr/bin/env python3
"""Compatibility entrypoint for the governed qualification matrix runner."""

from __future__ import annotations

import sys

from batch_submit_all import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
