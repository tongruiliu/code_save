#!/usr/bin/env python3
"""Main entrypoint for stage-2 rollout collection pipeline."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from data_pipeline.rollouts.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
