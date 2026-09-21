#!/usr/bin/env python3
"""Compatibility entry point for calibrate/world_base.py."""

import os
import runpy


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(ROOT, "calibrate", "world_base.py"),
               run_name="__main__")
