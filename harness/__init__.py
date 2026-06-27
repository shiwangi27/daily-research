"""daily-research training harness.

A small, config-driven engine for fast CPU-scale fine-tuning experiments.
The point is to change one knob at a time and measure, so the loop is plain
PyTorch (no Trainer magic) to keep optimizer / scheduler / loss fully exposed.
"""

__version__ = "0.1.0"
