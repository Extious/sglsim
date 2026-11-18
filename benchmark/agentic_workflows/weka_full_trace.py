"""Compatibility import for the AegisKV-owned Weka causal adapter.

The experiment implementation lives under ``AegisKV/experiments`` so the
three-layer campaign is self-contained.  Existing benchmark imports remain
stable through this module.
"""

from AegisKV.experiments.src.request.weka_causal import *  # noqa: F401,F403
from AegisKV.experiments.src.request.weka_causal import __all__
