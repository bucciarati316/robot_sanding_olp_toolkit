"""
main_app/__init__.py - Main Application Package

提供 SimulationState 和 Worker 线程类。
通过 `main_app.py` 中的 SimulationApp 类使用。
"""

from __future__ import annotations

# Re-export SimulationState from the facade
from .main_app import SimulationState

__all__ = [
    "SimulationState",
]
