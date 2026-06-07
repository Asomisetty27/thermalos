"""
E-LT (lead-time testbed) thermal simulation for Theta.

Answers the make-or-break question: does R_theta_eff rise detectably BEFORE
thermal throttling, and by how much lead time, per degradation mode?

Physics: 3-node Cauer RC thermal network, calibrated to Stage 1 Tesla T4 data,
integrated with a stiff ODE solver. Detector: the same steady-state-window +
baseline+k-sigma rule the OSS agent ships.
"""

from . import params, thermal_model, degradation, detector, experiment

__all__ = ["params", "thermal_model", "degradation", "detector", "experiment"]
