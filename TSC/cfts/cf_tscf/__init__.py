"""
TSCF: Time Series CounterFactuals

This module implements the TSCF algorithm for generating counterfactual explanations
using gradient-based optimization with temporal smoothness constraints.
"""

from .tscf import cf_ts #tscf_cf, tscf_batch_cf

__all__ = ['cf_ts']#['tscf_cf', 'tscf_batch_cf']
