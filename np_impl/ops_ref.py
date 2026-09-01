"""Reference functions for cpp/test/compare.py to diff the C++ op ports
against. Thin wrappers, not implementations — see docs/roadmap.md Phase 5.4.
"""
import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B
