# SPDX-License-Identifier: GPL-3.0-only
"""Minimal entrypoint and temporal decoder for the independent Stage 3 program."""

from __future__ import annotations

import numpy as np


ACCEL_CLASSES = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_CLASSES = ("LEFT", "STRAIGHT", "RIGHT")
VIDEO_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".hevc", ".webm", ".ts",
    ".3gp", ".3gpp", ".wmv",
}


def _viterbi(log_probability: np.ndarray, transition_penalty: float) -> np.ndarray:
    """Return the most likely label sequence under a fixed switch penalty."""
    n, classes = log_probability.shape
    if not n:
        return np.empty(0, np.int64)
    transitions = np.full((classes, classes), -transition_penalty)
    np.fill_diagonal(transitions, 0.0)
    score = log_probability[0].copy()
    pointers = np.empty((n, classes), np.int16)
    for index in range(1, n):
        values = score[:, None] + transitions
        pointers[index] = np.argmax(values, axis=0)
        score = np.max(values, axis=0) + log_probability[index]
    states = np.empty(n, np.int64)
    states[-1] = int(np.argmax(score))
    for index in range(n - 1, 0, -1):
        states[index - 1] = pointers[index, states[index]]
    return states


def predict(data_dir, model_dir):
    """Delegate to the current frozen multi-rate motion pipeline."""
    from .motion_v2 import predict as current_predict

    return current_predict(data_dir, model_dir)
