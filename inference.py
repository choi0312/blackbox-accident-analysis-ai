"""Black-box accident analysis entry points with fully local model assets.

The server supplies script.py; this file only returns the three DataFrames.
No test-time training, cross-video calibration, or network access is used.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import sys
import types

# The evaluation server has no network. Setting these before importing a Stage
# also prevents an accidental Hub lookup from turning into a long timeout.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

_ROOT = Path(__file__).resolve().parent
_LOG = logging.getLogger(__name__)
_COLUMNS = {
    1: ["ID", "answer"],
    2: ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
    3: ["ID", "sample_index", "accel_label", "steer_label"],
}
_LABELS = {
    "answer": {"ORIGINAL", "RERECORDED"},
    "entry_side": {"LEFT", "RIGHT"},
    "accel_label": {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"},
    "steer_label": {"LEFT", "STRAIGHT", "RIGHT"},
}


def _data_folder(data_dir, stage):
    """Accept either server stage paths or a data root for local verification."""
    base = Path(data_dir).resolve()
    child = "images" if stage == 2 else "videos"
    for candidate in (base, base / f"stage{stage}"):
        if (candidate / child).is_dir():
            return candidate
    if base.name == child and base.is_dir():
        return base.parent
    raise FileNotFoundError(f"Stage {stage}: expected an existing {child}/ input directory")


def _model_folder(model_dir, stage):
    base = Path(model_dir).resolve()
    nested = base / f"stage{stage}"
    if nested.is_dir():
        return nested
    if base.is_dir():
        return base
    raise FileNotFoundError(f"Stage {stage}: model directory does not exist")


def _load_stage(stage):
    """Load one packaged predictor without depending on the current directory.

    A private namespace also avoids colliding with an unrelated installed
    package named ``model`` while preserving relative imports inside each Stage.
    """
    parent_name = "_blackbox_analysis_runtime"
    package_name = f"{parent_name}.stage{stage}"
    module_name = f"{package_name}.predictor"
    if module_name in sys.modules:
        return sys.modules[module_name]
    for name, folder in (
        (parent_name, _ROOT / "model"),
        (package_name, _ROOT / "model" / f"stage{stage}"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(folder)]
            module.__package__ = name
            sys.modules[name] = module
    source = _ROOT / "model" / f"stage{stage}" / "predictor.py"
    if not source.is_file():
        raise FileNotFoundError(f"Stage {stage}: packaged predictor.py missing")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Stage {stage}: cannot load packaged predictor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _validate(frame, stage):
    """Normalize a Stage result and fail fast on every invalid submission row."""
    import numpy as np
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Stage {stage}: predictor must return a pandas.DataFrame")
    missing = set(_COLUMNS[stage]) - set(frame.columns)
    if missing:
        raise ValueError(f"Stage {stage}: missing output columns {sorted(missing)}")
    frame = frame.loc[:, _COLUMNS[stage]].copy()
    if frame.empty:
        raise ValueError(f"Stage {stage}: no predictions generated")
    if frame.isna().any().any():
        raise ValueError(f"Stage {stage}: missing prediction value")
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].str.strip().eq("").any():
        raise ValueError(f"Stage {stage}: empty video identifier")
    # Stage 3 legitimately emits many rows per video; the other Stages emit one.
    keys = ["ID", "sample_index"] if stage == 3 else ["ID"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Stage {stage}: duplicate prediction keys")
    for column, labels in _LABELS.items():
        if column in frame and not frame[column].isin(labels).all():
            raise ValueError(f"Stage {stage}: unsupported {column} label")
    for column in ("collision_frame", "entry_frame", "sample_index", "evasion_space"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Stage {stage}: {column} must contain nonnegative integers")
        frame[column] = values.astype(np.int64)
    if stage == 2:
        if not frame["evasion_space"].isin([0, 1]).all():
            raise ValueError("Stage 2: evasion_space must contain only 0 or 1")
        if (frame["entry_frame"] > frame["collision_frame"]).any():
            raise ValueError("Stage 2: entry prediction is after collision")
    if stage == 3:
        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        for _, part in frame.groupby("ID", sort=False):
            # The evaluator expects exactly one ordered label for every 0.1 s sample.
            if not np.array_equal(part["sample_index"].to_numpy(), np.arange(len(part))):
                raise ValueError("Stage 3: sample indices must be contiguous from zero")
    else:
        frame = frame.sort_values("ID", kind="stable").reset_index(drop=True)
    return frame


def _predict(stage, data_dir, model_dir):
    """Run a single frozen Stage under a shared resource and validation policy."""
    import cv2
    import torch

    # Leave resources available to video decoding and respect the 7-vCPU server.
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    directory = _data_folder(data_dir, stage)
    models = _model_folder(model_dir, stage)
    predictor = _load_stage(stage)
    with torch.inference_mode():
        result = predictor.predict(str(directory), str(models))
    return _validate(result, stage)


def predict_stage1(data_dir, model_dir):
    return _predict(1, data_dir, model_dir)


def predict_stage2(data_dir, model_dir):
    return _predict(2, data_dir, model_dir)


def predict_stage3(data_dir, model_dir):
    return _predict(3, data_dir, model_dir)
