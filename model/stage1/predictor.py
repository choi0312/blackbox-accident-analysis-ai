"""Frozen Qwen-only screen-recapture classification for every input video.

No DCT model, guard, reference labels, downloads, online fitting, or state shared
between videos. Uniform frames cover the decoded duration; not every frame is
fed to the visual encoder. Each frame contributes a full view and center detail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mpg',
                    '.mpeg', '.3gp', '.3gpp', '.wmv', '.hevc', '.ts'}
LABELS = ('ORIGINAL', 'RERECORDED')
QUESTION = (
    'Determine the acquisition process of this entire input video, not whether '
    'the depicted scene is real or computer-generated. The image pairs are '
    'chronological samples distributed across the SAME video from beginning to '
    'end. Each pair contains a full frame followed by its native-resolution '
    'center detail. Judge the video jointly using all these samples. '
    'A screen recapture is a camera recording of content displayed on an '
    'electronic screen. Look for a photographed screen border, reflected '
    'surroundings, a display pixel lattice, or coherent display-camera moire. '
    'Ordinary scene textures, dashcam text overlays, motion blur, resizing and '
    'JPEG artifacts alone do not establish screen recapture. '
    'Which acquisition process is better supported by the visual evidence?'
)


def _device():
    requested = os.environ.get('BLACKBOX_DEVICE', 'auto')
    if requested != 'auto':
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _model_path(model_dir):
    root = Path(model_dir)
    for candidate in (root / 'stage1', root / 'model/stage1', root):
        if (candidate / 'config.json').is_file():
            return candidate
    raise FileNotFoundError('Stage 1 config.json not found')


def _video_paths(data_dir):
    root = Path(data_dir)
    for folder in (root / 'stage1/videos', root / 'videos', root / 'stage1', root):
        if folder.is_dir():
            paths = sorted(p for p in folder.rglob('*')
                           if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
            if paths:
                if len({p.stem for p in paths}) != len(paths):
                    raise ValueError('Duplicate Stage 1 video IDs')
                return paths
    return []


def sample_views(path, count=12, full_side=640, crop_side=384):
    """Two streaming passes avoid trusting frame-count/FPS metadata or seeking.

    All decodable frames are counted, then at most count uniform frames are
    retained as bounded PIL views. Peak storage is independent of video length.
    """
    if count < 1 or full_side < 1 or crop_side < 1:
        raise ValueError('View limits must be positive')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {Path(path).name}')
    total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or not frame.size:
                raise RuntimeError('Invalid decoded video frame')
            total += 1
    finally:
        cap.release()
    if total == 0:
        raise RuntimeError(f'Empty video: {Path(path).name}')
    indices = np.unique(np.linspace(0, total - 1, min(count, total)).round().astype(int))
    wanted = set(indices.tolist())
    cap = cv2.VideoCapture(str(path))
    views = []
    try:
        for index in range(total):
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError('Video decode changed between sampling passes')
            if index not in wanted:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = image.size
            side = min(crop_side, w, h)
            left, top = (w - side) // 2, (h - side) // 2
            detail = image.crop((left, top, left + side, top + side))
            image.thumbnail((full_side, full_side), Image.Resampling.BILINEAR)
            views.extend((image, detail))
    finally:
        cap.release()
    if len(views) != 2 * len(indices):
        raise RuntimeError('Incomplete video sampling')
    return views


class RecapturePredictor:
    """Video-level recapture scorer with a frozen Qwen backend and fixed config."""

    def __init__(self, model_dir):
        root = _model_path(model_dir)
        self.config = json.loads((root / 'config.json').read_text())
        if self.config.get('backend') != 'qwen_only':
            raise ValueError('This predictor requires qwen_only configuration')
        self.device = _device()
        from ..mllm import FrozenMLLM
        self.model = FrozenMLLM(root / self.config['model_path'], self.device)

    @torch.inference_mode()
    def score_video(self, path):
        """Return the uncalibrated recapture score for one independent video."""
        views = sample_views(path, int(self.config['sample_frames']),
                             int(self.config['full_view_max_side']),
                             int(self.config['center_crop_max_side']))
        # binary() uses ALL supplied views; recapture() would reduce to 3 frames.
        result = self.model.binary(
            views, QUESTION,
            'A direct camera capture, not photographed from an electronic display.',
            'A screen recapture, photographed from an electronic display.')
        score = float(result['score'])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError('Invalid Qwen classification score')
        return score


def predict(data_dir, model_dir):
    """Classify every discovered video and return the expected Stage 1 schema."""
    paths = _video_paths(data_dir)
    if not paths:
        return pd.DataFrame(columns=['ID', 'answer'])
    predictor = RecapturePredictor(model_dir)
    threshold = float(predictor.config['threshold'])
    rows = []
    for path in paths:
        score = predictor.score_video(path)
        rows.append({'ID': path.stem, 'answer': LABELS[int(score >= threshold)]})
    return pd.DataFrame(rows, columns=['ID', 'answer'])
