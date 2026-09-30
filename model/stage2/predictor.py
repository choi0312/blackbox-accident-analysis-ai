"""Offline Stage 2 inference: YOLOPv2 perception and temporal geometry.

Frozen YOLOPv2 geometry with a conservative dual-SimpleTAD disagreement guard.
Anomaly onset is only a fallback hypothesis, not a general contact detector.
All temporal state belongs to one input video. No model/download/training calls
are made during prediction. See the accompanying research note for limitations.
YOLOPv2 tensor decoding is adapted from CAIC-AD/YOLOPv2 (MIT; license included).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms


COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
MAX_PERCEPTION_FRAMES = 180
MAX_MOTION_FRAMES = 720


def _smooth(values, sigma=1.0):
    a = np.asarray(values, dtype=np.float64)
    return gaussian_filter1d(a, sigma=sigma, mode="nearest") if len(a) > 2 else a.copy()


def _robust_positive(values):
    """Bounded evidence with a noise floor; an almost constant trace stays quiet."""
    a = np.nan_to_num(np.asarray(values, dtype=np.float64))
    med = np.median(a)
    scale = max(1.4826 * np.median(np.abs(a - med)), 0.03 * np.max(np.abs(a)), 1e-5)
    return np.clip((a - med) / scale, 0.0, 12.0) / 12.0


def _enumerate_frames(folder):
    records = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not p.is_file():
            continue
        m = re.search(r"(\d+)$", p.stem)
        if m:
            records.append((int(m.group(1)), p))
    records.sort(key=lambda x: x[0])
    if not records:
        raise ValueError(f"No numbered image frames in {folder}")
    if len({x[0] for x in records}) != len(records):
        raise ValueError(f"Duplicate original frame numbers in {folder}")
    return records


def _read(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


@dataclass
class Perception:
    boxes: np.ndarray
    road: np.ndarray
    lanes: np.ndarray
    left: np.ndarray
    right: np.ndarray
    lane_reliable: bool


def _lane_boundaries(lanes):
    """Fit visible lane strokes; use explicit geometric prior when unobserved."""
    h, w = lanes.shape
    mask = lanes.copy().astype(np.uint8) * 255
    mask[:int(h * 0.35)] = 0
    mask[int(h * 0.95):] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 12,
                            minLineLength=max(10, h // 12), maxLineGap=12)
    choices = {"left": [], "right": []}
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dy = (y2 - y1) / h
            if abs(dy) < 0.05:
                continue
            a = ((x2 - x1) / w) / dy
            b = x1 / w - a * y1 / h
            xb = a * 0.9 + b
            xt = a * 0.45 + b
            if not (0.20 < xt < 0.80) or abs(a) > 2.5:
                continue
            length = math.hypot((x2 - x1) / w, dy)
            if -0.20 < xb < 0.48 and a < 0.2:
                choices["left"].append((a, b, length, xb))
            if 0.52 < xb < 1.20 and a > -0.2:
                choices["right"].append((a, b, length, xb))
    # Perspective corridor: from x=.45/.55 at y=.45 to x=.12/.88 at y=.9.
    left = np.array([-0.733333, 0.78], dtype=float)
    right = np.array([0.733333, 0.22], dtype=float)
    observed = 0
    for key in ("left", "right"):
        c = choices[key]
        if not c:
            continue
        c.sort(key=lambda z: z[3], reverse=(key == "left"))
        closest = c[0][3]
        c = np.array([z for z in c if abs(z[3] - closest) < 0.10])
        coeff = np.average(c[:, :2], axis=0, weights=c[:, 2])
        if key == "left":
            left = coeff
        else:
            right = coeff
        observed += 1
    return left, right, observed == 2


class RoadPerception:
    """Frozen YOLOPv2 adapter for boxes, drivable area and lane boundaries."""

    def __init__(self, model_dir):
        base = Path(model_dir)
        candidates = [base / "yolopv2.pt", base / "stage2/yolopv2.pt",
                      base / "model/stage2/yolopv2.pt"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Local YOLOPv2 checkpoint missing: {candidates}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(path), map_location="cpu").eval().to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    @torch.inference_mode()
    def __call__(self, image):
        h, w = image.shape[:2]
        r = min(640.0 / h, 640.0 / w)
        nw, nh = round(w * r), round(h * r)
        # Same minimal stride-32 letterbox as the author's inference implementation.
        dw, dh = (640 - nw) % 32, (640 - nh) % 32
        left, top = dw // 2, dh // 2
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        x = torch.from_numpy(rgb).to(device=self.device, dtype=self.dtype)[None] / 255.0
        detection, area, lane = self.model(x)
        raw, anchor_grids = detection
        decoded = []
        for i, stride in enumerate((8, 16, 32)):
            bs, _, ny, nx = raw[i].shape
            y = raw[i].reshape(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).sigmoid()
            gy, gx = torch.meshgrid(torch.arange(ny, device=self.device),
                                    torch.arange(nx, device=self.device), indexing="ij")
            grid = torch.stack((gx, gy), dim=-1).to(self.dtype)[None, None]
            centers = (y[..., :2] * 2.0 - 0.5 + grid) * stride
            sizes = (y[..., 2:4] * 2.0).square() * anchor_grids[i]
            decoded.append(torch.cat((centers - sizes / 2, centers + sizes / 2,
                                      y[..., 4:]), dim=-1).reshape(-1, 85))
        pred = torch.cat(decoded)
        # Vehicle classes use the official COCO indexing in the released 85-channel head.
        classes = torch.tensor([2, 3, 5, 7], device=self.device)
        class_scores, _ = pred[:, 5 + classes].max(dim=1)
        scores = pred[:, 4] * class_scores
        valid = scores > 0.14
        boxes, scores = pred[valid, :4].float(), scores[valid].float()
        if len(boxes):
            keep = nms(boxes, scores, 0.50)[:40]
            boxes, scores = boxes[keep], scores[keep]
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / nw).clamp(0, 1)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / nh).clamp(0, 1)
            boxes = torch.cat((boxes, scores[:, None]), dim=1).cpu().numpy()
            boxes = boxes[((boxes[:, 2] - boxes[:, 0]) > 0.015) &
                          ((boxes[:, 3] - boxes[:, 1]) > 0.015)]
        else:
            boxes = np.empty((0, 5), dtype=np.float32)

        # Crop the actual letterbox for any aspect ratio, instead of fixed 12:372.
        ah, aw = area.shape[-2:]
        ph, pw = padded.shape[:2]
        y0, y1 = round(top * ah / ph), round((top + nh) * ah / ph)
        x0, x1 = round(left * aw / pw), round((left + nw) * aw / pw)
        road = area[0, :, y0:y1, x0:x1].float().argmax(dim=0).cpu().numpy().astype(np.uint8)
        lh, lw = lane.shape[-2:]
        ly0, ly1 = round(top * lh / ph), round((top + nh) * lh / ph)
        lx0, lx1 = round(left * lw / pw), round((left + nw) * lw / pw)
        lanes = (lane[0, 0, ly0:ly1, lx0:lx1] > 0.5).cpu().numpy().astype(np.uint8)
        road = cv2.resize(road, (320, 180), interpolation=cv2.INTER_NEAREST) > 0
        lanes = cv2.resize(lanes, (320, 180), interpolation=cv2.INTER_NEAREST) > 0
        lcoef, rcoef, reliable = _lane_boundaries(lanes)
        return Perception(boxes, road, lanes, lcoef, rcoef, reliable)


def _motion_signals(records):
    """Estimate per-video ego-motion discontinuities with sparse LK flow.

    Forward/backward checks reject unstable tracks, and an affine RANSAC fit
    separates coherent camera motion from local object motion. The returned
    ``shake`` value is a robust impact cue, not a learned probability.
    """
    n = len(records)
    positions = np.unique(np.linspace(0, n - 1, min(n, MAX_MOTION_FRAMES)).round().astype(int))
    signals = np.zeros((len(positions), 7), dtype=np.float64)
    previous = None
    for k, pos in enumerate(positions):
        image = _read(records[pos][1])
        gray = cv2.cvtColor(cv2.resize(image, (320, 180)), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            mask = np.zeros_like(previous)
            mask[25:150, 8:312] = 255
            points = cv2.goodFeaturesToTrack(previous, maxCorners=180, qualityLevel=0.012,
                                             minDistance=8, mask=mask)
            dt = max(1, records[pos][0] - records[positions[k - 1]][0])
            signals[k, 6] = np.mean(cv2.absdiff(previous, gray)) / 255.0
            if points is not None and len(points) >= 8:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None,
                                                       winSize=(21, 21), maxLevel=3)
                if nxt is None or status is None:
                    previous = gray
                    continue
                back, bst, _ = cv2.calcOpticalFlowPyrLK(gray, previous, nxt, None,
                                                      winSize=(21, 21), maxLevel=3)
                if back is None or bst is None:
                    previous = gray
                    continue
                keep = status[:, 0].astype(bool) & bst[:, 0].astype(bool)
                keep &= np.isfinite(nxt[:, 0]).all(axis=1) & np.isfinite(back[:, 0]).all(axis=1)
                keep &= np.linalg.norm(points[:, 0] - back[:, 0], axis=1) < 1.7
                p0, p1 = points[keep, 0], nxt[keep, 0]
                if len(p0) >= 8:
                    mat, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                             ransacReprojThreshold=1.8,
                                                             maxIters=200, confidence=0.98)
                    if mat is not None and inliers is not None and np.isfinite(mat).all():
                        scale = math.hypot(mat[0, 0], mat[1, 0])
                        # Center translation removes false shake induced by scale at image origin.
                        shift = mat[:, :2] @ np.array([160.0, 90.0]) + mat[:, 2] - [160.0, 90.0]
                        residual = np.linalg.norm(p1 - (p0 @ mat[:, :2].T + mat[:, 2]), axis=1)
                        signals[k, :6] = [shift[0] / dt, shift[1] / dt,
                                          160 * math.atan2(mat[1, 0], mat[0, 0]) / dt,
                                          160 * (scale - 1.0) / dt,
                                          np.median(residual) / dt, float(inliers.mean())]
        previous = gray
    trajectory = signals[:, :4]
    # Jerk and rotation are impact cues; large cuts with failed correspondences are not.
    jerk = np.linalg.norm(np.diff(trajectory, axis=0, prepend=trajectory[:1]), axis=1)
    rotation = np.abs(signals[:, 2])
    shake = 0.65 * _robust_positive(jerk) + 0.20 * _robust_positive(rotation)
    shake += 0.15 * _robust_positive(signals[:, 4])
    cut = (signals[:, 6] > 0.32) & (signals[:, 5] < 0.2)
    for i in np.flatnonzero(cut):
        shake[max(0, i - 1):min(len(shake), i + 2)] *= 0.05
    if len(shake) > 2:
        shake[:2] = 0
    return positions, shake, signals


def _iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:4], b[2:4])
    inter = np.maximum(hi - lo, 0).prod()
    aa = np.maximum(a[2:4] - a[:2], 0).prod()
    bb = np.maximum(b[2:4] - b[:2], 0).prod()
    return inter / max(aa + bb - inter, 1e-8)


@dataclass
class Track:
    positions: list = field(default_factory=list)
    boxes: list = field(default_factory=list)


def _tracks(positions, perceptions):
    """Associate vehicle boxes over sampled frames with Hungarian matching."""
    tracks = []
    # Fine event samples must not shrink the association lifetime on coarse sections.
    step = max(1.0, float(np.quantile(np.diff(positions), 0.9))) if len(positions) > 1 else 1.0
    for p in positions:
        boxes = perceptions[p].boxes
        alive = [t for t in tracks if p - t.positions[-1] <= 3.5 * step]
        used = set()
        if alive and len(boxes):
            costs = np.full((len(alive), len(boxes)), 9.0)
            for i, t in enumerate(alive):
                expected = t.boxes[-1][:4].copy()
                if len(t.positions) >= 2:
                    velocity = (t.boxes[-1][:4] - t.boxes[-2][:4]) / max(1, t.positions[-1] - t.positions[-2])
                    expected += np.clip(velocity * (p - t.positions[-1]), -0.12, 0.12)
                for j, b in enumerate(boxes):
                    overlap = _iou(expected, b)
                    dist = np.linalg.norm((expected[:2] + expected[2:4] - b[:2] - b[2:4]) / 2)
                    size = max(0.03, float(np.sqrt(np.prod(np.maximum(b[2:4] - b[:2], 0)))))
                    if overlap > 0.04 or dist < max(0.045, size * 0.6):
                        costs[i, j] = 1 - overlap + 0.6 * dist / max(size, 0.05)
            ii, jj = linear_sum_assignment(costs)
            for i, j in zip(ii, jj):
                if costs[i, j] < 1.7:
                    alive[i].positions.append(int(p))
                    alive[i].boxes.append(boxes[j])
                    used.add(int(j))
        for j, box in enumerate(boxes):
            if j not in used and box[4] >= 0.22:
                tracks.append(Track([int(p)], [box]))
    return tracks


def _proximity(boxes):
    b = np.asarray(boxes)
    width = np.maximum(b[:, 2] - b[:, 0], 0)
    height = np.maximum(b[:, 3] - b[:, 1], 0)
    area = width * height
    center = (b[:, 0] + b[:, 2]) / 2
    # Side contacts remain candidates, but distant roadside objects are de-emphasized.
    centrality = 0.40 + 0.60 * np.exp(-((center - 0.5) / 0.35) ** 2)
    bottom = np.clip((b[:, 3] - 0.38) / 0.55, 0, 1)
    return centrality * bottom * np.clip(np.sqrt(area) / 0.45, 0, 1)


def _choose_event(positions, perceptions, tracks, mpos, shake, n):
    """Fuse impact motion with nearby-vehicle evidence into one event candidate."""
    close = np.zeros(len(positions))
    for k, p in enumerate(positions):
        b = perceptions[p].boxes
        if len(b):
            close[k] = np.max(_proximity(b))
    pclose = np.interp(mpos, positions, close)
    smoothed = _smooth(pclose, 1.0)
    approach = _robust_positive(np.maximum(np.diff(smoothed, prepend=smoothed[:1]), 0))
    # Event shape uses both impact and nearby vehicles, with no assumed collision offset.
    score = shake * (0.35 + 0.65 * smoothed) + 0.18 * smoothed + 0.08 * approach
    if len(score) > 4:
        score[:2] *= 0.15
    event = int(mpos[int(np.argmax(score))])
    best_track, best = None, -1.0
    for t in tracks:
        if len(t.positions) < 2:
            continue
        b = np.asarray(t.boxes)
        prox = _proximity(b)
        dist = np.abs(np.asarray(t.positions) - event)
        evidence = prox * np.exp(-dist / max(10.0, n * 0.12))
        center = (b[:, 0] + b[:, 2]) / 2
        lateral = min(1.0, float(np.ptp(center)) / 0.25)
        growth = min(1.0, float(np.ptp(np.sqrt(np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0)))) / 0.25)
        quality = float(np.max(evidence)) * (0.75 + 0.15 * lateral + 0.10 * growth)
        quality *= min(1.0, len(t.positions) / 4.0)
        if quality > best:
            best, best_track = quality, t
    if best_track is not None:
        prox = _proximity(best_track.boxes)
        track_close = np.interp(mpos, best_track.positions, prox, left=0, right=0)
        score = shake * (0.30 + 0.70 * track_close) + 0.18 * track_close + 0.06 * approach
        if len(score) > 4:
            score[:2] *= 0.15
        event = int(mpos[int(np.argmax(score))])
    return event, best_track, score


def _impact_onset(event, track, mpos, shake):
    """Use the start of a connected impact burst, not a later hood/body rebound."""
    if track is None or not len(mpos):
        return event
    close = np.interp(mpos, track.positions, _proximity(track.boxes), left=0, right=0)
    i = int(np.argmin(np.abs(mpos - event)))
    if close[i] < 0.35 or shake[i] < 0.12:
        return event
    threshold = max(0.035, 0.20 * shake[i])
    start, quiet = i, 0
    for j in range(i - 1, max(-1, i - 13), -1):
        if close[j] < 0.55 * close[i]:
            break
        if shake[j] >= threshold:
            start, quiet = j, 0
        else:
            quiet += 1
            if quiet >= 2:
                break
    return int(mpos[start])


def _entry(track, collision, perceptions):
    """Infer entry time/side from the target track relative to the ego corridor."""
    if track is None:
        # No object track: fall back to the dominant observed lane-side occupancy.
        p = min(perceptions, key=lambda x: abs(x - collision))
        boxes = perceptions[p].boxes
        side = "LEFT" if not len(boxes) or np.mean(boxes[:, [0, 2]]) < 0.5 else "RIGHT"
        return collision, side
    eligible = [i for i, p in enumerate(track.positions) if p <= collision]
    if not eligible:
        eligible = [0]
    b = np.asarray(track.boxes)[eligible]
    pp = np.asarray(track.positions)[eligible]
    count = max(1, min(len(b), max(3, len(b) // 3)))
    center = (b[:, 0] + b[:, 2]) / 2
    relative = []
    for p, box in zip(pp[:count], b[:count]):
        view = perceptions[int(p)]
        y = float(np.clip(box[3], 0.45, 0.95))
        lc = np.polyval(view.left, y)
        rc = np.polyval(view.right, y)
        relative.append((box[0] + box[2]) / 2 - (lc + rc) / 2)
    start_offset = float(np.median(relative))
    if abs(start_offset) < 0.035 and len(center) >= 3:
        # A box arriving from the left normally has positive rightward screen motion.
        side = "LEFT" if center[-1] - center[0] >= 0 else "RIGHT"
    else:
        side = "LEFT" if start_offset < 0 else "RIGHT"
    margins = []
    for p, box in zip(pp, b):
        view = perceptions[int(p)]
        y = float(np.clip(box[3], 0.45, 0.95))
        lc, rc = np.polyval(view.left, y), np.polyval(view.right, y)
        # The near edge first crossing the ego lane is earlier than box-center crossing.
        m = box[2] - lc if side == "LEFT" else rc - box[0]
        margins.append(float(m / max(0.10, rc - lc)))
    margins = median_filter(np.asarray(margins), size=min(3, len(margins)), mode="nearest")
    crossed = margins > 0.025
    entry = int(pp[0])
    for i in range(len(pp)):
        if crossed[i] and (i == len(pp) - 1 or crossed[i + 1]):
            if i and margins[i] > margins[i - 1]:
                fraction = np.clip((0.025 - margins[i - 1]) / (margins[i] - margins[i - 1]), 0, 1)
                entry = int(round(pp[i - 1] + fraction * (pp[i] - pp[i - 1])))
            else:
                entry = int(pp[i])
            break
    else:
        entry = int(pp[int(np.argmax(margins))])
    return min(entry, collision), side


def _evasion(perceptions, collision):
    """Vote for visible adjacent clearance near collision; this is not safety proof."""
    # Adjacent *visible* road plus traffic clearance; not proof of safe physical escape.
    selected = sorted(perceptions, key=lambda x: abs(x - collision))[:3]
    votes = []
    yy, xx = np.mgrid[0:180, 0:320]
    y, x = (yy + 0.5) / 180, (xx + 0.5) / 320
    for p in selected:
        view = perceptions[p]
        left, right = np.polyval(view.left, y), np.polyval(view.right, y)
        width = np.clip(right - left, 0.12, 0.80)
        # Sample an adjacent-lane corridor at medium/near depth, excluding hood/sky.
        band = (y > 0.57) & (y < 0.89)
        occupied = np.zeros_like(view.road, dtype=bool)
        for box in view.boxes:
            x0 = max(0, int((box[0] - 0.02) * 320))
            x1 = min(320, int((box[2] + 0.02) * 320))
            y0 = max(0, int((box[1] - 0.01) * 180))
            y1 = min(180, int((box[3] + 0.02) * 180))
            occupied[y0:y1, x0:x1] = True
        any_free = False
        for region in [band & (x < left - width * 0.04) & (x > left - width * 0.80),
                       band & (x > right + width * 0.04) & (x < right + width * 0.80)]:
            if region.sum() < 100:
                continue
            drivable = float(view.road[region].mean())
            blockage = float(occupied[region].mean())
            free = float((view.road & ~occupied)[region].mean())
            if drivable > 0.48 and free > 0.40 and blockage < 0.18:
                any_free = True
        votes.append(any_free)
    return int(sum(votes) > len(votes) / 2)


def predict_video(records, perception, temporal_guard=None):
    """Run one Stage 2 video without carrying state into the next video."""
    # RANSAC's generator must not inherit the previous evaluation video's state.
    cv2.setRNGSeed(236753)
    n = len(records)
    positions = np.unique(np.linspace(0, n - 1, min(n, MAX_PERCEPTION_FRAMES)).round().astype(int))
    views = {int(p): perception(_read(records[p][1])) for p in positions}
    mpos, shake, _ = _motion_signals(records)
    tracks = _tracks(positions, views)
    collision, target, _ = _choose_event(positions, views, tracks, mpos, shake, n)
    entry, side = _entry(target, collision, views)

    # Local refinement decodes nearby original frames instead of returning sampled indices.
    step = max(1, int(math.ceil(n / MAX_PERCEPTION_FRAMES)))
    refine = set()
    if step > 1:
        for event in (collision, entry):
            low, high = max(0, event - 2 * step), min(n - 1, event + 2 * step)
            refine.update(np.linspace(low, high, min(high - low + 1, 49)).round().astype(int))
    for p in sorted(refine.difference(views)):
        views[p] = perception(_read(records[p][1]))
    if refine:
        positions = np.asarray(sorted(views))
        tracks = _tracks(positions, views)
        collision, target, _ = _choose_event(positions, views, tracks, mpos, shake, n)
        entry, side = _entry(target, collision, views)
    if n > MAX_MOTION_FRAMES:
        # On long clips, restore per-frame timing around the impact candidate.
        radius = min(80, max(12, int(math.ceil(n / MAX_MOTION_FRAMES)) * 4))
        low, high = max(0, collision - radius), min(n, collision + radius + 1)
        local_pos, local_shake, _ = _motion_signals(records[low:high])
        if np.max(local_shake) > 0.05:
            local_pos = local_pos + low
            closeness = np.zeros(len(positions))
            for k, p in enumerate(positions):
                if len(views[p].boxes):
                    closeness[k] = np.max(_proximity(views[p].boxes))
            close_local = np.interp(local_pos, positions, closeness)
            local_score = local_shake * (0.35 + 0.65 * close_local) + 0.08 * close_local
            collision = int(local_pos[int(np.argmax(local_score))])
            collision = _impact_onset(collision, target, local_pos, local_shake)
            entry, side = _entry(target, collision, views)
    else:
        collision = _impact_onset(collision, target, mpos, shake)
        entry, side = _entry(target, collision, views)
    if temporal_guard is not None:
        corrected = temporal_guard(records, int(np.clip(collision, 0, n-1)))
        if corrected != collision:
            collision = corrected
            for pos in range(max(0,collision-6), min(n,collision+7)):
                if pos not in views:
                    views[pos] = perception(_read(records[pos][1]))
            positions = np.asarray(sorted(views))
            tracks = _tracks(positions, views)
            usable = [track for track in tracks if len(track.positions)>=2]
            def support(track):
                proximity = _proximity(np.asarray(track.boxes))
                distance = np.abs(np.asarray(track.positions)-collision)
                return float(np.max(proximity*np.exp(-distance/10.0))) * min(1.0,len(track.positions)/4.0)
            target = max(usable,key=support) if usable else None
            entry,side = _entry(target,collision,views)
    collision = int(np.clip(collision, 0, n - 1))
    entry = int(np.clip(entry, 0, collision))
    return {"collision_frame": int(records[collision][0]),
            "entry_frame": int(records[entry][0]),
            "evasion_space": _evasion(views, collision), "entry_side": side}


def predict(data_dir, model_dir):
    """Return one valid row for each Stage 2 image folder, with original frame IDs."""
    root = Path(data_dir)
    image_root = root / "images" if (root / "images").is_dir() else root
    folders = sorted(p for p in image_root.iterdir() if p.is_dir())
    if not folders:
        return pd.DataFrame({"ID": pd.Series(dtype="str"),
                             "collision_frame": pd.Series(dtype="int64"),
                             "entry_frame": pd.Series(dtype="int64"),
                             "evasion_space": pd.Series(dtype="int64"),
                             "entry_side": pd.Series(dtype="str")})[COLUMNS]
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    perception = RoadPerception(model_dir)
    from .temporal_guard import TemporalGuard
    temporal_guard = TemporalGuard(model_dir, perception.device)
    rows = []
    for folder in folders:
        records = _enumerate_frames(folder)
        result = predict_video(records, perception, temporal_guard)
        rows.append({"ID": folder.name, **result})
    out = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("collision_frame", "entry_frame", "evasion_space"):
        out[column] = out[column].astype("int64")
    return out
