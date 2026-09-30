"""Frozen DoTA/DADA agreement guard; per-video state only, original frame positions.

SimpleTAD architecture and pretrained weights: tue-mps/simple-tad, CC-BY-NC-4.0.
Anomaly onset is NOT generally physical contact. Override only when both models
strongly contradict the geometric event and agree on a sharp alternative onset.
"""
from pathlib import Path
from collections import OrderedDict
from contextlib import nullcontext
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter1d
from .temporal_model import get_video_vit_small, prepare_image


def guarded_position(base, probabilities):
    """Override ``base`` only for a sharp, high-confidence two-model consensus.

    SimpleTAD predicts anomaly probability rather than physical contact. These
    checks therefore make abstention the default whenever either domain model
    supports the geometric candidate or their alternative onsets disagree.
    """
    if len(probabilities) != 2 or min(map(len, probabilities)) < 16:
        return int(base)
    smooth = [gaussian_filter1d(np.asarray(y, dtype=float), 1) for y in probabilities]
    if max(y[base] for y in smooth) > .2 or min(y.max() for y in smooth) < .9:
        return int(base)
    changes = [np.diff(y, prepend=y[0]) for y in smooth]
    peaks = [int(y.argmax()) for y in changes]
    if abs(peaks[0]-peaks[1]) > 3 or min(y[i] for y,i in zip(changes,peaks)) < .08:
        return int(base)
    return int(np.rint(np.mean(peaks)))


class TemporalGuard:
    """Lazy-loaded DoTA/DADA SimpleTAD ensemble used as a contradiction guard."""

    def __init__(self, model_dir, device):
        self.root = Path(model_dir)
        self.device = torch.device(device)
        self.models = []

    def _load(self):
        if self.models:
            return
        for filename in ('simpletad_dada.pth', 'simpletad_dota.pth'):
            net = get_video_vit_small()
            state = torch.load(self.root/filename, map_location='cpu', weights_only=True)
            net.load_state_dict(state.get('model', state), strict=True)
            self.models.append(net.eval().requires_grad_(False).to(self.device))

    @torch.inference_mode()
    def probabilities(self, records):
        """Return per-original-frame anomaly traces for both frozen checkpoints."""
        self._load()
        n = len(records)
        positions = np.unique(np.r_[np.arange(0,n,max(1,int(np.ceil(n/360)))),n-1]).astype(int)
        cache = OrderedDict()
        def frame(index):
            if index not in cache:
                im = cv2.imread(str(records[index][1]))
                if im is None:
                    raise ValueError('Cannot decode Stage 2 frame')
                im = cv2.resize(im, (224,224), interpolation=cv2.INTER_CUBIC)
                cache[index] = prepare_image(im, (.485,.456,.406), (.229,.224,.225))
            cache.move_to_end(index)
            value = cache[index]
            while len(cache)>80:
                cache.popitem(last=False)
            return value
        values = [[], []]
        batch_size = 4 if self.device.type == 'cuda' else 1
        for j in range(0,len(positions),batch_size):
            windows = []
            for pos in positions[j:j+batch_size]:
                windows.append(torch.stack([frame(max(0,int(pos)-15+k)) for k in range(16)],dim=1))
            tensor = torch.stack(windows).to(self.device)
            context = torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else nullcontext()
            with context:
                for i,net in enumerate(self.models):
                    original = net(tensor).float()[:,1]
                    reflected = net(tensor.flip(-1)).float()[:,1]
                    values[i].extend(((original+reflected)*.5).cpu().tolist())
        return [np.interp(np.arange(n),positions,y) for y in values]

    def __call__(self, records, baseline_position):
        if len(records)<16:
            return int(baseline_position)
        return guarded_position(baseline_position,self.probabilities(records))
