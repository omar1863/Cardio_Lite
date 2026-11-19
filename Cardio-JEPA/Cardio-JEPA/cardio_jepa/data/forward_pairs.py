"""Forward-prediction training pairs that re-draw on every access, plus record-level
hold-out helpers.

Add-on module: it only *imports* from the existing package (LongRecord, the window
preprocessing, InterleavedLoader), so it can be dropped into any version of
``cardio_jepa`` without replacing ``longform.py`` or ``build.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import SignalConfig
from .longform import LongRecord, collate_mixed
from .preprocess import expand_to_n_leads, is_valid, prepare_window


class ResamplingForwardPairDataset(Dataset):
    """(context strip, future window, horizon) samples for L_fwd.

    A sample is ``W`` consecutive windows ending at time t, plus the window that
    starts ``horizon`` seconds after t. When the hierarchical stack is enabled an
    extra far-future window is drawn at the Level-2 horizon.

    With ``resample=True`` (default) every ``__getitem__`` draws a fresh record
    (weighted by usable duration), start time and horizon, so the whole of every
    Holter recording is used over training instead of a fixed set of
    ``samples_per_record`` pairs. ``len()`` is then a nominal epoch size.
    """

    def __init__(self, records: Sequence[LongRecord], signal: SignalConfig,
                 context_windows: int = 1,
                 horizons_s: Sequence[float] = (10.0, 30.0, 60.0, 300.0),
                 samples_per_record: int = 200,
                 l2_horizons_s: Optional[Sequence[float]] = None,
                 seed: int = 0,
                 resample: bool = True,
                 loguniform: bool = True):
        self.records = [r for r in records if r.duration_s > 0]
        self.signal = signal
        self.W = max(1, context_windows)
        self.horizons = sorted(float(h) for h in horizons_s)
        self.l2_horizons = list(l2_horizons_s) if l2_horizons_s else None
        self.resample = resample
        self.loguniform = loguniform and len(self.horizons) > 1
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._worker_rng = None

        win_s = signal.window_s
        ctx_s = self.W * win_s
        # usable span per record for the smallest horizon; records that cannot fit
        # even that are dropped
        self.usable = np.array([max(0.0, r.duration_s - ctx_s - win_s - self.horizons[0])
                                for r in self.records])
        keep = self.usable > 0
        self.records = [r for r, k in zip(self.records, keep) if k]
        self.usable = self.usable[keep]
        self.p_rec = (self.usable / self.usable.sum()) if len(self.usable) else None
        self.n_nominal = samples_per_record * len(self.records)

        self.plan: List[Tuple[int, int, float, Optional[float]]] = []
        if not resample:
            for ri in range(len(self.records)):
                for _ in range(samples_per_record):
                    item = self._draw(self.rng, ri)
                    if item is not None:
                        self.plan.append(item)

    def _horizon(self, rng) -> float:
        if self.loguniform:
            lo, hi = np.log(self.horizons[0]), np.log(self.horizons[-1])
            return float(np.exp(rng.uniform(lo, hi)))
        return float(rng.choice(self.horizons))

    def _draw(self, rng, ri: Optional[int] = None):
        if ri is None:
            ri = int(rng.choice(len(self.records), p=self.p_rec))
        rec = self.records[ri]
        win_s = self.signal.window_s
        ctx_s = self.W * win_s
        h = self._horizon(rng)
        l2h = float(rng.choice(self.l2_horizons)) if self.l2_horizons else None
        need_s = ctx_s + max(h, l2h or 0.0) + win_s
        if rec.duration_s <= need_s:
            usable = rec.duration_s - ctx_s - win_s
            if usable <= self.horizons[0]:
                return None
            h = float(rng.uniform(self.horizons[0], usable)) if self.loguniform \
                else float(max(x for x in self.horizons if x < usable))
            if l2h is not None and ctx_s + l2h + win_s >= rec.duration_s:
                l2h = None
            need_s = ctx_s + max(h, l2h or 0.0) + win_s
        start_s = float(rng.uniform(0.0, rec.duration_s - need_s))
        return (ri, int(start_s * rec.fs), h, l2h)

    def __len__(self) -> int:
        return self.n_nominal if self.resample else len(self.plan)

    def _rng(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.rng
        if self._worker_rng is None or self._worker_rng[0] != info.id:
            # distinct, but reproducible, stream per worker (and per epoch, since
            # workers are re-created each epoch unless persistent)
            self._worker_rng = (info.id, np.random.default_rng(
                [self.seed, info.id, int(torch.initial_seed() % (2 ** 31))]))
        return self._worker_rng[1]

    def _prep(self, rec: LongRecord, start: int) -> np.ndarray:
        n_native = int(round(self.signal.window_s * rec.fs))
        raw = rec.read(start, n_native)
        x = prepare_window(raw, rec.fs, self.signal.fs, self.signal.window_samples,
                           reduced=(rec.n_channels >= 12))
        x = expand_to_n_leads(x, self.signal.n_leads)
        return x if is_valid(x) else np.zeros_like(x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.resample:
            rng = self._rng()
            item = None
            while item is None:
                item = self._draw(rng)
            ri, start, h, l2h = item
        else:
            ri, start, h, l2h = self.plan[idx]
        rec = self.records[ri]
        win_native = int(round(self.signal.window_s * rec.fs))

        ctx = np.stack([self._prep(rec, start + w * win_native) for w in range(self.W)])
        ctx_end = start + self.W * win_native
        fut = self._prep(rec, ctx_end + int(round(h * rec.fs)))

        out = {
            "ctx": torch.from_numpy(ctx).float(),                 # (W, C, T)
            "future": torch.from_numpy(fut).float(),              # (C, T)
            "has_future": torch.tensor(1, dtype=torch.bool),
            "horizon_s": torch.tensor(float(h)),
        }
        if self.l2_horizons is not None:
            # keep every item's keys identical so default collation never breaks
            if l2h is None:
                l2h = float(h)
            l2f = self._prep(rec, ctx_end + int(round(l2h * rec.fs)))
            out["l2_future"] = torch.from_numpy(l2f).float()
            out["l2_horizon_s"] = torch.tensor(float(l2h))
            out["has_l2"] = torch.tensor(1, dtype=torch.bool)
        return out


def holdout_long_records(records: Sequence[LongRecord], test_frac: float = 0.3,
                         seed: int = 0) -> Tuple[List[int], List[int]]:
    """Record-level train/held-out split, stratified by AF burden.

    Pretrain the forward objective on the train records only and evaluate forecast
    skill / anticipation / burden on the held-out ones. Otherwise every Stage 2/4
    "test" patient has already been seen -- future windows included -- during
    pretraining, which is optimistic for exactly the claim being tested.
    """
    n = len(records)
    if n < 2:
        return list(range(n)), []
    rng = np.random.default_rng(seed)
    order = sorted(range(n), key=lambda i: (records[i].burden(), rng.random()))
    n_test = max(1, int(round(n * test_frac)))
    step = n / n_test
    test = sorted({order[min(n - 1, int(k * step + rng.uniform(0, step)))]
                   for k in range(n_test)})
    train = [i for i in range(n) if i not in set(test)]
    return train, test


def restrict_to_records(samples: Sequence[object], record_ids: Sequence[int]
                        ) -> List[object]:
    keep = set(int(i) for i in record_ids)
    return [s for s in samples if s.record in keep]




def build_forward_loader(records: Sequence[LongRecord], cfg, samples_per_record: int = 200,
                         batch_size: Optional[int] = None) -> Optional[DataLoader]:
    """DataLoader of re-drawn (context, future, horizon) pairs for ``records``."""
    ds = ResamplingForwardPairDataset(
        records, cfg.signal, context_windows=cfg.forward.context_windows,
        horizons_s=cfg.forward.horizons_s, samples_per_record=samples_per_record,
        l2_horizons_s=(cfg.hierarchical.l2_horizons_s if cfg.hierarchical.enabled else None),
        seed=cfg.optim.seed, resample=True,
        loguniform=getattr(cfg.forward, "horizon_loguniform", True))
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size or cfg.optim.batch_size, shuffle=False,
                      num_workers=cfg.optim.num_workers, drop_last=True,
                      collate_fn=collate_mixed, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False)


def combine_loaders(short_loader, long_loader, long_weight: float = 2.0,
                    steps_per_epoch: Optional[int] = None):
    """Interleave short-clip and forward-pair batches (same rule as build.py)."""
    from ..engine import InterleavedLoader
    loaders, weights = [], []
    if short_loader is not None:
        loaders.append(short_loader); weights.append(1.0)
    if long_loader is not None:
        loaders.append(long_loader); weights.append(long_weight)
    return InterleavedLoader(loaders, weights, steps_per_epoch=steps_per_epoch)
