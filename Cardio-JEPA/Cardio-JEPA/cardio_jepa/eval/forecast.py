"""Held-out forecast-skill and horizon-sweep evaluation (add-on module).

Only imports from the existing package, so it can be dropped into any version of
``cardio_jepa`` without replacing ``benchmarks.py``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..config import EvalConfig, SignalConfig
from ..data.longform import LongRecord
from ..data.preprocess import expand_to_n_leads, prepare_window


@torch.no_grad()
def forecast_skill_v2(model, records: Sequence[LongRecord], signal: SignalConfig, device,
                   horizons_s: Sequence[float] = (10.0, 60.0, 300.0, 900.0),
                   n_samples: int = 512, context_windows: int = 1, seed: int = 0,
                   record_ids: Optional[Sequence[int]] = None,
                   batch_size: int = 64, verbose: bool = True):
    """Compare the predictor against a persistence baseline, per horizon.

    Everything is measured in the model's own *target space* -- the space L_fwd is
    trained in (standardised mean of LayerNormed target-encoder tokens). The previous
    version compared the prediction against the *raw* token mean and built
    persistence from the *online* encoder, i.e. three different spaces; its
    persistence cosines of 0.01-0.04 for adjacent windows of the same patient were a
    symptom of that, not a property of the data.

      * ``cos_pred``        centred cos( g(z_ctx, h),            target(x_future) )
      * ``cos_persistence`` centred cos( target(x_last_context), target(x_future) )
      * ``skill``           cos_pred - cos_persistence   (> 0: beats copying the present)
      * ``mse_skill``       1 - MSE_pred / MSE_persistence (forecast skill score)

    Pass ``record_ids`` = held-out records (see ``holdout_long_records``) so the
    score is not measured on recordings the predictor was trained on.
    """
    import pandas as pd

    model.eval()
    if not hasattr(model, "forward_predictor"):
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    pool = list(range(len(records))) if record_ids is None else list(record_ids)
    usable = [i for i in pool if records[i].duration_s > 0]
    if not usable:
        return pd.DataFrame()

    rows = []
    for H in horizons_s:
        ctxs, futs = [], []
        need_s = context_windows * signal.window_s + H + signal.window_s
        cand = [i for i in usable if records[i].duration_s > need_s]
        if not cand:
            if verbose:
                print(f"  horizon {H:.0f}s: no record long enough, skipping")
            continue
        w = np.array([records[i].duration_s - need_s for i in cand])
        for _ in range(n_samples):
            rec = records[cand[int(rng.choice(len(cand), p=w / w.sum()))]]
            t0 = float(rng.uniform(0.0, rec.duration_s - need_s))
            start = int(t0 * rec.fs)
            win_n = int(round(signal.window_s * rec.fs))

            def prep(s):
                raw = rec.read(s, win_n)
                x = prepare_window(raw, rec.fs, signal.fs, signal.window_samples,
                                   reduced=(rec.n_channels >= 12))
                return expand_to_n_leads(x, signal.n_leads)

            ctxs.append(np.stack([prep(start + w_ * win_n) for w_ in range(context_windows)]))
            futs.append(prep(start + context_windows * win_n + int(round(H * rec.fs))))

        if len(ctxs) < 8:
            continue

        P, Q, T = [], [], []
        for i in range(0, len(ctxs), batch_size):
            ctx = torch.from_numpy(np.stack(ctxs[i:i + batch_size])).float().to(device)
            fut = torch.from_numpy(np.stack(futs[i:i + batch_size])).float().to(device)
            if hasattr(model, "target_latent"):
                pred, persist = model.forecast_latent(ctx, H, return_persistence=True)
                true = model.target_latent(fut)
            else:                                   # older checkpoints / API
                pred = model.forecast_latent(ctx, H)
                true = model._pooled_target(fut)
                persist = model._pooled_target(ctx[:, -1])
            P.append(pred.float().cpu()); Q.append(persist.float().cpu())
            T.append(true.float().cpu())
        pred, persist, true = torch.cat(P), torch.cat(Q), torch.cat(T)

        def ccos(a, b):
            a = a - a.mean(0, keepdim=True)
            b = b - b.mean(0, keepdim=True)
            return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())

        cp, cq = ccos(pred, true), ccos(persist, true)
        mse_p = float(((pred - true) ** 2).mean())
        mse_q = float(((persist - true) ** 2).mean())
        mse_skill = 1.0 - mse_p / max(mse_q, 1e-12)
        rows.append({"horizon_s": H, "cos_pred": cp, "cos_persistence": cq,
                     "skill": cp - cq, "mse_pred": mse_p, "mse_persistence": mse_q,
                     "mse_skill": mse_skill, "n": len(ctxs)})
        if verbose:
            print(f"  horizon {H:6.0f}s  cos_pred={cp:+.4f}  "
                  f"cos_persistence={cq:+.4f}  skill={cp - cq:+.4f}  "
                  f"mse_skill={mse_skill:+.4f}")

    return pd.DataFrame(rows)



def horizon_sweep_heldout(model_fns: Dict[str, object], records: Sequence[LongRecord],
                          signal: SignalConfig, device, eval_cfg: EvalConfig,
                          horizons_s: Sequence[float], builder,
                          train_ids: Sequence[int], test_ids: Sequence[int],
                          context_windows: int = 1, verbose: bool = True):
    """``horizon_sweep`` with a *fixed* record-level split (the pretraining hold-out),
    so no test patient was seen during pretraining."""
    import pandas as pd
    from ..data.forward_pairs import restrict_to_records
    from .benchmarks import run_anticipation_probe

    rows = []
    for H in horizons_s:
        samples, stats = builder(records, signal, horizon_s=H,
                                 context_windows=context_windows)
        tr = restrict_to_records(samples, train_ids)
        te = restrict_to_records(samples, test_ids)
        n_te_pos = sum(1 for s in te if s.label > 0.5)
        if n_te_pos < 5 or len(te) - n_te_pos < 5 or not tr:
            if verbose:
                print(f"  horizon {H:.0f}s: too few held-out samples "
                      f"(pos={n_te_pos}, neg={len(te) - n_te_pos}), skipping")
            continue
        for name, spec in model_fns.items():
            model = spec["model"] if isinstance(spec, dict) else spec
            mode = spec.get("feature_mode", "context") if isinstance(spec, dict) else "context"
            r = run_anticipation_probe(model, records, tr, te, signal, device, eval_cfg,
                                       context_windows=context_windows,
                                       feature_mode=mode, horizons_s=[H], verbose=False)
            rows.append({"horizon_s": H, "model": name, "auroc": r.get("auroc"),
                         "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
                         "prevalence": r.get("prevalence"), "n_test": r.get("n_test")})
            if verbose:
                print(f"  horizon {H:6.0f}s  {name:28s} "
                      f"AUROC={r.get('auroc', float('nan')):.4f}")
    return pd.DataFrame(rows)
