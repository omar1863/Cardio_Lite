#!/usr/bin/env python3
"""Generate the lightweight Kaggle notebook (one GPU session, forward prediction first).

Usage:  python tools/build_lite_notebook.py [-o notebooks/cardio_jepa_kaggle_lite.ipynb]
"""

from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_N = [0]


def _id() -> str:
    _N[0] += 1
    return f"lite-{_N[0]:03d}"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _id(), "metadata": {},
            "source": src.strip("\n").splitlines(True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(True)}


CELLS = [
md(r"""
# Cardio-JEPA Lite: latent forward prediction on ECG (one Kaggle GPU session)

A lightweight, end-to-end run of the core claim: **an ECG encoder trained to predict the
latent of a future window can forecast where the heart's state is going, better than
copying the present**. It trains two models and evaluates them on held-out patients.

| model | objective |
|---|---|
| `dual` (Cardio-JEPA) | masked-latent prediction **+** cross-window forward prediction |
| `masked_only` (ECG-JEPA baseline) | masked-latent prediction only |

| evaluation | question | read |
|---|---|---|
| **A. Forecast skill** | does the predictor beat persistence (copying the present)? | `mse_skill > 0` |
| **B. AF-onset anticipation** | does AF start within *H* s of an AF-free context? | AUROC, `world` vs `context` features |
| **C. PTB-XL linear probe** | does forward prediction cost classification accuracy? | macro AUROC |

All Holter-based tests use **patients held out from pretraining**.

### How to run on Kaggle
1. **Settings → Accelerator → GPU P100** (or GPU T4 x2; this notebook uses one GPU).
2. **Get the code in**, either way:
   * **Add Input → Datasets → New dataset**, upload the repository zip (works with
     Internet off), or
   * set `REPO_URL` in the settings cell to your GitHub repository and turn
     **Settings → Internet → On**.
3. **Add Input → Datasets** with the ECG corpora, as many as you like (discovery matches folder names):
   * long Holter (needed for A and B): `sahilshahare12/long-term-af-database`, plus any
     MIT-BIH AF DB / SHDB-AF mirror
   * short 12-lead (pretraining + C): `khyeh0719/ptb-xl-dataset`,
     `physionet/china-physiological-signal-challenge-in-2018`,
     `erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database`
4. **Save Version → Save & Run All (Commit)**. The run checkpoints every epoch to
   `/kaggle/working/runs/` and stops cleanly before the time budget.

**If a session ends before training finishes:** add this notebook's own output as an
input (**Add Input → Your Work → this notebook**) and run it again; training resumes
from the last epoch.
"""),

md("## 1. Settings"),
code(r'''
# --- the knobs -----------------------------------------------------------------------
RUN_MODE        = "lite"          # "lite" (Kaggle GPU) | "smoke" (minutes, CPU-ok, pipeline check)
ARMS            = ("dual", "masked_only")   # drop "masked_only" to halve the training time
TIME_BUDGET_H   = 10.5            # whole notebook; Kaggle kills GPU sessions at 12 h
EVAL_RESERVE_H  = 1.0             # kept back from training for the evaluation cells
HOLDOUT_FRAC    = 0.3             # fraction of Holter records held out from pretraining
AUTO_FIT_BUDGET = True            # shrink steps/epoch if the projection overruns the budget
RUN_PTBXL_PROBE = True            # evaluation C

# Where the code comes from (first match wins):
#   1. the repo attached as a Kaggle dataset (Add Input -> Datasets) - works offline
#   2. git clone of REPO_URL (needs Settings -> Internet -> On)
REPO_URL = "https://github.com/YOUR-USERNAME/Cardio-JEPA.git"   # <- your repository
BRANCH   = ""                     # "" = the repository's default branch (e.g. main)

import os, sys, time, json, glob, shutil, subprocess, warnings
T0 = time.time()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
ON_KAGGLE = os.path.isdir("/kaggle")
INPUT_ROOT = "/kaggle/input" if ON_KAGGLE else os.path.abspath("input")
WORK = "/kaggle/working" if ON_KAGGLE else os.path.abspath("work")
os.makedirs(WORK, exist_ok=True)
'''),

md("## 2. Get the code"),
code(r'''
# --- locate cardio_jepa: already importable / attached as input / clone from GitHub --
def find_package_root(roots, max_depth=6):
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        stack = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if os.path.isfile(os.path.join(path, "cardio_jepa", "__init__.py")):
                return path
            if depth >= max_depth:
                continue
            try:
                for e in os.scandir(path):
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                        stack.append((e.path, depth + 1))
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                pass
    return None


CLONE_DIR = os.path.join(WORK, "Cardio-JEPA")
# 1) attached as a dataset (or left over from an earlier session / local checkout)
PKG_ROOT = find_package_root([INPUT_ROOT, CLONE_DIR,
                              None if ON_KAGGLE else os.getcwd()])
if PKG_ROOT is not None:
    print("code: found at", PKG_ROOT)
# 2) clone from GitHub
elif "YOUR-USERNAME" not in REPO_URL:
    cmd = ["git", "clone", "--depth", "1"] + (["--branch", BRANCH] if BRANCH else []) \
        + [REPO_URL, CLONE_DIR]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        PKG_ROOT = find_package_root([CLONE_DIR])
        print("code: cloned from", REPO_URL, f"({BRANCH or 'default branch'})")
    except subprocess.CalledProcessError as exc:
        print("git clone failed:", (exc.stderr or "").strip()[-500:])
    except Exception as exc:
        print("git clone failed:", exc)
if PKG_ROOT is None:
    raise FileNotFoundError(
        "cardio_jepa not found. Either attach the repository as a Kaggle dataset "
        "(Add Input -> Datasets -> upload the repo zip), or set REPO_URL to your GitHub "
        "repository and turn Settings -> Internet -> On.")

sys.path.insert(0, PKG_ROOT)
for _m in [m for m in list(sys.modules) if m == "cardio_jepa" or m.startswith("cardio_jepa.")]:
    del sys.modules[_m]

import numpy as np
import pandas as pd
import torch
import cardio_jepa as cj
from cardio_jepa.data import discover, summarize, build_onset_dataset
from cardio_jepa.data.build import (build_pretrain_loaders, build_probe_splits,
                                    load_long_corpus, probe_batch_from_loader)
from cardio_jepa.data.forward_pairs import (build_forward_loader, combine_loaders,
                                            holdout_long_records, restrict_to_records)
from cardio_jepa.engine import budget_steps_per_epoch, estimate_runtime, report_runtime
from cardio_jepa.eval import (bootstrap_auroc_delta, compare_models,
                              run_anticipation_probe, run_classification_probe)
from cardio_jepa.eval.forecast import forecast_skill_v2
from cardio_jepa.models import CardioJEPA

pd.set_option("display.width", 160)
DEVICE = cj.pick_device()
print(f"cardio_jepa {cj.__version__} from {os.path.dirname(cj.__file__)}")
print(f"torch {torch.__version__} | device {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)}, "
         f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB)"
         if DEVICE.type == "cuda" else ""))
if DEVICE.type != "cuda" and RUN_MODE != "smoke":
    print(">>> No GPU: switch Settings -> Accelerator to a GPU, or use RUN_MODE='smoke'.")
'''),

md("## 3. Configuration"),
code(r'''
CFG = cj.lite_config() if RUN_MODE == "lite" else cj.smoke_config()
if RUN_MODE == "smoke":
    # the forward-prediction fixes, on the tiny smoke model
    CFG.loss.anti_collapse, CFG.loss.reg_target, CFG.loss.beta_reg = "vicreg", "pooled", 0.1
    CFG.optim.on_collapse = "warn"
    CFG.forward.context_windows, CFG.forward.grad_context_windows = 2, 2
CFG.data.input_root = INPUT_ROOT
CFG.out_dir = os.path.join(WORK, "runs", RUN_MODE)
CFG.optim.num_workers = 0 if RUN_MODE == "smoke" else min(4, os.cpu_count() or 2)
os.makedirs(CFG.out_dir, exist_ok=True)
RESULTS_DIR = os.path.join(WORK, "results", RUN_MODE)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS = {}


def save_results():
    scal = {}
    for k, v in RESULTS.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(RESULTS_DIR, f"{k}.csv"), index=False)
        else:
            scal[k] = v
    with open(os.path.join(RESULTS_DIR, "scalars.json"), "w") as fh:
        json.dump(scal, fh, indent=2, default=str)


cj.set_seed(CFG.optim.seed)
m = CFG.model
print(f"mode={RUN_MODE}  encoder {m.encoder_embed_dim}d x {m.encoder_depth}L, "
      f"predictor {m.predictor_embed_dim}d x {m.predictor_depth}L")
print(f"epochs {CFG.optim.epochs} x <= {CFG.optim.max_steps_per_epoch} steps, "
      f"batch {CFG.optim.batch_size}, lr {CFG.optim.lr}")
print(f"forward: context {CFG.forward.context_windows} x {CFG.signal.window_s:.0f}s "
      f"(grad on last {CFG.forward.grad_context_windows}), "
      f"target={CFG.forward.cross_target}, horizons {CFG.forward.horizons_s}")
print(f"anti-collapse: {CFG.loss.anti_collapse} on {CFG.loss.reg_target} "
      f"(beta {CFG.loss.beta_reg})")

# Resume from an earlier session: if this notebook's own output is attached as an input,
# copy its checkpoints back into runs/ so training continues where it stopped.
for p in glob.glob(os.path.join(INPUT_ROOT, "**", "runs", RUN_MODE, "*"), recursive=True):
    if p.endswith((".pt", ".pth", ".json")):
        dst = os.path.join(CFG.out_dir, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
            print("restored from previous session:", os.path.basename(p))
'''),

md("## 4. Data"),
code(r'''
FOUND = discover(INPUT_ROOT, verbose=True)
print()
print(summarize(FOUND))
if not FOUND:
    print("\n>>> No corpora attached: running on synthetic ECG. The numbers will test the "
          "pipeline, not cardiology.")
'''),
code(r'''
t0 = time.time()
LONG_RECORDS = load_long_corpus(FOUND, CFG, verbose=True)
TRAIN_IDS, HELD_IDS = holdout_long_records(LONG_RECORDS, test_frac=HOLDOUT_FRAC,
                                           seed=CFG.optim.seed)
TRAIN_LONG = [LONG_RECORDS[i] for i in TRAIN_IDS]

_src, DATA_INFO = build_pretrain_loaders(FOUND, CFG, long_records=[], verbose=True)
SHORT_LOADER = next((l for l in _src.loaders
                     if "ForwardPair" not in type(l.dataset).__name__), None)
LONG_LOADER = build_forward_loader(TRAIN_LONG, CFG,
                                   samples_per_record=200 if RUN_MODE != "smoke" else 32)
LOADER = combine_loaders(SHORT_LOADER, LONG_LOADER, long_weight=2.0)
PROBE_BATCH = probe_batch_from_loader(LOADER, n=128 if RUN_MODE != "smoke" else 32)

hours = lambda ids: sum(LONG_RECORDS[i].duration_s for i in ids) / 3600
print(f"\nready in {time.time() - t0:.0f}s")
print(f"  short clips        : {DATA_INFO.get('n_short', 0)}")
print(f"  Holter records     : {len(LONG_RECORDS)} ({hours(range(len(LONG_RECORDS))):.0f} h)")
print(f"    pretraining      : {len(TRAIN_IDS)} ({hours(TRAIN_IDS):.0f} h)")
print(f"    held out (tests) : {len(HELD_IDS)} ({hours(HELD_IDS):.0f} h)")
print(f"  loader steps/epoch : {len(LOADER)} (capped at {CFG.optim.max_steps_per_epoch})")
if not any(r.episodes for r in LONG_RECORDS):
    print(">>> No rhythm-annotated Holter records: evaluation B will be skipped.")
'''),

md(r"""
## 5. Runtime check
Times real training steps on this GPU and projects the total. With `AUTO_FIT_BUDGET`,
steps per epoch are reduced so all arms fit `TIME_BUDGET_H - EVAL_RESERVE_H`.
"""),
code(r'''
_m = CardioJEPA(CFG.signal, CFG.model, CFG.loss, CFG.hierarchical, CFG.forward)
print(f"{sum(p.numel() for p in _m.parameters() if p.requires_grad) / 1e6:.1f}M "
      f"trainable parameters per model")
EST = estimate_runtime(_m, LOADER, CFG, DEVICE, n_steps=10, n_models=len(ARMS))
del _m
if DEVICE.type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    torch.cuda.empty_cache()
TRAIN_BUDGET_H = max(0.1, TIME_BUDGET_H - EVAL_RESERVE_H - (time.time() - T0) / 3600)
print(report_runtime(EST, budget_hours=TRAIN_BUDGET_H))
if AUTO_FIT_BUDGET and EST["hours_total"] > TRAIN_BUDGET_H and RUN_MODE != "smoke":
    fits = budget_steps_per_epoch(EST["s_per_step"], CFG.optim.epochs, len(ARMS),
                                  TRAIN_BUDGET_H)
    CFG.optim.max_steps_per_epoch = fits
    print(f"\nAUTO_FIT_BUDGET: steps/epoch -> {fits} "
          f"({fits * CFG.optim.epochs} steps per model)")
RESULTS["runtime_estimate"] = dict(EST)
'''),

md(r"""
## 6. Pretraining
Both arms share the architecture, data, seed and schedule; they differ only in the
objective. Each epoch line shows `xwin` (cross-window forecast loss) and `skill`
(training-batch forecast skill against persistence). `skill` should drift above 0 for
`dual`. A rolling `<arm>_last.pt` is written every epoch.
"""),
code(r'''
import copy

def arm_cfg(name):
    c = copy.deepcopy(CFG)
    c.name = f"{CFG.name}_{name}"
    if name == "masked_only":
        c.loss.weight_forward, c.loss.lambda_mask = 0.0, 1.0
    elif name == "forward_only":
        c.loss.weight_forward, c.loss.lambda_mask = 1.0, 0.0
    else:                                             # dual
        c.loss.weight_forward, c.loss.lambda_mask = 1.0, 1.0
    return c


MODELS, TRAIN = {}, {}
for i, arm in enumerate(ARMS):
    cfg = arm_cfg(arm)
    model = CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical, cfg.forward)
    final = os.path.join(CFG.out_dir, f"{arm}.pth")
    if os.path.isfile(final):
        model.load_state_dict(torch.load(final, map_location="cpu",
                                         weights_only=False)["model"])
        model.to(DEVICE)
        print(f"[{arm}] finished checkpoint found - loaded, not retrained")
        MODELS[arm] = model
        continue
    # share what is left of the training budget among the arms still to train
    left_h = TIME_BUDGET_H - EVAL_RESERVE_H - (time.time() - T0) / 3600
    budget_s = max(60.0, left_h * 3600 / (len(ARMS) - i))
    print(f"\n=== [{arm}]  budget {budget_s / 3600:.2f} h ===")
    res = cj.pretrain(model, LOADER, cfg, device=DEVICE, arm_name=arm,
                      probe_batch=PROBE_BATCH, save_path=final,
                      resume_path=os.path.join(CFG.out_dir, f"{arm}_last.pt"),
                      time_budget_s=budget_s,
                      log_every=0 if RUN_MODE == "smoke" else 100)
    TRAIN[arm] = res
    if res.finished:
        MODELS[arm] = model
        print(f"[{arm}] done in {res.seconds / 3600:.2f} h")
    else:
        print(f"[{arm}] NOT finished (time budget). Commit this version, attach its "
              f"output as an input, and run again to resume.")

if TRAIN:
    RESULTS["training_history"] = pd.concat(
        [pd.DataFrame(r.history).assign(arm=a) for a, r in TRAIN.items()], ignore_index=True)
save_results()
print("\ntrained and ready for evaluation:", list(MODELS))
'''),

md(r"""
## 7A. Forecast skill vs persistence (held-out patients)
Everything is measured in the space the forward loss is trained in.
* `mse_skill = 1 - MSE(prediction) / MSE(persistence)`: the fraction of the
  copy-the-present error the predictor removes. **> 0 = it forecasts.**
* `skill` = centred cosine(prediction, future) - centred cosine(present, future).

`masked_only` never trains its predictor, so it sits at persistence (≈ 0) and is the
reference.
"""),
code(r'''
if MODELS and HELD_IDS:
    HS = (10.0, 60.0, 300.0) if RUN_MODE == "smoke" else (10.0, 60.0, 300.0, 600.0, 1800.0)
    rows = []
    for arm, model in MODELS.items():
        print(f"[{arm}]")
        df = forecast_skill_v2(model, LONG_RECORDS, CFG.signal, DEVICE, horizons_s=HS,
                               n_samples=64 if RUN_MODE == "smoke" else 1024,
                               context_windows=CFG.forward.context_windows,
                               record_ids=HELD_IDS, seed=CFG.optim.seed, verbose=True)
        rows.append(df.assign(arm=arm))
    SKILL = pd.concat(rows, ignore_index=True)
    RESULTS["forecast_skill"] = SKILL
    piv = SKILL.pivot(index="horizon_s", columns="arm", values="mse_skill")
    print("\nmse_skill on held-out patients:")
    display(piv.round(4))
    if "dual" in piv:
        n_pos = int((piv["dual"] > 0).sum())
        print(f"Cardio-JEPA (dual) beats persistence at {n_pos}/{len(piv)} horizons.")
    try:
        import matplotlib.pyplot as plt
        ax = piv.plot(marker="o", logx=True, figsize=(6, 3.5))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("horizon (s)"); ax.set_ylabel("mse_skill")
        ax.set_title("Forecast skill vs persistence (held-out)")
        plt.tight_layout(); plt.show()
    except Exception as exc:
        print("(plot skipped:", exc, ")")
    save_results()
else:
    print("Evaluation A skipped (no finished model or no held-out Holter records).")
'''),

md(r"""
## 7B. AF-onset anticipation (held-out patients)
Context windows contain **no AF**; label = an AF episode starts within *H* seconds.
`context` features = the encoder's pooled embedding. `world` features = that plus the
predictor's forecast latent. Linear probe trained on the pretraining patients, scored on
the held-out ones.
"""),
code(r'''
if MODELS and any(r.episodes for r in LONG_RECORDS):
    HS_B = (300.0,) if RUN_MODE == "smoke" else (300.0, 900.0)
    rows, scores, y_ref = [], {}, {}
    for H in HS_B:
        samples, stats = build_onset_dataset(
            LONG_RECORDS, CFG.signal, horizon_s=H,
            context_windows=CFG.forward.context_windows,
            stride_s=CFG.evaluation.anticipation_stride_s,
            guard_s=CFG.evaluation.anticipation_guard_s,
            max_per_record=400 if RUN_MODE != "smoke" else 60)
        tr = restrict_to_records(samples, TRAIN_IDS)
        te = restrict_to_records(samples, HELD_IDS)
        n_pos = sum(s.label > 0.5 for s in te)
        print(f"H={H:.0f}s: {len(tr)} train / {len(te)} held-out windows "
              f"({n_pos} positive)")
        if n_pos < 5 or len(te) - n_pos < 5 or not tr:
            print("  too few held-out positives/negatives - skipped")
            continue
        for arm, model in MODELS.items():
            for mode in ("context", "world"):
                r = run_anticipation_probe(model, LONG_RECORDS, tr, te, CFG.signal, DEVICE,
                                           CFG.evaluation,
                                           context_windows=CFG.forward.context_windows,
                                           feature_mode=mode, horizons_s=[H],
                                           batch_size=64, verbose=False)
                if np.isnan(r.get("auroc", float("nan"))):
                    continue
                key = f"{arm}/{mode}"
                rows.append({"horizon_s": H, "model": key, "auroc": r["auroc"],
                             "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "n_test": r["n_test"], "prevalence": r["prevalence"]})
                scores.setdefault(H, {})[key] = r["y_score"]
                y_ref[H] = r["y_true"]
                print(f"  {key:22s} AUROC {r['auroc']:.4f} "
                      f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    if rows:
        ANT = pd.DataFrame(rows)
        RESULTS["anticipation"] = ANT
        display(ANT.pivot(index="model", columns="horizon_s", values="auroc").round(4))
        for H, sc in scores.items():
            if "masked_only/context" in sc and len(sc) > 1:
                print(f"\npaired bootstrap vs masked_only/context @ H={H:.0f}s")
                display(compare_models(y_ref[H], sc, reference="masked_only/context",
                                       n=CFG.evaluation.bootstrap_n).round(4))
    save_results()
else:
    print("Evaluation B skipped (needs a finished model and rhythm-annotated Holter data).")
'''),

md(r"""
## 7C. PTB-XL / CPSC linear probe
Frozen encoder, linear head. The question is whether adding the forward objective costs
anything on standard classification (`dual` vs `masked_only`).
"""),
code(r'''
if MODELS and RUN_PTBXL_PROBE:
    PROBE_SPLITS = build_probe_splits(FOUND, CFG, verbose=True)
    rows = []
    for task, splits in PROBE_SPLITS.items():
        task_scores, y_true = {}, None
        for arm, model in MODELS.items():
            out = run_classification_probe(model, splits, DEVICE, CFG.evaluation,
                                           batch_size=64, num_workers=0, verbose=False)
            rows.append({"task": task, "arm": arm, "macro_auroc": out["macro_auroc"]})
            task_scores[arm], y_true = out["y_score"], out["y_true"]
            print(f"  {task:10s} {arm:12s} macro AUROC {out['macro_auroc']:.4f}")
        if {"dual", "masked_only"} <= set(task_scores):
            d = bootstrap_auroc_delta(y_true, task_scores["dual"], task_scores["masked_only"],
                                      n=CFG.evaluation.bootstrap_n)
            print(f"  {task}: dual - masked_only = {d['delta']:+.4f} "
                  f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p_value']:.3f}")
            RESULTS[f"probe_delta_{task}"] = d
    if rows:
        PROBE = pd.DataFrame(rows)
        RESULTS["linear_probe"] = PROBE
        display(PROBE.pivot(index="arm", columns="task", values="macro_auroc").round(4))
    save_results()
else:
    print("Evaluation C skipped.")
'''),

md("## 8. Summary"),
code(r'''
save_results()
manifest = {"run_mode": RUN_MODE, "arms": list(ARMS), "finished": list(MODELS),
            "config": CFG.to_dict(),
            "datasets": {k: v.label for k, v in FOUND.items()},
            "holter_train": [LONG_RECORDS[i].name for i in TRAIN_IDS],
            "holter_heldout": [LONG_RECORDS[i].name for i in HELD_IDS],
            "hours_used": round((time.time() - T0) / 3600, 2)}
with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2, default=str)

print("=" * 72)
print(f"CARDIO-JEPA LITE  ({RUN_MODE}, {manifest['hours_used']} h)")
print("=" * 72)
for k, v in RESULTS.items():
    if k in ("training_history", "runtime_estimate"):
        continue
    print(f"\n--- {k} ---")
    print(v.round(4).to_string() if isinstance(v, pd.DataFrame) else v)
print("\nresults ->", RESULTS_DIR, "\ncheckpoints ->", CFG.out_dir)
'''),

md(r"""
## Reading the results
* **A is the core claim.** Positive `mse_skill` for `dual` on held-out patients means the
  predictor forecasts beyond copying the present. Expect skill to shrink with horizon.
* **B:** `dual/world` > `dual/context` means the forecast latent adds anticipatory
  information. `dual` vs `masked_only` tests whether the forward objective helps.
* **C** is the cost check: `dual` should match `masked_only`.
* `smoke` mode and synthetic data only verify that the code runs.
* Lite is a 6 M-parameter, ~12k-step model. Treat it as a fast, honest test of the idea;
  paper-scale numbers come from `full_config()` on a larger GPU.
"""),
]


def build() -> dict:
    return {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "gpu", "isInternetEnabled": True,
                                    "language": "python", "sourceType": "notebook"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(REPO, "notebooks", "cardio_jepa_kaggle_lite.ipynb"))
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(build(), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("wrote", a.out, f"({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
