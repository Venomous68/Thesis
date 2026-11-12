# Sen2Fire wildfire segmentation (thesis)

Master's thesis on machine learning for Earth observation: pixel-level wildfire
segmentation on the Sen2Fire dataset (Sentinel-2, 12 bands + aerosol = 13 channels).
The user is a student. Explain steps, work one small step at a time, and do not
silently rewrite large files.

## Setup
- Windows, PowerShell, VS Code, `.venv` with Python 3.12.
- Activate: `.venv\Scripts\Activate.ps1`
- GPU: GeForce 940MX (2 GB, sm_50). Torch is `2.14.1+cu126`, installed from
  the cu126 index and pinned `<2.15`. Fallback: `torch==2.5.1` (cu124).
- Keep `requirements.txt` and the install line in NOTES.md up to date.

## Data rules (never break these)
- Scene 4 is the locked final test set. Never load, plot, tune on, or inspect it.
  Scripts must assert scenes are a subset of {1, 2, 3}.
- Evaluate by leave-one-scene-out (LOSO) over scenes 1-3.
- Do not change a frozen rule or threshold after seeing held-out results.

## Evaluation
- Score every model with `evalkit`: AP, lift, and recall at fixed FPR
  (0.125% and 1%).
- Precision, recall and F1 use a best-F1 threshold chosen on the held-out scene.
  They are optimistic. Label them as such and do not use them to rank models.
- Compare models only within the same held-out scene. AP and lift depend on
  fire prevalence.
- Run-to-run spread is about 0.04 AP (scene 3: 0.311 vs 0.274). Do not claim a
  difference smaller than that from a single seed.

## Models and scripts
- `unet_slim.py`: `UNetSlim` (base=16, 1.08M params). 128 crops, batch 8 (half
  fire crops), cross-entropy, Adam lr 1e-3 cosine, 3000 steps, seed 0.
- `unet_loso.py`: `python unet_loso.py --folds 1 2 3` (about 22 min per fold).
  `--smoke` is a quick test whose numbers are meaningless.
- Outputs go to `results/`. Checkpoints (`.pt`) and result CSVs are committed
  only after a run finishes.

## Working style
- Ask before running anything longer than a few minutes or using the GPU.
- Log every result with its settings in NOTES.md.
- Type terminal commands plainly. Avoid long one-liners.

## Boundaries
- Always: set seeds, save settings next to results, record the exact command run.
- Ask first: deleting files in `results/` or `data/`, changing the evaluation
  code, adding dependencies.
- Never: touch scene 4, report optimistic metrics as the headline, commit raw data.