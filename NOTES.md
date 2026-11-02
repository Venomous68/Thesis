# Sen2Fire thesis - working notes

Last updated: 2026-10-28

## 1. Project layout

- 01_data_inspection.ipynb - zip, extraction, keys/shapes, label scan, visual checks, splits check, ratio exploration
- 02_baseline_threshold.ipynb - histogram cache, threshold tuning, frozen rule, validation, size analysis
- 03_random_forest.ipynb - pixel sampling, Random Forest, feature-set comparison
- data/ - raw zip, extracted patches, reference code, caches (baseline_hist.npz, rf_samples.npz). Not for git.
- results/ - frozen_rule_v1.json, baseline_threshold_results.csv, rf_pixel_results.csv, rf_pixel_results_scene_tuned.csv
- requirements.txt - pinned direct dependencies

## 2. Dataset facts (verified)

- Source: Zenodo record 10881058, Sen2Fire.zip, MD5 matches the published value.
- 2,466 .npz patches: scene1 864, scene2 594, scene3 504, scene4 504.
- Keys per file: image (12, 512, 512) int16; aerosol (512, 512) float32; label (512, 512) uint8 with values {0, 1}.
- Image / 10,000 = reflectance. Channel 3 = red, 7 = NIR, 10 = SWIR (B11-like), 11 = SWIR (B12-like). RGB = channels 3, 2, 1.
- Aerosol is a separate key; it is the 13th channel only in the authors' "aerosol" modes. Values were negative in the sample patch; units unknown.
- Fire patches: 349 of 2,466 (14.2%). Scene 1: 115, scene 2: 38, scene 3: 94, scene 4: 102.
- Fire pixels: scene 1 7,779,854 (3.43%); scene 2 1,784,143 (1.15%); scene 3 4,694,012 (3.55%); scene 4 7,632,679 (5.78%).
- Splits follow the authors' lists: train = scenes 1-2 (1,458 patches), val = scene 3 (504), test = scene 4 (504). No overlaps, no missing files.

## 3. Label characteristics

- Labels are blocky, often rectangles (about 45% of 60 sampled fire patches have bounding-box fill >= 0.99; median fill 0.948).
- Labels cover the fire plus much surrounding area; hot pixels are a small fraction of label pixels (<= ~7% in the sampled patches).
- Some labelled patches show no visible fire (2 of 6 inspected). One strip label (scene3 patch 3_24) shows no spectral contrast against the pixels below it.
- About 1/3 of hot pixels fall outside labels in the sample (e.g. scene 3 patches 10_2, 10_34); cause unknown.

## 4. Evaluation protocol

- Tune only on the training scenes. Validation (scene 3) is looked at once per frozen method. The test scene (scene 4) is untouched until the final evaluation.
- Pixel level: precision, recall, F1, IoU of the fire class. Reference: "flag everything" F1 = 2p/(1+p) (train 0.049, val 0.069).
- Patch level: patch is fire if it has any fire label pixel; prediction needs >= k hot pixels.
- Average precision (AP) is the main metric for comparing feature sets; F1 depends on the chosen cutoff and on scene fire rate.

## 5. Baseline: spectral threshold rule (frozen, results/frozen_rule_v1.json)

- B12/B11 > 1 separates flaming pixels from bright bare ground (ground median ratio about 0.6-0.7).
- Pixel level (B12 >= 0.10, ratio >= 1.15), validation: precision 0.508, recall 0.035, F1 0.066, IoU 0.034. Flag-all F1 0.069. Precision is about 14x the base rate, but hot pixels cover only about 3.5% of label area.
- Patch level, tuned on train only: fire if >= 200 pixels have B12 >= 0.24 and B12/B11 >= 1.45. Train F1 0.603; validation precision 0.757, recall 0.596, F1 0.667 (flag-all F1 0.314; TP 56, FP 18, FN 38, TN 392).
- F1 optimum at k = 200 is interior (k = 300 ... 5000 decrease monotonically); F1 is flat across B12 / ratio settings.
- Recall by label size on train fire patches: < 5,000 px 0/20 detected (median hot px 0); 5,000-20,000 px 12/33 (0.36, median 6 hot px); > 20,000 px 58/100 (0.58, median 596).
- Train false alarms: 9 of 1,305 non-fire patches (0.7%), median 320 hot px (possibly unlabelled fire; unchecked).

## 6. Random Forest (per pixel)

- Setup: 100 trees, max_depth 20, min_samples_leaf 10, max_samples 0.3. Train sample: up to 1,000 fire + 1,000 non-fire pixels per patch. Validation: 2,000 uniformly random pixels per patch (unbiased at natural fire rate).
- Features: 12 bands, B12/B11 ratio, aerosol. Threshold chosen with natural-prevalence weights on held-out data.
- Random 20% patch holdout, validation: 12 bands F1 0.132 / AP 0.073; + ratio F1 0.142 / AP 0.092; + ratio + aerosol F1 0.099 / AP 0.072.
- Scene-tuned protocol (train scene 1, tune on scene 2): AP_val 0.074 / 0.088 / 0.074 and AP_scene2 0.020 / 0.031 / 0.036 for the same three feature sets; F1 0.122 / 0.109 / 0.110.
- The ratio improves AP in both scenes (+0.014 val, +0.011 scene 2): keep it.
- Aerosol has the highest RF importance (0.34) but its effect on AP is scene-dependent: no reliable benefit.
- The F1-optimal cutoff varies from 0.37 to 0.58 across runs; use AP to compare feature sets.
- RF flags about 5x as many pixels as are fire (recall / precision), i.e. diffuse areas around the labels.
- Single seed and single validation scene: differences of about 0.01 are suggestive only.
- RF example maps (12 bands + ratio, thr 0.37, val scene 3; results/rf_examples_val.png):
  patch 2_24: P 0.89 R 0.34 (label covers unburnt forest; FN on left half)
  patch 7_21: P 0.52 R 0.40 (fire spot found; possible label omission at a label notch)
  patch 14_26: P 0.15 R 0.48 (large FP over hillsides)
  patch 6_22: P 0.23 R 0.77, threshold rule finds no hot pixels here
  patch 4_29 (no fire): large smooth FP region; patch 8_27 (no fire): nearly clean
- Failure mode: broad, smooth false alarms (not fire fronts). Hypothesis: confusion with some land-cover/terrain type. Untested.
- Pixel diagnostics (val, RF 12 bands + ratio, thr 0.37; medians; 6 hand-picked patches):
  TP: SWIR2 0.085, B12/B11 0.665, NBR 0.374, aerosol 2.91 (not flame-like; ratio < 1)
  FP patch 4_29: SWIR2 0.086, ratio 0.672, NBR 0.366, aerosol 1.40 (spectrally same as TP)
  FP patch 14_26: SWIR2 0.106, ratio 0.569, NBR 0.316, aerosol 1.30 (between TN and TP)
  FN inside labels: NDVI 0.397, NBR 0.464 (healthy vegetation in label blocks)
  TN: SWIR2 0.114, ratio 0.623, NBR 0.248, aerosol 1.40
- Per-pixel spectra cannot separate the 4_29 false alarms from detected fire -> spatial context needed.
- Aerosol in val: ~2.8-2.9 in fire patches vs ~1.4 elsewhere (smoke-like), but scene-1 non-fire patch had mean -1.08:
  hypothesis = scene-dependent baseline explains aerosol failing to transfer. TO CHECK (per-scene aerosol medians).
  If confirmed: test aerosol minus scene median as a feature.
  - Aerosol per-patch medians (train + val only; test not used):
  scene1: non-fire -0.77 (n=749), fire 1.14 (n=115)
  scene2: non-fire -0.33 (n=556), fire -0.18 (n=38)  -> almost no separation
  scene3: non-fire  0.67 (n=410), fire 2.33 (n=94)
- Non-fire baseline shifts by scene (-0.77, -0.33, +0.67): absolute aerosol thresholds do not transfer across scenes. Likely explanation for aerosol failing in RF.
- Many non-fire patches have high aerosol (p90 3.6 in scene 1, 3.1 in scene 3): regional smoke, not a per-patch fire indicator.
- TODO next session: RF with aerosol minus scene median (label-free statistic); check that this is allowed at test time (needs whole test scene's aerosol).

## 7. Open questions / TODO

- Patch-level bootstrap over validation patches for the ratio gain and the aerosol effect.
- Patch-level evaluation of the Random Forest (aggregate pixel predictions) to compare with the threshold rule (patch F1 0.667).
- Meaning and units of the aerosol channel; check authors' mode definitions.
- Inspect the 9 train false-alarm patches and the missed fire patches (possible label problems).
- U-Net with the authors' normalisation (Max_values) and the same train/val/test protocol.
- Final one-time evaluation on the test scene (scene 4) for: threshold rule, Random Forest, U-Net.
