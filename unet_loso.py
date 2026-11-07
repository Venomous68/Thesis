import gc, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import evalkit
from unet_slim import UNetSlim

PROJECT_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = PROJECT_DIR / "data" / "sen2fire"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP = 128

def load_index():
    idx = pd.read_csv(PROJECT_DIR / "data" / "unet_index.csv")
    folder = idx.rel.str.replace("\\", "/", regex=False).str.split("/").str[0]
    idx["scene"] = folder.str.extract(r"(\d+)")[0].astype(int)
    assert set(idx.scene) <= {1, 2, 3}, "scene 4 must never be loaded here"
    return idx

def load_patch(rel):
    with np.load(EXTRACT_DIR / rel) as d:
        return d["image"], d["aerosol"], d["label"]

def build_bank(train_df, n_fire, seed, max_bg=None):
    rng = np.random.default_rng(seed)
    fire_p = train_df[train_df.fire_px > 0].reset_index(drop=True)
    bg_p = train_df[train_df.fire_px == 0].reset_index(drop=True)
    if max_bg:
        bg_p = bg_p.iloc[:max_bg]
    N = n_fire + len(bg_p)
    X = np.empty((N, 12, CROP, CROP), dtype=np.int16)
    A = np.empty((N, CROP, CROP), dtype=np.float16)
    Y = np.empty((N, CROP, CROP), dtype=np.uint8)
    per = [n_fire // len(fire_p) + (1 if i < n_fire % len(fire_p) else 0) for i in range(len(fire_p))]
    k = 0
    for rel, n in zip(fire_p.rel, per):
        if n == 0:
            continue
        img, aer, lab = load_patch(rel)
        fr, fc = np.nonzero(lab)
        for _ in range(n):
            j = rng.integers(len(fr))
            r = int(np.clip(fr[j] - rng.integers(16, CROP - 16), 0, 512 - CROP))
            c = int(np.clip(fc[j] - rng.integers(16, CROP - 16), 0, 512 - CROP))
            X[k] = img[:, r:r + CROP, c:c + CROP]
            A[k] = aer[r:r + CROP, c:c + CROP]
            Y[k] = lab[r:r + CROP, c:c + CROP]
            k += 1
    for rel in bg_p.rel:
        img, aer, lab = load_patch(rel)
        r, c = (int(v) for v in rng.integers(0, 512 - CROP + 1, size=2))
        X[k] = img[:, r:r + CROP, c:c + CROP]
        A[k] = aer[r:r + CROP, c:c + CROP]
        Y[k] = lab[r:r + CROP, c:c + CROP]
        k += 1
    assert k == N
    return {"X": X, "A": A, "Y": Y, "n_fire": n_fire}

def norm_stats(bank):
    sub = bank["X"][::6].astype(np.float32) / 10000.0
    mean, std = sub.mean(axis=(0, 2, 3)), sub.std(axis=(0, 2, 3)) + 1e-6
    a = bank["A"][::6].astype(np.float32)
    mean13 = np.concatenate([mean, [a.mean()]]).astype(np.float32).reshape(1, 13, 1, 1)
    std13 = np.concatenate([std, [a.std() + 1e-6]]).astype(np.float32).reshape(1, 13, 1, 1)
    return mean13, std13

def make_batch(bank, stats, rng, bs=8):
    X, A, Y, nf = bank["X"], bank["A"], bank["Y"], bank["n_fire"]
    ids = np.concatenate([rng.choice(nf, bs // 2, replace=False),
                          nf + rng.choice(len(X) - nf, bs - bs // 2, replace=False)])
    x = np.concatenate([X[ids].astype(np.float32) / 10000.0, A[ids].astype(np.float32)[:, None]], axis=1)
    x = (x - stats[0]) / stats[1]
    y = Y[ids].astype(np.int64)
    xs, ys = [], []
    for i in range(len(ids)):
        k = int(rng.integers(4))
        xi, yi = np.rot90(x[i], k, axes=(1, 2)), np.rot90(y[i], k)
        if rng.random() < 0.5:
            xi, yi = xi[:, :, ::-1], yi[:, ::-1]
        xs.append(np.ascontiguousarray(xi))
        ys.append(np.ascontiguousarray(yi))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

def train(bank, stats, steps, seed, ckpt):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNetSlim(n_classes=2, n_channels=13, base=16).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    every = min(100, steps)
    model.train()
    run, t0 = 0.0, time.perf_counter()
    for step in range(1, steps + 1):
        xb, yb = make_batch(bank, stats, rng)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % every == 0:
            print(f"  step {step:>4}  loss {run / every:.4f}  {time.perf_counter() - t0:.0f} s", flush=True)
            run = 0.0
    torch.save(model.state_dict(), ckpt)
    return model

def score_scene(model, stats, test_df, bsz=4):
    model.eval()
    n = len(test_df)
    S = np.empty((n, 512, 512), dtype=np.float16)
    YV = np.empty((n, 512, 512), dtype=np.uint8)
    with torch.no_grad():
        for s in range(0, n, bsz):
            items = [load_patch(r) for r in test_df.rel.iloc[s:s + bsz]]
            x = np.stack([np.concatenate([i[0].astype(np.float32) / 10000.0,
                                          i[1].astype(np.float32)[None]], axis=0) for i in items])
            x = (x - stats[0]) / stats[1]
            z = model(torch.from_numpy(x).to(DEVICE))
            S[s:s + len(items)] = (z[:, 1] - z[:, 0]).cpu().numpy().astype(np.float16)
            YV[s:s + len(items)] = np.stack([i[2] for i in items])
            if (s // bsz) % 50 == 0:
                print(f"  scored {s}/{n}", flush=True)
    return S, YV

def metrics(S, YV):
    NB = 4000
    edges = np.linspace(float(S.min()), float(S.max()), NB + 1)
    hp, hn = np.zeros(NB, np.int64), np.zeros(NB, np.int64)
    for i in range(len(S)):
        s = S[i].astype(np.float32)
        y = YV[i] > 0
        hp += np.histogram(s[y], bins=edges)[0]
        hn += np.histogram(s[~y], bins=edges)[0]
    c = 0.5 * (edges[:-1] + edges[1:])
    y_h, s_h = np.r_[np.ones(NB), np.zeros(NB)], np.r_[c, c]
    w_h = np.r_[hp, hn].astype(np.float64)
    ap = evalkit.average_precision(y_h, s_h, weight=w_h)
    prev = hp.sum() / w_h.sum()
    thr = evalkit.best_f1_threshold(y_h, s_h, weight=w_h)
    sc = evalkit.scores(evalkit.confusion(y_h, (s_h >= thr).astype(int), weight=w_h))
    score_p = np.array([np.sort(S[i].ravel().astype(np.float32))[-200:].mean() for i in range(len(S))])
    y_p = (YV.reshape(len(YV), -1).max(axis=1) > 0).astype(int)
    return {"ap": ap, "lift": ap / prev,
            "r_fpr_0125": evalkit.recall_at_fpr(y_h, s_h, 0.00125, weight=w_h),
            "r_fpr_1": evalkit.recall_at_fpr(y_h, s_h, 0.01, weight=w_h),
            "precision": sc["precision"], "recall": sc["recall"], "f1": sc["f1"],
            "patch_ap": evalkit.average_precision(y_p, score_p),
            "fire_patches": int(y_p.sum()), "patches": len(y_p)}

def run_fold(held_out, steps=3000, seed=0, smoke=False):
    assert held_out in (1, 2, 3)
    idx = load_index()
    train_df = idx[idx.scene != held_out].reset_index(drop=True)
    test_df = idx[idx.scene == held_out].reset_index(drop=True)
    if smoke:
        test_df = pd.concat([test_df.sort_values("fire_px", ascending=False).head(4),
                             test_df.sort_values("fire_px").head(4)]).reset_index(drop=True)
    print(f"fold {held_out}: train {len(train_df)} patches, test {len(test_df)} patches", flush=True)
    t0 = time.perf_counter()
    bank = build_bank(train_df, n_fire=60 if smoke else 1500, seed=seed, max_bg=60 if smoke else None)
    print(f"  bank built in {time.perf_counter() - t0:.0f} s, {len(bank['X'])} crops", flush=True)
    stats = norm_stats(bank)
    tag = f"unet_b16_s{steps}_heldout{held_out}" + ("_smoke" if smoke else "")
    model = train(bank, stats, steps, seed, PROJECT_DIR / "results" / f"{tag}.pt")
    del bank
    gc.collect()
    S, YV = score_scene(model, stats, test_df)
    row = {"held_out": held_out, "model": "U-Net (base 16)", **metrics(S, YV)}
    pd.DataFrame([row]).to_csv(PROJECT_DIR / "results" / f"{tag}_results.csv", index=False)
    print(row, flush=True)
    del model, S, YV
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    for f in a.folds:
        run_fold(f, steps=20 if a.smoke else a.steps, smoke=a.smoke)
