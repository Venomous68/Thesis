import numpy as np, pandas as pd
from pathlib import Path

P = Path(__file__).resolve().parent
EX = P / "data" / "sen2fire"
idx = pd.read_csv(P / "data" / "unet_index.csv")
idx["scene"] = idx.rel.str.replace("\\", "/", regex=False).str.split("/").str[0].str.extract(r"(\d)")[0].astype(int)
assert set(idx.scene) <= {1, 2, 3}

defs = {
    "swir>0.5": lambda b11, b12: b12 > 0.5,
    "rule (B12>=0.10 & B12/B11>=1.15)": lambda b11, b12: (b12 >= 0.10) & (b12 / np.maximum(b11, 1e-6) >= 1.15),
}

rows = []
for sc in (1, 2, 3):
    acc = {k: dict(lab=0, hot_in=0, hot_out_fire=0, hot_out_nofire=0) for k in defs}
    for rel in idx[idx.scene == sc].rel:
        with np.load(EX / rel) as d:
            lab = d["label"].astype(bool)
            img = d["image"].astype(np.float32) / 10000.0
        b11, b12 = img[10], img[11]
        for k, f in defs.items():
            h = f(b11, b12)
            a = acc[k]
            if lab.any():
                a["lab"] += lab.sum()
                a["hot_in"] += (h & lab).sum()
                a["hot_out_fire"] += (h & ~lab).sum()
            else:
                a["hot_out_nofire"] += h.sum()
    for k, a in acc.items():
        tot = a["hot_in"] + a["hot_out_fire"] + a["hot_out_nofire"]
        tot_f = a["hot_in"] + a["hot_out_fire"]
        rows.append(dict(scene=sc, definition=k,
            hot_share_of_label=round(a["hot_in"] / max(a["lab"], 1), 4),
            inside_label_of_hot_in_fire_patches=round(a["hot_in"] / max(tot_f, 1), 3),
            inside_label_of_all_hot=round(a["hot_in"] / max(tot, 1), 3)))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(P / "results" / "hotpixel_audit.csv", index=False)