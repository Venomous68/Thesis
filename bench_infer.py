import time, statistics, torch
from unet_slim import UNetSlim

torch.set_num_threads(4)
dev = "cpu"

def bench(n_ch, runs=15, warm=3):
    m = UNetSlim(n_classes=2, n_channels=n_ch, base=16).to(dev).eval()
    params = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, n_ch, 512, 512)
    ts = []
    with torch.no_grad():
        for i in range(warm + runs):
            t0 = time.perf_counter()
            m(x)
            dt = time.perf_counter() - t0
            if i >= warm:
                ts.append(dt)
    return params, statistics.median(ts), min(ts), max(ts)

for n in (13, 12, 4):
    p, med, lo, hi = bench(n)
    print(f"{n:>2} ch: params {p:,}  median {med:.3f} s  min {lo:.3f}  max {hi:.3f}")