import sys, ctypes
from ctypes import wintypes

class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
k32.GetCurrentProcess.restype = wintypes.HANDLE
ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]

def mem():
    c = PMC(); c.cb = ctypes.sizeof(c)
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6, c.PeakWorkingSetSize / 1e6

n = int(sys.argv[1])
import torch
from unet_slim import UNetSlim
torch.set_num_threads(4)
base, _ = mem()
m = UNetSlim(n_classes=2, n_channels=n, base=16).eval()
x = torch.randn(1, n, 512, 512)
with torch.no_grad():
    for _ in range(5):
        m(x)
now, peak = mem()
print(f"{n:>2} ch: after imports {base:.0f} MB, after inference {now:.0f} MB, peak {peak:.0f} MB")