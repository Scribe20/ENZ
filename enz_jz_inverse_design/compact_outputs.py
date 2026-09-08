"""Store hard-binary designs as uint8 (values 0/1) to keep the repository small.
Idempotent; loaders cast to float64 (torch.as_tensor(..., dtype=float64) / np.asarray > 0.5)."""
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "outputs")
n = 0
for f in root.rglob("rho_hard_binary.npy"):
    a = np.load(f)
    if a.dtype != np.uint8:
        assert set(np.unique(a)) <= {0.0, 1.0}, f
        np.save(f, a.astype(np.uint8)); n += 1
print(f"compacted {n} hard-binary files under {root}")
