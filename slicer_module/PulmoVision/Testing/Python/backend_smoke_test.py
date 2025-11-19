import os
import sys
import numpy as np

# This file lives in: slicer_module/PulmoVision/Testing/Python
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two levels: .../PulmoVision
MODULE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# Allow "from PulmoBackend..." imports
sys.path.insert(0, MODULE_DIR)

from PulmoBackend.pipeline import run_pulmo_pipeline


def main():
    # Fake CT volume: (H, W, D), roughly lung HU range
    ct_hwd = np.random.uniform(-1000, 400, size=(64, 64, 64)).astype(np.float32)

    mask = run_pulmo_pipeline(ct_hwd)

    print("Input shape:", ct_hwd.shape)
    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Unique values:", np.unique(mask))


if __name__ == "__main__":
    main()
