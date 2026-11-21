import os

# PyTorch (loaded by ultralytics) and NCNN each bring an OpenMP thread pool. With the
# default busy-wait policy the two pools starve each other on a 4-core CPU (Pi 5):
# measured 18 vs 180 frames in 20 s. Idle OpenMP threads must sleep instead of spin.
# Must be set before torch/ncnn are imported; importing this package guarantees that.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
