import torch

ckpt_path = "outputs/model/diff-params-ARGS=1/metal_nut/params-last.pt"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint keys:", list(ckpt.keys()))
except Exception as e:
    print(f"Failed to load checkpoint from {ckpt_path}: {e}")