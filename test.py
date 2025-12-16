# python merge_ckpt.py --params outputs/model/diff-params-ARGS=2/mat_truc/params-last.pt --seg outputs/model/diff-params-ARGS=2/mat_truc/seg-best.pt --output outputs/model/diff-params-ARGS=2/mat_truc/merged-params-best.pt
import torch

ckpt_path = "outputs/model/diff-params-ARGS=2/mat_truc/seg-best.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')
print(ckpt['epoch'])