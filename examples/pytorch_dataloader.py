"""Run DAXIS straight off a PyTorch DataLoader, using a frozen ResNet-50 backbone.

This is the pattern for image domain-generalisation benchmarks (PACS,
Office-Home, Camelyon17, ...): freeze an ImageNet backbone, let DAXIS pull the
features out of your existing loader, and read the GO / NO-GO verdict.

    pip install daxis[torch] torchvision
    python examples/pytorch_dataloader.py
"""
import torch
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset

import daxis

# ---------------------------------------------------------------------------
# 1) Your data.  Any DataLoader works; here we fake one whose batches are
#    (image, label, domain) so the example runs without downloading anything.
# ---------------------------------------------------------------------------
n, n_dom = 240, 3
imgs = torch.randn(n, 3, 64, 64)
labels = torch.randint(0, 2, (n,))
# give each domain a slightly different signal so the demo is not degenerate
domains = torch.arange(n) % n_dom
imgs += (labels.float()[:, None, None, None] - 0.5) * 2.0
imgs += 0.3 * domains.float()[:, None, None, None]
loader = DataLoader(TensorDataset(imgs, labels, domains), batch_size=32)

# ---------------------------------------------------------------------------
# 2) A frozen feature extractor (ImageNet ResNet-50 with the head removed).
# ---------------------------------------------------------------------------
backbone = tv.models.resnet50(weights="IMAGENET1K_V2")
backbone.fc = torch.nn.Identity()          # -> 2048-d features
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 3) One call.  domain_fn pulls the domain out of each batch (here element 2).
#    mode="classwise" is the right choice for high-dimensional deep features.
# ---------------------------------------------------------------------------
res = daxis.from_torch(
    loader,
    backbone=backbone,
    device=device,
    domain_fn=lambda batch: batch[2],
    mode="binary",            # 2 classes here; use "classwise" for many classes
)
print(res.report())
res.plot("pytorch_daxis_matrix.png")
print("[saved] pytorch_daxis_matrix.png")
