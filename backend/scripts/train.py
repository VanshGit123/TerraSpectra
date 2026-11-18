from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from app.data import make_synthetic_cube
from app.model import TerraSpectraHybrid


def main():
    parser = argparse.ArgumentParser(description="Train TerraSpectra demo segmentation model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cube, mask = make_synthetic_cube(seed=args.seed)
    # Train on overlapping patches to keep Transformer attention memory bounded.
    patch_size, stride = 32, 16
    patches_x, patches_y = [], []
    for row in range(0, cube.shape[0] - patch_size + 1, stride):
        for col in range(0, cube.shape[1] - patch_size + 1, stride):
            patches_x.append(cube[row:row+patch_size, col:col+patch_size, :])
            patches_y.append(mask[row:row+patch_size, col:col+patch_size])
    x = torch.from_numpy(np.stack(patches_x)).permute(0, 3, 1, 2).unsqueeze(1)
    y = torch.from_numpy(np.stack(patches_y)).float().unsqueeze(1)
    dataset = TensorDataset(x, y)
    # Tiny synthetic patches are for an end-to-end engineering smoke test only.
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TerraSpectraHybrid().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            logits = model(x[:2].to(device)); pred = torch.sigmoid(logits) >= .5
            truth = y[:2].to(device) >= .5
            tp = (pred & truth).sum().item(); fp = (pred & ~truth).sum().item(); fn = (~pred & truth).sum().item()
            iou = tp / max(tp + fp + fn, 1)
        print(f"epoch {epoch+1:02d}/{args.epochs} loss={np.mean(losses):.4f} demo_iou={iou:.4f}")
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    os.makedirs(out, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(out, "terraspectra_model.pt"))
    print(f"Saved checkpoint: {os.path.join(out, 'terraspectra_model.pt')}")

if __name__ == "__main__": main()
