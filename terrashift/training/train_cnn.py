"""Train the bi-temporal ResNet-50 with early stopping on validation macro-F1."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import SEED, CNNTrainConfig
from ..data.dataset import ChangePairDataset, train_val_split
from ..data.io import TileRecord
from ..models.resnet import BiTemporalResNet50, resolve_device
from .metrics import evaluate

log = logging.getLogger(__name__)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def train_resnet(
    root: str | Path,
    records: list[TileRecord],
    out_path: str | Path,
    cfg: CNNTrainConfig | None = None,
) -> tuple[BiTemporalResNet50, dict]:
    cfg = cfg or CNNTrainConfig()
    torch.manual_seed(SEED)
    device = resolve_device(cfg.device)
    train_recs, val_recs = train_val_split(records, cfg.val_fraction, SEED)

    train_ds = ChangePairDataset(root, train_recs, train=True, seed=SEED)
    val_ds = ChangePairDataset(root, val_recs, train=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = BiTemporalResNet50(pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1, best_state, best_metrics, stale = -1.0, None, {}, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        y_true, y_pred = predict_loader(model, val_dl, device)
        metrics = evaluate(y_true, y_pred)
        log.info("epoch %d loss=%.4f val_acc=%.4f val_macro_f1=%.4f", epoch, total / max(n, 1), metrics["accuracy"], metrics["macro_f1"])

        if metrics["macro_f1"] > best_f1:
            best_f1, best_metrics, stale = metrics["macro_f1"], metrics, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().save(out_path, extra={"val_metrics": best_metrics})
    model.eval()
    return model, best_metrics
