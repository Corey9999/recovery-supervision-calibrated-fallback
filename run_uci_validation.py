"""Longitudinal real-data evaluation on the UCI Gas Sensor Array Drift Dataset.

The real sensor measurements and six gas labels are retained. Missing modalities
and reported/silent faults are injected in standardized feature space to test the
same reliability mechanism used in the controlled benchmark. The script downloads
the CC BY 4.0 dataset from UCI when no local ZIP path is supplied.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
OUT.mkdir(exist_ok=True)
UCI_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
UCI_DOI = "10.24432/C5RP6W"


@dataclass
class Config:
    modalities: int = 4
    feature_dim: int = 32
    classes: int = 6
    batch_size: int = 256
    epochs: int = 60
    lr: float = 2e-3
    weight_decay: float = 1e-4
    train_missing: float = 0.10
    test_missing: float = 0.20
    corruption_fraction: float = 0.40
    corruption_scale: float = 3.0
    seeds: tuple[int, ...] = tuple(range(101, 111))


CFG = Config()
DEVICE = torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_zip(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    cache = ROOT / "external_data" / "gas_sensor_array_drift.zip"
    cache.parent.mkdir(exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(UCI_URL, cache)
    return cache


def read_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, labels, batches = [], [], []
    with zipfile.ZipFile(path) as zf:
        for batch in range(1, 11):
            name = f"Dataset/batch{batch}.dat"
            for line in zf.read(name).decode("utf-8").splitlines():
                parts = line.split()
                x = np.zeros(128, dtype=np.float32)
                for token in parts[1:]:
                    idx, value = token.split(":", 1)
                    x[int(idx) - 1] = float(value)
                rows.append(x)
                labels.append(int(parts[0]) - 1)
                batches.append(batch)
    return np.stack(rows), np.asarray(labels, dtype=np.int64), np.asarray(batches, dtype=np.int64)


def prepare(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x, y, batches = read_dataset(path)
    train_sel, val_sel, test_sel = batches <= 6, batches == 7, batches >= 8
    scaler = StandardScaler().fit(x[train_sel])
    x = np.clip(scaler.transform(x), -8.0, 8.0).astype(np.float32)
    # Original ordering: 16 sensors with 8 aggregate response features each.
    x = x.reshape(-1, CFG.modalities, CFG.feature_dim)
    return {
        "train": (x[train_sel], y[train_sel], batches[train_sel]),
        "val": (x[val_sel], y[val_sel], batches[val_sel]),
        "test": (x[test_sel], y[test_sel], batches[test_sel]),
    }


def random_mask(n: int, rng: np.random.Generator, p: float) -> np.ndarray:
    mask = (rng.random((n, CFG.modalities)) > p).astype(np.float32)
    empty = mask.sum(1) == 0
    mask[empty, rng.integers(0, CFG.modalities, size=empty.sum())] = 1.0
    return mask


def make_view(x: np.ndarray, seed: int, regime: str, train: bool = False,
              fault_modality: int = 0, corruption_scale: float | None = None,
              corruption_fraction: float | None = None, reported_quality: float = 0.15):
    rng = np.random.default_rng(seed)
    xx = x.copy()
    q = np.ones((len(x), CFG.modalities), dtype=np.float32)
    p = CFG.train_missing if train else (0.0 if regime == "complete_drift" else CFG.test_missing)
    mask = random_mask(len(x), rng, p)
    if regime in {"reported_fault", "silent_fault"}:
        fraction = CFG.corruption_fraction if corruption_fraction is None else corruption_fraction
        scale = CFG.corruption_scale if corruption_scale is None else corruption_scale
        affected = rng.random(len(x)) < fraction
        xx[affected, fault_modality] += rng.normal(0, scale,
                                      size=(affected.sum(), CFG.feature_dim)).astype(np.float32)
        if regime == "reported_fault":
            q[affected, fault_modality] = reported_quality
    xx *= mask[:, :, None]
    return xx.astype(np.float32), mask, q


def make_paired_silent_audit(x: np.ndarray, seed: int, fault_modality: int = 0):
    """Return matched pre/post-fault views and the affected, observed samples."""
    rng = np.random.default_rng(seed)
    mask = random_mask(len(x), rng, CFG.test_missing)
    quality = np.ones((len(x), CFG.modalities), dtype=np.float32)
    affected = rng.random(len(x)) < CFG.corruption_fraction
    base = x.copy()
    fault = x.copy()
    fault[affected, fault_modality] += rng.normal(
        0, CFG.corruption_scale, size=(affected.sum(), CFG.feature_dim)).astype(np.float32)
    base *= mask[:, :, None]
    fault *= mask[:, :, None]
    valid = affected & (mask[:, fault_modality] > 0)
    return (base.astype(np.float32), mask, quality), (fault.astype(np.float32), mask, quality), valid


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(CFG.feature_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.logits = nn.Linear(32, CFG.classes)


class EarlyFusion(nn.Module):
    def __init__(self, modality_dropout: float = 0.0):
        super().__init__()
        self.drop = modality_dropout
        self.net = nn.Sequential(nn.Linear(CFG.modalities * CFG.feature_dim + 2 * CFG.modalities, 128),
                                 nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, CFG.classes))

    def forward(self, x, mask, quality):
        if self.training and self.drop > 0:
            keep = (torch.rand_like(mask) > self.drop).float() * mask
            empty = keep.sum(1) == 0
            keep[empty, 0] = mask[empty, 0]
            x = x * keep.unsqueeze(-1)
            mask = keep
        fused = self.net(torch.cat([x.reshape(len(x), -1), mask, quality], 1))
        z = torch.zeros_like(mask)
        zm = torch.zeros(len(x), CFG.modalities, CFG.classes)
        return fused, z, zm, z


class LateFusion(nn.Module):
    def __init__(self, gated=False, reliable=False, use_quality=False):
        super().__init__()
        self.gated, self.reliable, self.use_quality = gated, reliable, use_quality
        self.encoders = nn.ModuleList([Encoder() for _ in range(CFG.modalities)])
        if reliable:
            self.variance = nn.ModuleList([nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
                                           for _ in range(CFG.modalities)])
        if gated:
            self.gate = nn.Sequential(nn.Linear(CFG.modalities * 32 + 2 * CFG.modalities, 64),
                                      nn.ReLU(), nn.Linear(64, CFG.modalities))

    def forward(self, x, mask, quality):
        hs, logits = [], []
        for m, encoder in enumerate(self.encoders):
            h = encoder.body(x[:, m])
            hs.append(h)
            logits.append(encoder.logits(h))
        modality_logits = torch.stack(logits, 1)
        if self.reliable:
            logvar = torch.cat([head(h) for head, h in zip(self.variance, hs)], 1)
            strength = 0.25 if self.use_quality else 1.0
            weights = mask * torch.exp(-strength * logvar.clamp(-3, 3))
            if self.use_quality:
                weights = weights * quality.clamp(0.01, 1.0)
        elif self.gated:
            g = self.gate(torch.cat(hs + [mask, quality], 1)).masked_fill(mask == 0, -1e9)
            weights = torch.softmax(g, 1) * mask
            logvar = torch.zeros_like(mask)
        elif self.use_quality:
            weights = mask * quality.clamp(0.01, 1.0)
            logvar = torch.zeros_like(mask)
        else:
            weights = mask
            logvar = torch.zeros_like(mask)
        fused = (weights.unsqueeze(-1) * modality_logits).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)
        return fused, logvar, modality_logits, weights


def build(name: str) -> nn.Module:
    if name == "EF": return EarlyFusion(0.0)
    if name == "MD": return EarlyFusion(0.25)
    if name == "UF": return LateFusion()
    if name == "GF": return LateFusion(gated=True)
    if name in {"RWF", "CFA", "CRF_NR", "CRF"}: return LateFusion(reliable=True, use_quality=True)
    raise ValueError(name)


def ce_per(logits, y, weights):
    return F.cross_entropy(logits, y, weight=weights, reduction="none")


def fit(name, train, val, class_weights, seed, rank_weight: float = 0.15):
    set_seed(seed)
    model = build(name).to(DEVICE)
    tx, tm, tq, ty = train
    vx, vm, vq, vy = val
    ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(tm), torch.from_numpy(tq), torch.from_numpy(ty))
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    cw = torch.from_numpy(class_weights.astype(np.float32))
    best, state, patience = float("inf"), None, 0
    for _ in range(CFG.epochs):
        model.train()
        for x, mask, quality, y in loader:
            opt.zero_grad(set_to_none=True)
            fused, logvar, modality_logits, weights = model(x, mask, quality)
            loss = F.cross_entropy(fused, y, weight=cw)
            if name in {"RWF", "CFA", "CRF_NR", "CRF"}:
                per = torch.stack([ce_per(modality_logits[:, m], y, cw) for m in range(CFG.modalities)], 1)
                hetero = ((torch.exp(-logvar) * per + logvar) * mask).sum() / mask.sum().clamp_min(1)
                loss = loss + 0.20 * hetero
            if name in {"CFA", "CRF_NR", "CRF"}:
                prior = torch.tensor([0.45, 0.25, 0.18, 0.12]).unsqueeze(0)
                probs = mask * prior
                chosen = torch.multinomial(probs / probs.sum(1, keepdim=True), 1).squeeze(1)
                degraded = F.one_hot(chosen, CFG.modalities).float() * mask
                rho = 0.15 + 0.35 * torch.rand(len(x), 1)
                xcf = x.clone()
                xcf += torch.randn_like(xcf) * (2.5 * (1 - rho)).unsqueeze(-1) * degraded.unsqueeze(-1)
                qcf = quality.clone()
                reported = (torch.rand(len(x), 1) > 0.5).float()
                qfactor = reported * rho + (1 - reported)
                qcf = qcf * (1 - degraded + degraded * qfactor)
                cf_logits, cf_s, _, cf_w = model(xcf, mask, qcf)
                cf_task = F.cross_entropy(cf_logits, y, weight=cw)
                loss = loss + 0.15 * cf_task
                if name in {"CRF_NR", "CRF"}:
                    valid = mask.sum(1) > 1
                    masked = mask * (1 - degraded)
                    xm = x * masked.unsqueeze(-1)
                    masked_logits, _, _, _ = model(xm, masked, quality)
                    if valid.any():
                        consistency = F.mse_loss(torch.softmax(cf_logits[valid], 1),
                                                 torch.softmax(masked_logits[valid], 1).detach())
                        nw0 = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
                        nw1 = cf_w / cf_w.sum(1, keepdim=True).clamp_min(1e-6)
                        mono = torch.relu((nw1 * degraded).sum(1)[valid] -
                                          (nw0 * degraded).sum(1)[valid]).mean()
                    else:
                        consistency = torch.zeros((), dtype=loss.dtype)
                        mono = torch.zeros((), dtype=loss.dtype)
                    loss = loss + 0.10 * consistency + 0.05 * mono
                if name == "CRF":
                    if valid.any():
                        s0, s1 = (logvar * degraded).sum(1), (cf_s * degraded).sum(1)
                        rank = torch.relu(0.5 + s0[valid].detach() - s1[valid]).mean()
                    else:
                        rank = torch.zeros((), dtype=loss.dtype)
                    loss = loss + rank_weight * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits, _, _, _ = model(torch.from_numpy(vx), torch.from_numpy(vm), torch.from_numpy(vq))
            value = F.cross_entropy(val_logits, torch.from_numpy(vy)).item()
        if value < best - 1e-4:
            best, patience = value, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience > 8:
                break
    model.load_state_dict(state)
    return model.eval()


def fit_temperature(model, val):
    x, mask, q, y = val
    with torch.no_grad():
        logits, _, _, _ = model(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(q))
        logits = logits.numpy().astype(np.float64)
    def objective(log_t):
        z = logits / np.exp(log_t)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return log_loss(y, p, labels=np.arange(CFG.classes))
    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == y
    value = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if sel.any():
            value += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(value)


def evaluate(model, data, temperature=1.0):
    x, mask, q, y = data
    with torch.no_grad():
        logits, _, _, _ = model(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(q))
        p = torch.softmax(logits / temperature, 1).numpy()
    pred, conf = p.argmax(1), p.max(1)
    order = np.argsort(-conf)
    coverage = np.linspace(0.1, 1.0, 10)
    risks = []
    for c in coverage:
        n = max(1, int(round(c * len(y))))
        risks.append(float(np.mean(pred[order[:n]] != y[order[:n]])))
    ybin = label_binarize(y, classes=np.arange(CFG.classes))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_auroc": float(roc_auc_score(ybin, p, average="macro", multi_class="ovr")),
        "nll": float(log_loss(y, p, labels=np.arange(CFG.classes))),
        "ece": ece(y, p),
        "aurc": float(np.trapezoid(risks, coverage)),
    }


def audit_model(model, testx, seed):
    before, after, valid = make_paired_silent_audit(testx, seed)
    with torch.no_grad():
        _, s0, _, w0 = model(*(torch.from_numpy(v) for v in before))
        _, s1, _, w1 = model(*(torch.from_numpy(v) for v in after))
        nw0 = w0 / w0.sum(1, keepdim=True).clamp_min(1e-6)
        nw1 = w1 / w1.sum(1, keepdim=True).clamp_min(1e-6)
    idx = torch.from_numpy(valid)
    return {"n_affected_observed": int(valid.sum()),
            "delta_log_variance": float((s1[idx, 0] - s0[idx, 0]).mean()),
            "delta_clipped_log_variance": float((
                s1[idx, 0].clamp(-3, 3) - s0[idx, 0].clamp(-3, 3)).mean()),
            "pre_upper_clip_fraction": float((s0[idx, 0] >= 3).float().mean()),
            "post_upper_clip_fraction": float((s1[idx, 0] >= 3).float().mean()),
            "delta_normalized_weight": float((nw1[idx, 0] - nw0[idx, 0]).mean())}


def main(zip_path: Path | None = None):
    path = ensure_zip(zip_path)
    split = prepare(path)
    counts = np.bincount(split["train"][1], minlength=CFG.classes)
    class_weights = len(split["train"][1]) / (CFG.classes * counts)
    rows, temperatures, cost_rows, robustness_rows, audit_rows = [], [], [], [], []
    rank_rows, rank_audit_rows, rank_cost_rows = [], [], []
    methods = ["EF", "MD", "UF", "GF", "RWF", "CFA", "CRF_NR", "CRF"]
    robustness_methods = {"UF", "RWF", "CFA", "CRF_NR", "CRF"}
    regimes = ["complete_drift", "missing_drift", "reported_fault", "silent_fault"]
    for seed in CFG.seeds:
        tx, ty, tb = split["train"]
        vx, vy, vb = split["val"]
        testx, testy, testb = split["test"]
        train_view = (*make_view(tx, 10000 + seed, "missing_drift", train=True), ty)
        val_view = (*make_view(vx, 20000 + seed, "missing_drift", train=True), vy)
        tests = {r: (*make_view(testx, 30000 + seed, r), testy) for r in regimes}
        for name in methods:
            started = time.perf_counter()
            model = fit(name, train_view, val_view, class_weights, seed)
            cost_rows.append({"seed": seed, "model": name,
                              "train_seconds": time.perf_counter() - started,
                              "parameters": sum(p.numel() for p in model.parameters())})
            temperature = fit_temperature(model, val_view)
            temperatures.append({"seed": seed, "model": name, "temperature": temperature})
            for regime, data in tests.items():
                evaluated = evaluate(model, data, temperature)
                for metric, value in evaluated.items():
                    rows.append({"seed": seed, "model": name, "regime": regime,
                                 "metric": metric, "value": value})
                if name in {"CRF_NR", "CRF"}:
                    rank_weight = 0.0 if name == "CRF_NR" else 0.15
                    for metric in ("macro_auroc", "nll", "aurc"):
                        rank_rows.append({"seed": seed, "rank_weight": rank_weight,
                                          "regime": regime, "metric": metric,
                                          "value": evaluated[metric]})
            if name in robustness_methods:
                robustness_specs = []
                for level in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
                    robustness_specs.append(("severity", level, dict(
                        regime="silent_fault", corruption_scale=level)))
                for level in (0.10, 0.25, 0.40, 0.60, 0.80):
                    robustness_specs.append(("fraction", level, dict(
                        regime="silent_fault", corruption_fraction=level)))
                for level in range(CFG.modalities):
                    robustness_specs.append(("fault_modality", float(level), dict(
                        regime="silent_fault", fault_modality=level)))
                for level in (0.15, 0.40, 0.70, 1.00):
                    robustness_specs.append(("reported_quality", level, dict(
                        regime="reported_fault", reported_quality=level)))
                for stress_type, level, kwargs in robustness_specs:
                    view = (*make_view(testx, 40000 + seed, **kwargs), testy)
                    evaluated = evaluate(model, view, temperature)
                    for metric in ("macro_auroc", "nll", "aurc"):
                        robustness_rows.append({"seed": seed, "model": name,
                                                "stress_type": stress_type, "level": level,
                                                "metric": metric, "value": evaluated[metric]})
            if name in {"RWF", "CFA", "CRF_NR", "CRF"}:
                record = audit_model(model, testx, 50000 + seed)
                audit_rows.append({"seed": seed, "model": name, **record})
                if name in {"CRF_NR", "CRF"}:
                    rank_weight = 0.0 if name == "CRF_NR" else 0.15
                    rank_audit_rows.append({"seed": seed, "rank_weight": rank_weight, **record})

        # Rank-weight sensitivity reuses the endpoints at 0 and 0.15 above and
        # trains only the three intermediate/high settings.
        for rank_weight in (0.03, 0.07, 0.30):
            started = time.perf_counter()
            model = fit("CRF", train_view, val_view, class_weights, seed, rank_weight=rank_weight)
            rank_cost_rows.append({"seed": seed, "rank_weight": rank_weight,
                                   "train_seconds": time.perf_counter() - started})
            temperature = fit_temperature(model, val_view)
            for regime, data in tests.items():
                evaluated = evaluate(model, data, temperature)
                for metric in ("macro_auroc", "nll", "aurc"):
                    rank_rows.append({"seed": seed, "rank_weight": rank_weight,
                                      "regime": regime, "metric": metric,
                                      "value": evaluated[metric]})
            rank_audit_rows.append({"seed": seed, "rank_weight": rank_weight,
                                    **audit_model(model, testx, 50000 + seed)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "uci_results_long.csv", index=False)
    summary = result.groupby(["model", "regime", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
    summary["ci95"] = 1.96 * summary.sd / np.sqrt(summary.n)
    summary.to_csv(OUT / "uci_results_summary.csv", index=False)
    pd.DataFrame(temperatures).to_csv(OUT / "uci_temperatures.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "uci_training_costs.csv", index=False)
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(OUT / "uci_robustness_long.csv", index=False)
    robustness_summary = robustness.groupby(["model", "stress_type", "level", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
    robustness_summary["ci95"] = 1.96 * robustness_summary.sd / np.sqrt(robustness_summary.n)
    robustness_summary.to_csv(OUT / "uci_robustness_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "uci_counterfactual_audit.csv", index=False)
    rank_results = pd.DataFrame(rank_rows)
    rank_results.to_csv(OUT / "uci_rank_sensitivity_long.csv", index=False)
    rank_summary = rank_results.groupby(["rank_weight", "regime", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
    rank_summary["ci95"] = 1.96 * rank_summary.sd / np.sqrt(rank_summary.n)
    rank_summary.to_csv(OUT / "uci_rank_sensitivity_summary.csv", index=False)
    pd.DataFrame(rank_audit_rows).to_csv(OUT / "uci_rank_sensitivity_audit.csv", index=False)
    pd.DataFrame(rank_cost_rows).to_csv(OUT / "uci_rank_sensitivity_costs.csv", index=False)
    metadata = {
        "dataset": "UCI Gas Sensor Array Drift Dataset",
        "doi": UCI_DOI,
        "license": "CC BY 4.0",
        "url": UCI_URL,
        "split": {"train_batches": [1, 2, 3, 4, 5, 6], "validation_batch": [7],
                  "test_batches": [8, 9, 10]},
        "sample_counts": {k: len(v[1]) for k, v in split.items()},
        "calibration": "one scalar temperature per seed and model, fitted on validation batch 7",
        "config": CFG.__dict__,
        "methods": methods,
        "strict_ablation": {
            "CFA": "RWF plus matched counterfactual task augmentation only",
            "CRF_NR": "CFA plus consistency and weight monotonicity, without variance ranking",
            "CRF": "CRF_NR plus variance-ranking loss",
        },
        "robustness_grid": {
            "severity": [0, 1, 2, 3, 4, 5],
            "affected_fraction": [0.10, 0.25, 0.40, 0.60, 0.80],
            "fault_modality": [0, 1, 2, 3],
            "reported_quality": [0.15, 0.40, 0.70, 1.00],
        },
        "rank_weight_sensitivity": [0.0, 0.03, 0.07, 0.15, 0.30],
    }
    (OUT / "uci_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="one-seed smoke test")
    args = parser.parse_args()
    if args.quick:
        CFG.seeds = (101,)
        CFG.epochs = 20
    main(args.zip_path)
