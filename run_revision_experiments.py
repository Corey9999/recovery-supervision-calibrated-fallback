"""Revision experiments for fault breadth, routed experts, rolling origins and bounded CRF.

This script imports the audited UCI data pipeline from ``run_uci_validation.py``
and writes separate revision outputs without overwriting the original study files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import run_uci_validation as base


OUT = Path(__file__).resolve().parent / "source_data"
METHODS = ("UF", "GF", "MOME_A", "CFA", "CRF_NR", "CRF", "BCRF")
FAULTS = ("silent_gaussian", "bias", "clipping", "burst", "correlated", "misleading_quality", "delayed_quality")
BOUNDARY_PENALTY = 0.50


class RoutedModalityExperts(nn.Module):
    """Task-adapted routed experts inspired by recent modality-expert fusion.

    Four unimodal experts and one joint expert are selected by a quality-aware
    router. This is an adaptation to tabular sensor classification, not a direct
    reproduction of a task-specific detector or segmentation architecture.
    """

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([base.Encoder() for _ in range(base.CFG.modalities)])
        joint_in = base.CFG.modalities * 32 + 2 * base.CFG.modalities
        self.joint = nn.Sequential(nn.Linear(joint_in, 64), nn.ReLU(), nn.Linear(64, base.CFG.classes))
        self.router = nn.Sequential(nn.Linear(joint_in, 64), nn.ReLU(), nn.Linear(64, base.CFG.modalities + 1))

    def forward(self, x, mask, quality):
        hs, logits = [], []
        for m, encoder in enumerate(self.encoders):
            h = encoder.body(x[:, m])
            hs.append(h)
            logits.append(encoder.logits(h))
        modality_logits = torch.stack(logits, 1)
        context = torch.cat(hs + [mask, quality], 1)
        joint_logits = self.joint(context)
        router_logits = self.router(context)
        router_logits[:, :base.CFG.modalities] = router_logits[:, :base.CFG.modalities].masked_fill(mask == 0, -1e9)
        routing = torch.softmax(router_logits, 1)
        experts = torch.cat([modality_logits, joint_logits.unsqueeze(1)], 1)
        fused = (routing.unsqueeze(-1) * experts).sum(1)
        return fused, torch.zeros_like(mask), modality_logits, routing[:, :base.CFG.modalities]


class BoundedLateFusion(base.LateFusion):
    """CRF reliability architecture with a smooth bounded log-variance head."""

    def forward(self, x, mask, quality):
        hs, logits = [], []
        for m, encoder in enumerate(self.encoders):
            h = encoder.body(x[:, m])
            hs.append(h)
            logits.append(encoder.logits(h))
        modality_logits = torch.stack(logits, 1)
        raw = torch.cat([head(h) for head, h in zip(self.variance, hs)], 1)
        logvar = 3.0 * torch.tanh(raw / 3.0)
        weights = mask * quality.clamp(0.01, 1.0) * torch.exp(-0.25 * logvar)
        fused = (weights.unsqueeze(-1) * modality_logits).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)
        return fused, logvar, modality_logits, weights


def build(name: str) -> nn.Module:
    if name == "MOME_A":
        return RoutedModalityExperts()
    if name == "BCRF":
        return BoundedLateFusion(reliable=True, use_quality=True)
    if name in {"CAGF", "DWR"}:
        return base.LateFusion(gated=True)
    return base.build(name)


def fit(name, train, val, class_weights, seed, epochs=None, degradation_prior=None,
        target_mode="removal", severity_margin=False):
    base.set_seed(seed)
    model = build(name).to(base.DEVICE)
    tx, tm, tq, ty = train
    vx, vm, vq, vy = val
    loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(tm),
                                      torch.from_numpy(tq), torch.from_numpy(ty)),
                        batch_size=base.CFG.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=base.CFG.lr, weight_decay=base.CFG.weight_decay)
    cw = torch.from_numpy(class_weights.astype(np.float32))
    best, state, patience = float("inf"), None, 0
    reliability = name in {"CFA", "CRF_NR", "CRF", "BCRF"}
    intervention = name in {"CFA", "CRF_NR", "CRF", "BCRF", "CAGF", "DWR"}
    structured = name in {"CRF_NR", "CRF", "BCRF", "CAGF", "DWR"}
    ranking = name in {"CRF", "BCRF"}
    max_epochs = base.CFG.epochs if epochs is None else epochs
    for _ in range(max_epochs):
        model.train()
        for x, mask, quality, y in loader:
            opt.zero_grad(set_to_none=True)
            fused, logvar, modality_logits, weights = model(x, mask, quality)
            loss = F.cross_entropy(fused, y, weight=cw)
            if reliability:
                per = torch.stack([base.ce_per(modality_logits[:, m], y, cw)
                                   for m in range(base.CFG.modalities)], 1)
                hetero = ((torch.exp(-logvar) * per + logvar) * mask).sum() / mask.sum().clamp_min(1)
                loss = loss + 0.20 * hetero
                if name == "BCRF":
                    # Penalize occupancy near the smooth bound without fixing a target scale.
                    loss = loss + BOUNDARY_PENALTY * (((logvar / 3.0) ** 4) * mask).sum() / mask.sum().clamp_min(1)
            if intervention:
                if degradation_prior is None:
                    degradation_prior = (0.45, 0.25, 0.18, 0.12)
                prior = torch.tensor(degradation_prior, dtype=mask.dtype).unsqueeze(0)
                probs = mask * prior
                no_eligible = probs.sum(1, keepdim=True) <= 0
                probs = torch.where(no_eligible, mask, probs)
                chosen = torch.multinomial(probs / probs.sum(1, keepdim=True), 1).squeeze(1)
                degraded = F.one_hot(chosen, base.CFG.modalities).float() * mask
                rho = 0.15 + 0.35 * torch.rand(len(x), 1)
                xcf = x + torch.randn_like(x) * (2.5 * (1 - rho)).unsqueeze(-1) * degraded.unsqueeze(-1)
                reported = (torch.rand(len(x), 1) > 0.5).float()
                qfactor = reported * rho + (1 - reported)
                qcf = quality * (1 - degraded + degraded * qfactor)
                cf_logits, cf_s, _, cf_w = model(xcf, mask, qcf)
                loss = loss + 0.15 * F.cross_entropy(cf_logits, y, weight=cw)
                valid = mask.sum(1) > 1
                if structured and valid.any():
                    masked = mask * (1 - degraded)
                    masked_logits, _, _, _ = model(x * masked.unsqueeze(-1), masked, quality)
                    clean_prob = torch.softmax(fused[valid], 1).detach()
                    removed_prob = torch.softmax(masked_logits[valid], 1).detach()
                    if target_mode == "clean":
                        target_prob = clean_prob
                    elif target_mode == "interpolation":
                        alpha = (1.0 - rho[valid]).clamp(0, 1)
                        target_prob = (1.0 - alpha) * clean_prob + alpha * removed_prob
                    elif target_mode == "teacher":
                        target_prob = 0.5 * clean_prob + 0.5 * removed_prob
                    else:
                        target_prob = removed_prob
                    consistency = F.mse_loss(torch.softmax(cf_logits[valid], 1), target_prob)
                    nw0 = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
                    nw1 = cf_w / cf_w.sum(1, keepdim=True).clamp_min(1e-6)
                    mono = torch.relu((nw1 * degraded).sum(1)[valid] -
                                      (nw0 * degraded).sum(1)[valid]).mean()
                    loss = loss + 0.10 * consistency + 0.05 * mono
                if ranking and valid.any():
                    s0 = (logvar * degraded).sum(1)
                    s1 = (cf_s * degraded).sum(1)
                    margin = 0.5 * (1.0 - rho.squeeze(1)[valid]) / 0.85 if severity_margin else 0.5
                    loss = loss + 0.15 * torch.relu(margin + s0[valid].detach() - s1[valid]).mean()
                if name == "DWR" and valid.any():
                    direct_rank = torch.relu(0.10 + (nw1 * degraded).sum(1)[valid] -
                                             (nw0 * degraded).sum(1)[valid]).mean()
                    loss = loss + 0.15 * direct_rank
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
    if state is None:
        raise RuntimeError(f"No model state selected for {name}, seed {seed}")
    model.load_state_dict(state)
    return model.eval()


def prepare_origin(path: Path, train_max: int, val_batch: int, test_batches: tuple[int, ...]):
    x, y, batches = base.read_dataset(path)
    train_sel = batches <= train_max
    val_sel = batches == val_batch
    test_sel = np.isin(batches, test_batches)
    scaler = StandardScaler().fit(x[train_sel])
    x = np.clip(scaler.transform(x), -8.0, 8.0).astype(np.float32)
    x = x.reshape(-1, base.CFG.modalities, base.CFG.feature_dim)
    return {"train": (x[train_sel], y[train_sel]),
            "val": (x[val_sel], y[val_sel]),
            "test": (x[test_sel], y[test_sel])}


def fault_view(x, seed: int, kind: str):
    rng = np.random.default_rng(seed)
    xx = x.copy()
    mask = base.random_mask(len(x), rng, base.CFG.test_missing)
    q = np.ones((len(x), base.CFG.modalities), dtype=np.float32)
    affected = rng.random(len(x)) < base.CFG.corruption_fraction
    m = 0
    if kind == "silent_gaussian":
        xx[affected, m] += rng.normal(0, 3.0, (affected.sum(), base.CFG.feature_dim)).astype(np.float32)
    elif kind == "bias":
        xx[affected, m] += 2.5
    elif kind == "clipping":
        xx[affected, m] = np.clip(xx[affected, m], -0.5, 0.5)
    elif kind == "burst":
        width = max(1, int(round(0.40 * len(x))))
        start = int(rng.integers(0, len(x) - width + 1))
        affected[:] = False
        affected[start:start + width] = True
        xx[affected, m] += rng.normal(0, 3.0, (width, base.CFG.feature_dim)).astype(np.float32)
    elif kind == "correlated":
        common = rng.normal(0, 2.5, (affected.sum(), base.CFG.feature_dim)).astype(np.float32)
        xx[affected, 0] += common
        xx[affected, 1] += common
    elif kind == "misleading_quality":
        xx[affected, m] += rng.normal(0, 3.0, (affected.sum(), base.CFG.feature_dim)).astype(np.float32)
        q[affected, 1:] = 0.20
    elif kind == "delayed_quality":
        idx = np.flatnonzero(affected)
        xx[affected, m] += rng.normal(0, 3.0, (affected.sum(), base.CFG.feature_dim)).astype(np.float32)
        q[idx[len(idx) // 2:], m] = 0.15
    else:
        raise ValueError(kind)
    xx *= mask[:, :, None]
    return xx.astype(np.float32), mask, q, affected


def boundary_audit(model, x, seed):
    before, after, valid = base.make_paired_silent_audit(x, seed)
    with torch.no_grad():
        _, s0, _, w0 = model(*(torch.from_numpy(v) for v in before))
        _, s1, _, w1 = model(*(torch.from_numpy(v) for v in after))
    idx = torch.from_numpy(valid)
    nw0 = w0 / w0.sum(1, keepdim=True).clamp_min(1e-6)
    nw1 = w1 / w1.sum(1, keepdim=True).clamp_min(1e-6)
    return {"n": int(valid.sum()),
            "delta_logvar": float((s1[idx, 0] - s0[idx, 0]).mean()),
            "near_upper_bound": float((s1[idx, 0] >= 2.9).float().mean()),
            "delta_weight": float((nw1[idx, 0] - nw0[idx, 0]).mean())}


def summarize(frame, keys):
    out = frame.groupby(keys, as_index=False).agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
    out["ci95"] = 1.96 * out.sd / np.sqrt(out.n)
    return out


def run(zip_path=None, quick=False):
    path = base.ensure_zip(zip_path)
    split0 = prepare_origin(path, 6, 7, (8, 9, 10))
    standard_rows, audit_rows, cost_rows = [], [], []
    seeds = (101,) if quick else base.CFG.seeds
    epochs = 3 if quick else base.CFG.epochs
    for seed in seeds:
        tx, ty = split0["train"]
        vx, vy = split0["val"]
        testx, testy = split0["test"]
        weights = len(ty) / (base.CFG.classes * np.bincount(ty, minlength=base.CFG.classes))
        train = (*base.make_view(tx, 10000 + seed, "missing_drift", train=True), ty)
        val = (*base.make_view(vx, 20000 + seed, "missing_drift", train=True), vy)
        for method in METHODS:
            start = time.perf_counter()
            model = fit(method, train, val, weights, seed, epochs)
            cost_rows.append({"seed": seed, "method": method, "seconds": time.perf_counter() - start,
                              "parameters": sum(p.numel() for p in model.parameters())})
            temperature = base.fit_temperature(model, val)
            for fault in FAULTS:
                xx, mm, qq, affected = fault_view(testx, 60000 + seed, fault)
                values = base.evaluate(model, (xx, mm, qq, testy), temperature)
                for metric, value in values.items():
                    standard_rows.append({"seed": seed, "method": method, "fault": fault,
                                          "metric": metric, "value": value,
                                          "affected_fraction": float(affected.mean())})
            if method in {"CRF", "BCRF"}:
                audit_rows.append({"seed": seed, "method": method,
                                   **boundary_audit(model, testx, 70000 + seed)})

    origins = ((4, 5, (6,)), (5, 6, (7,)), (6, 7, (8, 9, 10)))
    rolling_rows = []
    rolling_seeds = seeds[:1] if quick else seeds[:5]
    for origin_id, (train_max, val_batch, test_batches) in enumerate(origins, 1):
        split = prepare_origin(path, train_max, val_batch, test_batches)
        for seed in rolling_seeds:
            tx, ty = split["train"]
            vx, vy = split["val"]
            testx, testy = split["test"]
            weights = len(ty) / (base.CFG.classes * np.bincount(ty, minlength=base.CFG.classes))
            train = (*base.make_view(tx, 11000 + 100 * origin_id + seed, "missing_drift", train=True), ty)
            val = (*base.make_view(vx, 21000 + 100 * origin_id + seed, "missing_drift", train=True), vy)
            for method in ("UF", "MOME_A", "CRF", "BCRF"):
                model = fit(method, train, val, weights, seed + 1000 * origin_id, epochs)
                temperature = base.fit_temperature(model, val)
                views = {
                    "natural": (*base.make_view(testx, 31000 + 100 * origin_id + seed, "complete_drift"), testy),
                    "silent": (*fault_view(testx, 61000 + 100 * origin_id + seed, "silent_gaussian")[:3], testy),
                }
                for regime, data in views.items():
                    values = base.evaluate(model, data, temperature)
                    for metric in ("macro_auroc", "nll", "aurc"):
                        rolling_rows.append({"origin": origin_id, "train_max_batch": train_max,
                                             "validation_batch": val_batch,
                                             "test_batches": "-".join(map(str, test_batches)),
                                             "seed": seed, "method": method, "regime": regime,
                                             "metric": metric, "value": values[metric]})

    standard = pd.DataFrame(standard_rows)
    rolling = pd.DataFrame(rolling_rows)
    audit = pd.DataFrame(audit_rows)
    standard.to_csv(OUT / "revision_faults_long.csv", index=False)
    summarize(standard, ["method", "fault", "metric"]).to_csv(OUT / "revision_faults_summary.csv", index=False)
    rolling.to_csv(OUT / "revision_rolling_long.csv", index=False)
    summarize(rolling, ["origin", "method", "regime", "metric"]).to_csv(OUT / "revision_rolling_summary.csv", index=False)
    audit.to_csv(OUT / "revision_boundary_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "revision_training_costs.csv", index=False)
    metadata = {"methods": METHODS, "faults": FAULTS, "standard_seeds": list(seeds),
                "rolling_seeds": list(rolling_seeds), "origins": origins,
                "bounded_head": f"3*tanh(raw/3), with coefficient {BOUNDARY_PENALTY} on mean((s/3)^4)",
                "inference_scope": "intervals across seeds quantify optimization variability conditional on each fixed split"}
    (OUT / "revision_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summarize(standard, ["method", "fault", "metric"]).query("metric == 'macro_auroc'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.zip_path, args.quick)
