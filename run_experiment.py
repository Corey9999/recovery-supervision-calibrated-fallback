"""Reproducible simulation study for reliability-weighted multimodal fusion.

The benchmark is intentionally synthetic: the data-generating process exposes
the latent signal and per-modality reliability, so robustness claims can be
tested without pretending that simulated observations are clinical or field
measurements.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "source_data").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)


@dataclass
class Config:
    n_train: int = 5000
    n_val: int = 1500
    n_test: int = 3000
    n_modalities: int = 4
    feature_dim: int = 13
    latent_dim: int = 6
    epochs: int = 80
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = tuple(range(11, 31))


CFG = Config()
if os.getenv("QUICK") == "1":
    CFG.seeds = (11,)
    CFG.epochs = 50
DEVICE = torch.device("cpu")
_proj_rng = np.random.default_rng(20260710)
PROJ_A = _proj_rng.normal(scale=0.75, size=(CFG.n_modalities, CFG.feature_dim - 1, CFG.latent_dim)).astype(np.float32)
PROJ_B = _proj_rng.normal(scale=0.25, size=(CFG.n_modalities, CFG.feature_dim - 1, CFG.latent_dim)).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_split(n: int, seed: int, regime: str = "train", missing_scale: float = 1.0,
               stress_noise: float = 0.0) -> dict[str, np.ndarray]:
    """Generate paired modalities with heteroscedastic noise and MNAR dropout."""
    rng = np.random.default_rng(seed)
    m, d, k = CFG.n_modalities, CFG.feature_dim, CFG.latent_dim
    z = rng.normal(size=(n, k)).astype(np.float32)

    x = np.empty((n, m, d), dtype=np.float32)
    reliability = np.empty((n, m), dtype=np.float32)
    true_noise_scale = np.empty((n, m), dtype=np.float32)
    signal_scales = (1.35, 1.05, 0.80, 0.65)
    for j in range(m):
        q = rng.beta(5.0, 2.0, size=n).astype(np.float32)
        observed_q = q.copy()
        clean = signal_scales[j] * (z @ PROJ_A[j].T + 0.35 * np.tanh(z @ PROJ_B[j].T))
        noise_scale = 0.18 + 1.35 * (1.0 - q)
        true_noise_scale[:, j] = noise_scale
        noise = rng.normal(size=(n, d - 1)).astype(np.float32) * noise_scale[:, None]
        if regime == "shifted" and j == 0:
            noise += rng.normal(scale=1.8, size=(n, d - 1)).astype(np.float32)
            observed_q *= 0.25
        if regime == "outlier" and j == 0:
            outlier = rng.random(n) < 0.35
            noise[outlier] += rng.normal(scale=6.0, size=(outlier.sum(), d - 1)).astype(np.float32)
            observed_q[outlier] *= 0.10
        if regime == "stress" and j == 0 and stress_noise > 0:
            noise += rng.normal(scale=stress_noise, size=(n, d - 1)).astype(np.float32)
            observed_q *= 1.0 / (1.0 + stress_noise)
        if regime == "silent" and j == 0:
            # Silent failure: features degrade but the diagnostic remains
            # over-optimistic, testing reliance on metadata alone.
            noise += rng.normal(scale=2.5, size=(n, d - 1)).astype(np.float32)
        reliability[:, j] = observed_q
        x[:, j, :-1] = clean + noise
        # Sensor self-diagnostics are treated as an observed quality channel.
        x[:, j, -1] = observed_q

    # A nonlinear binary task. The label is not directly exposed to models.
    margin = 1.15 * z[:, 0] + 0.75 * z[:, 1] * z[:, 2] - 0.55 * z[:, 3]
    if regime == "shifted":
        margin = margin + 0.22 * z[:, 4]
    p = sigmoid(margin)
    y = rng.binomial(1, p).astype(np.float32)

    # Missingness is correlated with quality, creating a realistic MNAR stress.
    miss_prob = np.clip((0.08 + 0.42 * (1.0 - reliability)) * missing_scale, 0.0, 0.92)
    mask = (rng.random((n, m)) > miss_prob).astype(np.float32)
    empty = mask.sum(axis=1) == 0
    mask[empty, rng.integers(0, m, size=empty.sum())] = 1.0
    x *= mask[:, :, None]
    return {"x": x, "mask": mask, "y": y, "reliability": reliability,
            "true_noise_scale": true_noise_scale,
            "target_probability": p.astype(np.float32)}


def flatten(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = d["x"].reshape(len(d["x"]), -1)
    return x.astype(np.float32), d["mask"].astype(np.float32), d["y"].astype(np.float32)


class Encoder(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.logit = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit(self.net(x)).squeeze(-1)


class EarlyFusion(nn.Module):
    def __init__(self, training_modality_dropout: float = 0.0):
        super().__init__()
        self.drop = training_modality_dropout
        self.net = nn.Sequential(
            nn.Linear(CFG.n_modalities * (CFG.feature_dim + 1), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.drop > 0:
            keep = (torch.rand_like(mask) > self.drop).float() * mask
            keep[:, 0] = torch.maximum(keep[:, 0], (mask.sum(1) == 1).float())
            x = x.view(-1, CFG.n_modalities, CFG.feature_dim) * keep.unsqueeze(-1)
            mask = keep
        xx = torch.cat([x.view(x.shape[0], -1), mask], dim=1)
        z = torch.zeros_like(mask)
        return self.net(xx).squeeze(-1), z, z, z


class LateFusion(nn.Module):
    def __init__(self, gated: bool = False, reliable: bool = False, use_quality: bool = False):
        super().__init__()
        self.gated = gated
        self.reliable = reliable
        self.use_quality = use_quality
        self.encoders = nn.ModuleList([Encoder(CFG.feature_dim) for _ in range(CFG.n_modalities)])
        if reliable:
            self.uncertainty = nn.ModuleList([
                nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
                for _ in range(CFG.n_modalities)
            ])
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(CFG.n_modalities * (32 + 1), 64),
                nn.ReLU(),
                nn.Linear(64, CFG.n_modalities),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xs = x.view(-1, CFG.n_modalities, CFG.feature_dim)
        hs, logits = [], []
        for j, enc in enumerate(self.encoders):
            h = enc.net(xs[:, j])
            hs.append(h)
            logits.append(enc.logit(h).squeeze(-1))
        hstack = torch.cat([h for h in hs] + [mask[:, j:j + 1] for j in range(CFG.n_modalities)], dim=1)
        l = torch.stack(logits, dim=1)
        if self.reliable:
            logvar = torch.cat([head(h) for head, h in zip(self.uncertainty, hs)], dim=1)
            quality = xs[:, :, -1].clamp(0.01, 1.0)
            precision_strength = 0.25 if self.use_quality else 1.0
            weights = mask * torch.exp(-precision_strength * logvar.clamp(-3.0, 3.0))
            if self.use_quality:
                weights = weights * quality
        elif self.use_quality:
            quality = xs[:, :, -1].clamp(0.01, 1.0)
            weights = mask * quality
            logvar = torch.zeros_like(l)
        elif self.gated:
            gates = self.gate(hstack).masked_fill(mask == 0, -1e9)
            weights = torch.softmax(gates, dim=1) * mask
            logvar = torch.zeros_like(l)
        else:
            weights = mask
            logvar = torch.zeros_like(l)
        fused = (weights * l).sum(1) / weights.sum(1).clamp_min(1e-6)
        return fused, logvar, l, weights


def loss_for(model_name: str, logits: torch.Tensor, logvar: torch.Tensor,
             modality_logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fused = nn.functional.binary_cross_entropy_with_logits(logits, y)
    if model_name in {"LWF", "RWF", "CFA", "CRF_NR", "CRF"}:
        per = nn.functional.binary_cross_entropy_with_logits(
            modality_logits,
            y.unsqueeze(1).expand_as(logvar), reduction="none"
        )
        # Heteroscedastic classification objective: difficult/noisy modality
        # observations are assigned larger variance and receive less fusion weight.
        aux = (torch.exp(-logvar) * per + logvar) * mask
        return fused + 0.20 * aux.sum() / mask.sum().clamp_min(1.0)
    return fused


def fit(model_name: str, train: dict[str, np.ndarray], val: dict[str, np.ndarray], seed: int) -> nn.Module:
    set_seed(seed)
    if model_name == "EF":
        model = EarlyFusion(0.0)
    elif model_name == "MD":
        model = EarlyFusion(0.25)
    elif model_name == "UF":
        model = LateFusion(gated=False, reliable=False)
    elif model_name == "GF":
        model = LateFusion(gated=True, reliable=False)
    elif model_name == "QWF":
        model = LateFusion(gated=False, reliable=False, use_quality=True)
    elif model_name == "LWF":
        model = LateFusion(gated=False, reliable=True, use_quality=False)
    elif model_name == "RWF":
        model = LateFusion(gated=False, reliable=True, use_quality=True)
    elif model_name in {"CFA", "CRF_NR", "CRF"}:
        model = LateFusion(gated=False, reliable=True, use_quality=True)
    else:
        raise ValueError(model_name)
    model.to(DEVICE)
    tx, tm, ty = flatten(train)
    vx, vm, vy = flatten(val)
    ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(tm), torch.from_numpy(ty))
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    best, best_state, patience = float("inf"), None, 0
    for _ in range(CFG.epochs):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits, logvar, modality_logits, weights = model(xb, mb)
            loss = loss_for(model_name, logits, logvar, modality_logits, yb, mb)
            if model_name in {"CFA", "CRF_NR", "CRF"}:
                # Counterfactual sensor degradation: reduce the diagnostic
                # quality and add noise to one available modality per sample.
                xr = xb.view(-1, CFG.n_modalities, CFG.feature_dim)
                prior = torch.tensor([0.45, 0.25, 0.18, 0.12], dtype=mb.dtype).unsqueeze(0)
                choice_prob = mb * prior
                chosen = torch.multinomial(choice_prob / choice_prob.sum(1, keepdim=True), 1).squeeze(1)
                degraded = nn.functional.one_hot(chosen, CFG.n_modalities).float() * mb
                rho = 0.15 + 0.35 * torch.rand(xb.shape[0], 1)
                xcf = xr.clone()
                noise = torch.randn_like(xcf[:, :, :-1]) * (2.5 * (1.0 - rho)).unsqueeze(-1)
                xcf[:, :, :-1] = xcf[:, :, :-1] + noise * degraded.unsqueeze(-1)
                reported = (torch.rand(xb.shape[0], 1) > 0.5).float()
                q_factor = reported * rho + (1.0 - reported)
                xcf[:, :, -1] = xcf[:, :, -1] * (1.0 - degraded + degraded * q_factor)
                cf_logits, cf_logvar, cf_modality_logits, cf_weights = model(xcf.reshape(xb.shape[0], -1), mb)

                cf_task = nn.functional.binary_cross_entropy_with_logits(cf_logits, yb)
                # CFA controls for exposure to the same degraded examples.
                # CRF_NR additionally includes consistency and weight monotonicity;
                # CRF uniquely adds explicit variance ranking.
                loss = loss + 0.15 * cf_task
                if model_name in {"CRF_NR", "CRF"}:
                    valid = mb.sum(1) > 1
                    masked = mb * (1.0 - degraded)
                    xm = xr * masked.unsqueeze(-1)
                    masked_logits, _, _, _ = model(xm.reshape(xb.shape[0], -1), masked)
                    if valid.any():
                        consistency = nn.functional.mse_loss(
                            torch.sigmoid(cf_logits[valid]), torch.sigmoid(masked_logits[valid]).detach())
                        w0 = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
                        w1 = cf_weights / cf_weights.sum(1, keepdim=True).clamp_min(1e-6)
                        selected_w0 = (w0 * degraded).sum(1)
                        selected_w1 = (w1 * degraded).sum(1)
                        monotonic = torch.relu(selected_w1[valid] - selected_w0[valid]).mean()
                    else:
                        consistency = torch.zeros((), dtype=loss.dtype)
                        monotonic = torch.zeros((), dtype=loss.dtype)
                    loss = loss + 0.10 * consistency + 0.05 * monotonic
                if model_name == "CRF":
                    if valid.any():
                        s0 = (logvar * degraded).sum(1)
                        s1 = (cf_logvar * degraded).sum(1)
                        reliability_rank = torch.relu(0.50 + s0[valid].detach() - s1[valid]).mean()
                    else:
                        reliability_rank = torch.zeros((), dtype=loss.dtype)
                    loss = loss + 0.15 * reliability_rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, vv, vmod, vw = model(torch.from_numpy(vx), torch.from_numpy(vm))
            val_loss = nn.functional.binary_cross_entropy_with_logits(vl, torch.from_numpy(vy)).item()
        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def metrics(y: np.ndarray, p: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(float)
    order = np.argsort(uncertainty)
    covs = np.linspace(0.1, 1.0, 10)
    risks = []
    for c in covs:
        n = max(1, int(round(c * len(y))))
        risks.append(1.0 - correct[order[:n]].mean())
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "aurc": float(np.trapezoid(risks, covs)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "coverage_80_risk": float(risks[7]),
    }


def predict(model: nn.Module, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x, mask, _ = flatten(data)
    with torch.no_grad():
        logits, logvar, modality_logits, weights = model(torch.from_numpy(x), torch.from_numpy(mask))
        p = torch.sigmoid(logits).cpu().numpy()
        if isinstance(model, LateFusion) and model.reliable:
            xt = torch.from_numpy(x).view(-1, CFG.n_modalities, CFG.feature_dim)
            mt = torch.from_numpy(mask)
            quality = xt[:, :, -1].clamp(0.01, 1.0)
            precision = (mt * quality * torch.exp(-logvar.clamp(-3, 3))).sum(1)
            # Selective prediction uses the calibrated fused probability; the
            # modality-level precision has already affected that probability.
            uncertainty = (1.0 - np.maximum(p, 1 - p))
        else:
            uncertainty = (1.0 - np.maximum(p, 1 - p))
    return p, uncertainty


def save_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    # Store one canonical benchmark split for transparent reproduction.
    canonical = {
        "train": make_split(CFG.n_train, 1001, "train"),
        "val": make_split(CFG.n_val, 1002, "train"),
        "test_id": make_split(CFG.n_test, 1003, "train"),
        "test_shifted": make_split(CFG.n_test, 1004, "shifted"),
        "test_outlier": make_split(CFG.n_test, 1005, "outlier"),
    }
    for split, d in canonical.items():
        np.savez_compressed(OUT / "source_data" / f"{split}.npz", **d)

    rows = []
    stress_rows = []
    audit_rows = []
    cost_rows = []
    model_names = ["EF", "MD", "UF", "GF", "QWF", "LWF", "RWF", "CFA", "CRF_NR", "CRF"]
    for seed in CFG.seeds:
        train = make_split(CFG.n_train, 10000 + seed, "train")
        val = make_split(CFG.n_val, 20000 + seed, "train")
        test_sets = {
            "ID_missing": make_split(CFG.n_test, 30000 + seed, "train"),
            "noise_shift": make_split(CFG.n_test, 40000 + seed, "shifted"),
            "outlier_shift": make_split(CFG.n_test, 50000 + seed, "outlier"),
            "silent_failure": make_split(CFG.n_test, 55000 + seed, "silent"),
        }
        for model_name in model_names:
            started = time.perf_counter()
            model = fit(model_name, train, val, seed)
            cost_rows.append({"seed": seed, "model": model_name,
                              "train_seconds": time.perf_counter() - started,
                              "parameters": sum(p.numel() for p in model.parameters())})
            for regime, test in test_sets.items():
                p, u = predict(model, test)
                for k, v in metrics(test["y"], p, u).items():
                    rows.append({"seed": seed, "model": model_name, "regime": regime, "metric": k, "value": v})
                if seed == 11 and model_name in {"RWF", "CRF"} and regime == "outlier_shift":
                    observed = test["mask"].sum(1)
                    true_q = (test["reliability"] * test["mask"]).sum(1) / observed
                    pd.DataFrame({
                        "y": test["y"], "probability": p, "predicted_uncertainty": u,
                        "mean_observed_quality": true_q, "available_modalities": observed,
                    }).to_csv(OUT / "source_data" / f"{model_name.lower()}_example_predictions.csv", index=False)
            if seed == 11 and model_name in {"LWF", "RWF", "CFA", "CRF_NR", "CRF"}:
                audit = test_sets["ID_missing"]
                ax, am, ay = flatten(audit)
                xt = torch.from_numpy(ax).view(-1, CFG.n_modalities, CFG.feature_dim)
                mt = torch.from_numpy(am)
                valid = mt[:, 0] == 1
                torch.manual_seed(20260710)
                xdeg = xt.clone()
                xdeg[:, 0, :-1] += torch.randn_like(xdeg[:, 0, :-1]) * 2.5
                with torch.no_grad():
                    _, s0, _, w0 = model(xt.reshape(len(xt), -1), mt)
                    _, s1, _, w1 = model(xdeg.reshape(len(xt), -1), mt)
                    nw0 = w0 / w0.sum(1, keepdim=True).clamp_min(1e-6)
                    nw1 = w1 / w1.sum(1, keepdim=True).clamp_min(1e-6)
                idx = torch.where(valid)[0].numpy()
                for ii in idx:
                    audit_rows.append({"model": model_name, "sample": int(ii),
                                       "delta_log_variance": float(s1[ii, 0] - s0[ii, 0]),
                                       "delta_normalized_weight": float(nw1[ii, 0] - nw0[ii, 0])})
            for missing_scale in (0.5, 1.0, 1.5, 2.0):
                test = make_split(CFG.n_test, 60000 + seed + int(100 * missing_scale),
                                  "train", missing_scale=missing_scale)
                p, u = predict(model, test)
                mm = metrics(test["y"], p, u)
                for metric in ("auroc", "ece", "nll", "aurc"):
                    stress_rows.append({"seed": seed, "model": model_name, "stress_type": "missingness",
                                        "level": missing_scale, "metric": metric, "value": mm[metric]})
            for noise_level in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                test = make_split(CFG.n_test, 70000 + seed + int(100 * noise_level),
                                  "stress", stress_noise=noise_level)
                p, u = predict(model, test)
                mm = metrics(test["y"], p, u)
                for metric in ("auroc", "ece", "nll", "aurc"):
                    stress_rows.append({"seed": seed, "model": model_name, "stress_type": "sensor_noise",
                                        "level": noise_level, "metric": metric, "value": mm[metric]})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "source_data" / "results_long.csv", index=False)
    summary = results.groupby(["model", "regime", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n=("value", "count")
    )
    summary["ci95"] = 1.96 * summary["sd"] / np.sqrt(summary["n"])
    summary.to_csv(OUT / "source_data" / "results_summary.csv", index=False)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(OUT / "source_data" / "stress_results_long.csv", index=False)
    stress_summary = stress.groupby(["model", "stress_type", "level", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n=("value", "count")
    )
    stress_summary["ci95"] = 1.96 * stress_summary["sd"] / np.sqrt(stress_summary["n"])
    stress_summary.to_csv(OUT / "source_data" / "stress_results_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "source_data" / "counterfactual_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "source_data" / "training_costs.csv", index=False)
    save_json({"config": CFG.__dict__, "device": str(DEVICE), "models": model_names,
               "strict_ablation": {
                   "CFA": "RWF plus matched counterfactual task augmentation only",
                   "CRF_NR": "CFA plus consistency and weight monotonicity, without variance ranking",
                   "CRF": "CRF_NR plus variance-ranking loss"
               }}, OUT / "source_data" / "run_metadata.json")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
