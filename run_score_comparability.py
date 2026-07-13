"""Synthetic cross-sensor comparability audit for the bounded reliability score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_experiment as sim


OUT = sim.OUT / "source_data"


class BoundedReliabilityFusion(sim.LateFusion):
    def __init__(self):
        super().__init__(reliable=True, use_quality=True)

    def forward(self, x, mask):
        xs = x.view(-1, sim.CFG.n_modalities, sim.CFG.feature_dim)
        hs, logits = [], []
        for j, enc in enumerate(self.encoders):
            h = enc.net(xs[:, j]); hs.append(h); logits.append(enc.logit(h).squeeze(-1))
        modal_logits = torch.stack(logits, 1)
        raw = torch.cat([head(h) for head, h in zip(self.uncertainty, hs)], 1)
        score = 3.0 * torch.tanh(raw / 3.0)
        quality = xs[:, :, -1].clamp(0.01, 1.0)
        weights = mask * quality * torch.exp(-0.25 * score)
        fused = (weights * modal_logits).sum(1) / weights.sum(1).clamp_min(1e-6)
        return fused, score, modal_logits, weights


def fit(train, val, seed):
    sim.set_seed(seed)
    model = BoundedReliabilityFusion()
    tx, tm, ty = sim.flatten(train); vx, vm, vy = sim.flatten(val)
    loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(tm), torch.from_numpy(ty)),
                        batch_size=sim.CFG.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=sim.CFG.lr, weight_decay=sim.CFG.weight_decay)
    best, state, patience = float("inf"), None, 0
    for _ in range(sim.CFG.epochs):
        model.train()
        for x, mask, y in loader:
            opt.zero_grad(set_to_none=True)
            fused, score, modal_logits, weights = model(x, mask)
            per = nn.functional.binary_cross_entropy_with_logits(
                modal_logits, y[:, None].expand_as(score), reduction="none")
            bounded_aux = ((torch.exp(-score) * per + score) * mask).sum() / mask.sum().clamp_min(1)
            boundary = (((score / 3.0) ** 4) * mask).sum() / mask.sum().clamp_min(1)
            loss = nn.functional.binary_cross_entropy_with_logits(fused, y) + 0.20 * bounded_aux + 0.50 * boundary
            prior = torch.full((1, sim.CFG.n_modalities), 0.25)
            chosen = torch.multinomial(mask * prior / (mask * prior).sum(1, keepdim=True), 1).squeeze(1)
            degraded = nn.functional.one_hot(chosen, sim.CFG.n_modalities).float() * mask
            rho = 0.15 + 0.35 * torch.rand(len(x), 1)
            xr = x.view(-1, sim.CFG.n_modalities, sim.CFG.feature_dim)
            xcf = xr.clone()
            xcf[:, :, :-1] += torch.randn_like(xcf[:, :, :-1]) * (2.5 * (1 - rho)).unsqueeze(-1) * degraded.unsqueeze(-1)
            reported = (torch.rand(len(x), 1) > 0.5).float()
            xcf[:, :, -1] *= 1 - degraded + degraded * (reported * rho + 1 - reported)
            cf_logits, cf_score, _, cf_weights = model(xcf.reshape(len(x), -1), mask)
            removed = mask * (1 - degraded)
            removed_logits, _, _, _ = model((xr * removed.unsqueeze(-1)).reshape(len(x), -1), removed)
            valid = mask.sum(1) > 1
            clean_prob = torch.sigmoid(fused[valid]).detach()
            removed_prob = torch.sigmoid(removed_logits[valid]).detach()
            alpha = (1 - rho.squeeze(1)[valid]).clamp(0, 1)
            target = (1 - alpha) * clean_prob + alpha * removed_prob
            consistency = nn.functional.mse_loss(torch.sigmoid(cf_logits[valid]), target)
            w0 = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
            w1 = cf_weights / cf_weights.sum(1, keepdim=True).clamp_min(1e-6)
            monotonic = torch.relu((w1 * degraded).sum(1)[valid] - (w0 * degraded).sum(1)[valid]).mean()
            s0 = (score * degraded).sum(1); s1 = (cf_score * degraded).sum(1)
            margin = 0.5 * (1 - rho.squeeze(1)[valid]) / 0.85
            ranking = torch.relu(margin + s0[valid].detach() - s1[valid]).mean()
            loss = loss + 0.15 * nn.functional.binary_cross_entropy_with_logits(cf_logits, y)
            loss = loss + 0.10 * consistency + 0.05 * monotonic + 0.15 * ranking
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            v, _, _, _ = model(torch.from_numpy(vx), torch.from_numpy(vm))
            value = nn.functional.binary_cross_entropy_with_logits(v, torch.from_numpy(vy)).item()
        if value < best - 1e-4:
            best, patience = value, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience > 12: break
    model.load_state_dict(state)
    return model.eval()


def partial_rank_corr(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    X = np.c_[np.ones(len(z)), rz]
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


rows = []
calibration_rows = []
for seed in sim.CFG.seeds:
    train = sim.make_split(sim.CFG.n_train, 10000 + seed, "train")
    val = sim.make_split(sim.CFG.n_val, 20000 + seed, "train")
    test = sim.make_split(sim.CFG.n_test, 30000 + seed, "train")
    model = fit(train, val, seed)
    vx, vmask, _ = sim.flatten(val)
    with torch.no_grad():
        _, vscore, _, _ = model(torch.from_numpy(vx), torch.from_numpy(vmask))
    vavailable = val["mask"].astype(bool)
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        vscore.numpy()[vavailable], val["true_noise_scale"][vavailable])
    x, mask, y = sim.flatten(test)
    with torch.no_grad():
        fused, score, modal_logits, weights = model(torch.from_numpy(x), torch.from_numpy(mask))
    score = score.numpy(); available = test["mask"].astype(bool)
    noise = test["true_noise_scale"]
    ambiguity = 1.0 - 2.0 * np.abs(test["target_probability"] - 0.5)
    s, n = score[available], noise[available]
    predicted_noise = iso.predict(s)
    calibration_rows.append({"seed": seed, "metric": "isotonic_noise_mae",
                             "value": float(np.mean(np.abs(predicted_noise-n)))})
    calibration_rows.append({"seed": seed, "metric": "isotonic_noise_rmse",
                             "value": float(np.sqrt(np.mean((predicted_noise-n)**2)))})
    order = np.argsort(predicted_noise)
    bins = np.array_split(order, 10)
    sce = sum(len(b)/len(n)*abs(predicted_noise[b].mean()-n[b].mean()) for b in bins if len(b))
    calibration_rows.append({"seed": seed, "metric": "score_calibration_error_10bin",
                             "value": float(sce)})
    amb = np.repeat(ambiguity[:, None], sim.CFG.n_modalities, axis=1)[available]
    rows.append({"seed": seed, "metric": "global_spearman", "value": spearmanr(s, n).statistic})
    rows.append({"seed": seed, "metric": "global_kendall", "value": kendalltau(s, n).statistic})
    rows.append({"seed": seed, "metric": "partial_spearman_controlling_task_ambiguity",
                 "value": partial_rank_corr(s, n, amb)})
    correct, total = 0, 0
    for i in range(len(test["x"])):
        ids = np.flatnonzero(available[i])
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                j, k = ids[a], ids[b]
                correct += ((score[i, j] - score[i, k]) * (noise[i, j] - noise[i, k]) > 0)
                total += 1
    rows.append({"seed": seed, "metric": "cross_sensor_pairwise_accuracy", "value": correct / total})
    for m in range(sim.CFG.n_modalities):
        sel = available[:, m]
        rows.append({"seed": seed, "metric": f"sensor_{m+1}_spearman",
                     "value": spearmanr(score[sel, m], noise[sel, m]).statistic})
    # Cross-sensor agreement at matched true-noise deciles.
    edges = np.quantile(n, np.linspace(0, 1, 11))
    ranges = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        means = []
        for m in range(sim.CFG.n_modalities):
            sel = available[:, m] & (noise[:, m] >= lo) & (noise[:, m] <= hi)
            if sel.sum() >= 10: means.append(score[sel, m].mean())
        if len(means) > 1: ranges.append(max(means)-min(means))
    calibration_rows.append({"seed": seed, "metric": "matched_noise_between_sensor_score_range",
                             "value": float(np.mean(ranges))})
    # Score-noise association by task-ambiguity quartile.
    qcuts = np.quantile(ambiguity, np.linspace(0,1,5))
    for qid,(lo,hi) in enumerate(zip(qcuts[:-1],qcuts[1:]),1):
        selobs=(ambiguity>=lo)&(ambiguity<=hi)
        sel=(np.repeat(selobs[:,None],sim.CFG.n_modalities,axis=1)&available)
        calibration_rows.append({"seed":seed,"metric":f"ambiguity_q{qid}_spearman",
                                 "value":float(spearmanr(score[sel],noise[sel]).statistic)})

    # One unseen degraded sensor per observation; identify it from post-degradation scores.
    rng = np.random.default_rng(70000 + seed)
    xt = torch.from_numpy(x).view(-1, sim.CFG.n_modalities, sim.CFG.feature_dim)
    mt = torch.from_numpy(test["mask"])
    chosen = np.array([rng.choice(np.flatnonzero(available[i])) for i in range(len(x))])
    severity = rng.uniform(0.5, 3.0, len(x)).astype(np.float32)
    xd = xt.clone()
    torch.manual_seed(80000 + seed)
    for m in range(sim.CFG.n_modalities):
        idx = np.flatnonzero(chosen == m)
        xd[idx, m, :-1] += torch.randn((len(idx), sim.CFG.feature_dim - 1)) * torch.from_numpy(severity[idx, None])
    with torch.no_grad():
        _, post, _, _ = model(xd.reshape(len(x), -1), mt)
    post = post.numpy(); labels = np.zeros_like(post); labels[np.arange(len(x)), chosen] = 1
    rows.append({"seed": seed, "metric": "fault_sensor_identification_auroc",
                 "value": roc_auc_score(labels[available], post[available])})
    delta = post[np.arange(len(x)), chosen] - score[np.arange(len(x)), chosen]
    rows.append({"seed": seed, "metric": "severity_delta_spearman",
                 "value": spearmanr(severity, delta).statistic})

out = pd.DataFrame(rows)
out.to_csv(OUT / "score_comparability_long.csv", index=False)
summary = out.groupby("metric", as_index=False).agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
summary["ci95"] = 1.96 * summary.sd / np.sqrt(summary.n)
summary.to_csv(OUT / "score_comparability_summary.csv", index=False)
cal = pd.DataFrame(calibration_rows)
cal.to_csv(OUT / "score_calibration_long.csv", index=False)
cal.groupby("metric",as_index=False).agg(mean=("value","mean"),sd=("value","std"),n=("value","count")).to_csv(
    OUT / "score_calibration_summary.csv", index=False)
print(summary.to_string(index=False))
