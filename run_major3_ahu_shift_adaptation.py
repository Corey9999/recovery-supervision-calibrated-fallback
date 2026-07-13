"""Cross-building AHU shift diagnosis and low-cost generalization/adaptation.

The script separates:
1) source-only zero-shot ERM and GroupDRO;
2) unsupervised target normalization and target-to-source CORAL, both of which
   use target-building feature statistics but no target labels;
3) a task-specific HGB diagnostic baseline.

No target label is used for model selection, normalization or alignment.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import run_ahu_field_validation as ahu
import run_ahu_temporal_validation as temporal
import run_major_revision_experiments as major
import run_q1_risk_sensitive as risk

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"source_data"
QUICK = os.getenv("M3_AHU_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 106))
EPOCHS = 3 if QUICK else 12
if QUICK:
    ahu.QUICK = True

RO_NOQ = replace(
    major.SPECS["PDRF"], use_quality=False, train_quality=False,
    recovery_distillation=True, w_recovery=.30, brier_regularization=True,
    interior_barrier=True, w_interior=.005, degraded_auc=True,
)
EF = major.SPECS["EF_PD"]


def add_domain(data, domain):
    out = dict(data)
    out["domain"] = np.full(len(data["y"]), domain, np.int64)
    return out


def flatten_view(view):
    x, mask, _, _ = view
    return np.concatenate([x.reshape(len(x), -1), mask], axis=1).astype(np.float32)


def safe_ece(y, p, bins=15):
    return risk.ece(y, p, bins)


def all_binary_metrics(y, p):
    out = major.metrics(y, p)
    onehot = np.eye(2)[y]
    out["brier"] = float(np.square(p-onehot).sum(1).mean())
    out["ece15"] = safe_ece(y, p)
    out["reversed_auroc"] = float(roc_auc_score(y, 1-p[:, 1]))
    return out


def matrix_sqrt(a, inverse=False, eps=1e-4):
    values, vectors = eigh((a+a.T)/2)
    values = np.clip(values, eps, None)
    power = -0.5 if inverse else 0.5
    return (vectors*(values**power))@vectors.T


def coral_target_to_source(source_flat, target_flat):
    """Align unlabeled target covariance to the source feature space."""
    ms, mt = source_flat.mean(0), target_flat.mean(0)
    cs = np.cov(source_flat, rowvar=False) + np.eye(source_flat.shape[1])*1e-3
    ct = np.cov(target_flat, rowvar=False) + np.eye(target_flat.shape[1])*1e-3
    aligned = (target_flat-mt)@matrix_sqrt(ct, inverse=True)@matrix_sqrt(cs) + ms
    return aligned.astype(np.float32)


def transformed_target(raw, source_mean, source_std, regime, source_train_flat):
    if regime == "source_norm":
        return ahu.view(raw, source_mean, source_std)
    if regime == "target_norm":
        tm, ts = ahu.fit_scaler(raw["x"])
        return ahu.view(raw, tm, ts)
    if regime == "coral_adapt":
        base_view = ahu.view(raw, source_mean, source_std)
        flat = base_view[0].reshape(len(raw["y"]), -1)
        aligned = coral_target_to_source(source_train_flat, flat)
        x = aligned.reshape(-1, 5, 4)
        x *= raw["mask"][:, :, None]
        return (x.astype(np.float32), raw["mask"].astype(np.float32),
                np.ones_like(raw["mask"], np.float32), raw["y"].astype(np.int64))
    raise ValueError(regime)


def fit_groupdro_early(train, select, domains, class_weights, seed):
    major.set_seed(seed)
    tx, tm, tq, ty = train
    vx, vm, vq, vy = select
    model = major.Fusion(tx.shape[1], tx.shape[2], 2, EF)
    loader = DataLoader(TensorDataset(
        torch.from_numpy(tx), torch.from_numpy(tm), torch.from_numpy(tq),
        torch.from_numpy(ty), torch.from_numpy(domains)),
        batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cw = torch.from_numpy(class_weights.astype(np.float32))
    q = torch.ones(int(domains.max())+1)/float(int(domains.max())+1)
    best, state, stale = float("inf"), None, 0
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        for x, mask, quality, y, domain in loader:
            opt.zero_grad(set_to_none=True)
            clean = model(x, mask, quality)[0]
            chosen = torch.multinomial(mask/mask.sum(1, keepdim=True), 1).squeeze(1)
            degraded = F.one_hot(chosen, mask.shape[1]).float()*mask
            rho = .15+.35*torch.rand(len(x), 1)
            x2 = x+torch.randn_like(x)*(2.5*(1-rho)).unsqueeze(-1)*degraded.unsqueeze(-1)
            fault = model(x2, mask, quality)[0]
            per = F.cross_entropy(clean, y, weight=cw, reduction="none")
            per = per+.15*F.cross_entropy(fault, y, weight=cw, reduction="none")
            losses, present = [], []
            for g in range(len(q)):
                take = domain == g
                if take.any():
                    losses.append(per[take].mean()); present.append(g)
            lv = torch.stack(losses)
            with torch.no_grad():
                ids = torch.tensor(present, dtype=torch.long)
                q[ids] *= torch.exp(.05*lv.detach().clamp(max=10))
                q /= q.sum()
            loss = torch.stack([q[g]*v for g, v in zip(present, losses)]).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = F.cross_entropy(model(torch.from_numpy(vx), torch.from_numpy(vm),
                                        torch.from_numpy(vq))[0],
                                  torch.from_numpy(vy)).item()
        if val < best-1e-4:
            best, stale = val, 0
            state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale > 8:
                break
    model.load_state_dict(state)
    return model.eval(), {"train_seconds":time.perf_counter()-t0,
                          "parameters":sum(p.numel() for p in model.parameters()),
                          "epochs":epoch+1, "final_domain_weights":q.numpy().tolist()}


def distribution_audit(source_raw, target_raw, held_out):
    rows = []
    sx, tx = source_raw["x"], target_raw["x"]
    names = {
        (0,0):"set_point_temperature", (1,0):"return_temperature",
        (2,0):"supply_air_temperature", (3,0):"supply_fan",
        (3,1):"valve_position", (4,0):"heating_supply_temperature",
        (4,1):"cooling_supply_temperature", (4,2):"heating_pump",
        (4,3):"cooling_pump",
    }
    rng = np.random.default_rng(91001)
    for group in range(5):
        for pos in range(4):
            s = sx[:, group, pos]; t = tx[:, group, pos]
            s = s[np.isfinite(s)]; t = t[np.isfinite(t)]
            if not len(s) or not len(t):
                continue
            ss = rng.choice(s, min(len(s), 10000), replace=False)
            tt = rng.choice(t, min(len(t), 10000), replace=False)
            pooled = np.sqrt((np.var(ss)+np.var(tt))/2)+1e-8
            q1, q99 = np.quantile(ss, [.01, .99])
            rows.append({
                "held_out_building":held_out, "group":group+1, "position":pos+1,
                "feature":names.get((group,pos), f"group_{group+1}_position_{pos+1}"),
                "source_n":len(s), "target_n":len(t),
                "source_mean":np.mean(ss), "target_mean":np.mean(tt),
                "source_sd":np.std(ss), "target_sd":np.std(tt),
                "source_q05":np.quantile(ss,.05), "target_q05":np.quantile(tt,.05),
                "source_median":np.median(ss), "target_median":np.median(tt),
                "source_q95":np.quantile(ss,.95), "target_q95":np.quantile(tt,.95),
                "standardized_mean_difference":(np.mean(tt)-np.mean(ss))/pooled,
                "ks_statistic":ks_2samp(ss, tt).statistic,
                "wasserstein":wasserstein_distance(ss, tt),
                "target_outside_source_1_99":np.mean((tt<q1)|(tt>q99)),
                "source_missing_rate":1-len(s)/len(sx),
                "target_missing_rate":1-len(t)/len(tx),
            })
    return rows


def global_shift(source_view, target_view, held_out):
    rng = np.random.default_rng(91002)
    s = flatten_view(source_view); t = flatten_view(target_view)
    s = s[rng.choice(len(s), min(len(s), 2000), replace=False)]
    t = t[rng.choice(len(t), min(len(t), 2000), replace=False)]
    sm, tm = s.mean(0), t.mean(0)
    cs, ct = np.cov(s, rowvar=False), np.cov(t, rowvar=False)
    coral = np.linalg.norm(cs-ct, "fro")/(4*cs.shape[0]**2)
    linear_mmd = np.square(sm-tm).sum()
    probe = np.vstack([s[:500], t[:500]])
    dist = cdist(probe, probe, "sqeuclidean")
    med = np.median(dist[dist>0]); gamma = 1/max(med, 1e-8)
    kss = np.exp(-gamma*cdist(s, s, "sqeuclidean"))
    ktt = np.exp(-gamma*cdist(t, t, "sqeuclidean"))
    kst = np.exp(-gamma*cdist(s, t, "sqeuclidean"))
    rbf_mmd = ((kss.sum()-np.trace(kss))/(len(s)*(len(s)-1))+
               (ktt.sum()-np.trace(ktt))/(len(t)*(len(t)-1))-2*kst.mean())
    both = np.vstack([s, t]); domain = np.r_[np.zeros(len(s)), np.ones(len(t))]
    emb = PCA(n_components=2, random_state=91002).fit_transform(both)
    ids = np.r_[np.arange(len(s)), np.arange(len(t))]
    pca = pd.DataFrame({"held_out_building":held_out, "domain":np.where(domain, "target", "source"),
                        "sample":ids, "pc1":emb[:,0], "pc2":emb[:,1]})
    summary = {"held_out_building":held_out, "linear_mmd":linear_mmd,
               "rbf_mmd":rbf_mmd, "coral_distance":coral}
    return summary, pca


def append_metrics(rows, held_out, seed, method, regime, raw, p, cost):
    y = raw["y"]
    for metric, value in all_binary_metrics(y, p).items():
        rows.append({"held_out_building":held_out, "seed":seed, "method":method,
                     "regime":regime, "metric":metric, "value":value})
    pred = p.argmax(1)
    for subtype in ("Return air temperature fault", "Supply air temperature fault"):
        take = raw["subtype"] == subtype
        rows.append({"held_out_building":held_out, "seed":seed, "method":method,
                     "regime":regime, "metric":"recall_"+subtype.lower().replace(" ","_"),
                     "value":float((pred[take]==1).mean()) if take.any() else np.nan})
    conf = p.max(1); correct = pred == y
    order = np.argsort(-conf)
    for coverage in (.5, .8, 1.0):
        n = max(1, int(coverage*len(y))); take = order[:n]
        rows.append({"held_out_building":held_out, "seed":seed, "method":method,
                     "regime":regime, "metric":f"selective_error_at_{coverage:g}",
                     "value":float(1-correct[take].mean())})
    for key, value in cost.items():
        if key == "final_domain_weights":
            continue
        rows.append({"held_out_building":held_out, "seed":seed, "method":method,
                     "regime":regime, "metric":key, "value":value})


def main():
    OUT.mkdir(exist_ok=True)
    raw = {name:ahu.load_building(name) for name in ahu.BUILDINGS}
    metric_rows, feature_rows, shift_rows, prior_rows, pca_parts, pred_rows = [], [], [], [], [], []
    for held_out in ahu.BUILDINGS:
        source_names = [n for n in ahu.BUILDINGS if n != held_out]
        source_parts = [add_domain(raw[n], i) for i, n in enumerate(source_names)]
        source_all = ahu.concatenate(source_parts)
        train_raw, select_raw, cal_raw = ahu.split_sources(source_all)
        sm, ss = ahu.fit_scaler(train_raw["x"])
        train, select, calibration = [ahu.view(x, sm, ss) for x in (train_raw, select_raw, cal_raw)]
        source_train_flat = train[0].reshape(len(train[0]), -1)
        tests = {r:transformed_target(raw[held_out], sm, ss, r, source_train_flat)
                 for r in ("source_norm", "target_norm", "coral_adapt")}
        cw = len(train[3])/(2*np.bincount(train[3], minlength=2))

        feature_rows.extend(distribution_audit(train_raw, raw[held_out], held_out))
        gshift, pca = global_shift(train, tests["source_norm"], held_out)
        shift_rows.append(gshift); pca_parts.append(pca)
        for scope, data in (("source_pool", source_all), ("source_train", train_raw),
                            ("target_test", raw[held_out])):
            prior_rows.append({
                "held_out_building":held_out, "scope":scope, "n":len(data["y"]),
                "fault_prevalence":float(data["y"].mean()),
                "return_fault_prevalence":float(np.mean(data["subtype"]=="Return air temperature fault")),
                "supply_fault_prevalence":float(np.mean(data["subtype"]=="Supply air temperature fault")),
                "fan_on_rate":float(np.nanmean(data["x"][:,3,0] > 0)),
            })

        ensemble = {}
        for seed in SEEDS:
            for method, spec in (("EF-PD", EF), ("RO-PDRF-NOQ", RO_NOQ)):
                model, cost = major.fit(spec, train, select, cw, seed,
                                        prior=tuple([.2]*5), epochs=EPOCHS)
                temp = major.fit_temperature(model, calibration, neutral_q=(method!="EF-PD"))
                for regime, test in tests.items():
                    p, _, _, _, infer = major.predict(model, *test[:3], temp)
                    c = {**cost, "inference_ms_per_observation":1000*infer/len(test[3])}
                    append_metrics(metric_rows, held_out, seed, method, regime, raw[held_out], p, c)
                    ensemble.setdefault((method, regime), []).append(p)

            gdro, cost = fit_groupdro_early(train, select, train_raw["domain"], cw, seed)
            temp = major.fit_temperature(gdro, calibration)
            p, _, _, _, infer = major.predict(gdro, *tests["source_norm"][:3], temp)
            c = {**cost, "inference_ms_per_observation":1000*infer/len(p)}
            append_metrics(metric_rows, held_out, seed, "GroupDRO-EF-PD",
                           "source_only", raw[held_out], p, c)
            ensemble.setdefault(("GroupDRO-EF-PD", "source_only"), []).append(p)

            # Source-only task-specific supervised diagnostic baseline.
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                max_leaf_nodes=31, l2_regularization=1e-3, random_state=seed,
                early_stopping=True)
            trf = flatten_view(train); caf = flatten_view(calibration)
            tef = flatten_view(tests["source_norm"])
            t0 = time.perf_counter()
            clf.fit(trf, train[3], sample_weight=cw[train[3]])
            train_s = time.perf_counter()-t0
            p, _ = temporal.scale_probabilities(clf.predict_proba(caf), calibration[3],
                                                clf.predict_proba(tef))
            append_metrics(metric_rows, held_out, seed, "HGB-current",
                           "source_only", raw[held_out], p,
                           {"train_seconds":train_s, "parameters":np.nan, "epochs":np.nan,
                            "inference_ms_per_observation":np.nan})
            ensemble.setdefault(("HGB-current", "source_only"), []).append(p)

        # Store fixed five-member ensemble probabilities for diagnosis and
        # reversal auditing. These are predictions only, not fitting targets.
        for (method, regime), members in ensemble.items():
            p = np.mean(members, axis=0)
            # Hospital is the prespecified collapse audit. Store every hospital
            # prediction; retain a deterministic 10% audit sample elsewhere.
            ids = np.arange(len(p)) if held_out == "hospital" else np.arange(0, len(p), 10)
            for i in ids:
                pred_rows.append({
                    "held_out_building":held_out, "method":method, "regime":regime,
                    "sample":i, "subtype":raw[held_out]["subtype"][i],
                    "y":int(raw[held_out]["y"][i]), "p0":float(p[i,0]), "p1":float(p[i,1]),
                })

    pd.DataFrame(metric_rows).to_csv(OUT/"major3_ahu_adaptation_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT/"major3_ahu_feature_shift.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(OUT/"major3_ahu_global_shift.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(OUT/"major3_ahu_prior_shift.csv", index=False)
    pd.concat(pca_parts, ignore_index=True).to_csv(OUT/"major3_ahu_pca.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/"major3_ahu_adaptation_predictions.csv", index=False)
    (OUT/"major3_ahu_shift_design.json").write_text(json.dumps({
        "zero_shot":["EF-PD source normalization", "RO-PDRF-NOQ source normalization",
                     "GroupDRO-EF-PD", "HGB-current"],
        "unsupervised_adaptation":["target-building normalization", "target-to-source CORAL"],
        "target_labels_used_for_adaptation":False,
        "seeds":list(SEEDS), "epochs":EPOCHS,
        "reverse_auc_definition":"AUROC using 1-p(fault); diagnostic only",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
