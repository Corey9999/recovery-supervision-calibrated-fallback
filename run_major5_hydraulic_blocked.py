"""Class-preserving blocked hydraulic split with guard bands.

A naive global chronological split is invalid for this dataset because some
component states occur only in restricted acquisition ranges. This audit keeps
every native state represented but assigns contiguous within-state blocks to
train, selection, calibration and test, discarding guard cycles at boundaries.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import run_hydraulic_validation as hyd
import run_major_revision_experiments as major
import run_major3_objective_matched as m3
import run_q1_risk_sensitive as risk

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
QUICK = os.getenv("M5_HYD_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101,106))
EPOCHS = 3 if QUICK else 80
GAP = 5
METHODS = {"RO-CAGF":m3.RO_CAGF, "RO-PDRF":m3.RO_PDRF}


def blocked_split(x, profile, column):
    stable_ids = np.flatnonzero(profile[:,4] == 0)
    xs = x[stable_ids]
    labels = profile[stable_ids,column]
    values = np.sort(np.unique(labels))
    mapping = {v:i for i,v in enumerate(values)}
    y = np.array([mapping[v] for v in labels],np.int64)
    pieces = {k:[] for k in ("train","selection","calibration","test")}
    guards = []
    for cls in range(len(values)):
        ids = np.flatnonzero(y == cls)  # acquisition order is preserved
        usable = len(ids)-3*GAP
        n_train = int(np.floor(.60*usable))
        n_sel = int(np.floor(.15*usable))
        n_cal = int(np.floor(.10*usable))
        a = n_train
        b = a+GAP
        c = b+n_sel
        d = c+GAP
        e = d+n_cal
        f = e+GAP
        pieces["train"].extend(ids[:a])
        guards.extend(ids[a:b])
        pieces["selection"].extend(ids[b:c])
        guards.extend(ids[c:d])
        pieces["calibration"].extend(ids[d:e])
        guards.extend(ids[e:f])
        pieces["test"].extend(ids[f:])
    for k in pieces:
        pieces[k] = np.array(sorted(pieces[k]),dtype=int)
    scaler = StandardScaler().fit(xs[pieces["train"]].reshape(len(pieces["train"]),-1))
    z = np.clip(scaler.transform(xs.reshape(len(xs),-1)),-8,8).reshape(xs.shape).astype(np.float32)
    def view(ids):
        q = np.ones((len(ids),5),np.float32)
        return z[ids],q.copy(),q,y[ids]
    return (view(pieces["train"]), view(pieces["selection"]),
            view(pieces["calibration"]), view(pieces["test"]),
            values, pieces, np.array(guards), z, y)


def similarity_audit(z, y, pieces, task):
    train = pieces["train"]
    test = pieces["test"]
    flat = z.reshape(len(z),-1)
    nn = NearestNeighbors(n_neighbors=1,metric="euclidean").fit(flat[train])
    dist, nearest = nn.kneighbors(flat[test])
    train_norm = flat[train]/np.linalg.norm(flat[train],axis=1,keepdims=True).clip(1e-8)
    test_norm = flat[test]/np.linalg.norm(flat[test],axis=1,keepdims=True).clip(1e-8)
    cosine = (test_norm*train_norm[nearest[:,0]]).sum(1)
    rows = []
    for i,tidx in enumerate(test):
        rows.append({"task":task,"test_index":int(tidx),
                     "nearest_train_index":int(train[nearest[i,0]]),
                     "same_class":bool(y[tidx]==y[train[nearest[i,0]]]),
                     "euclidean_distance":float(dist[i,0]),
                     "cosine_similarity":float(cosine[i])})
    return rows


def main():
    x, profile = hyd.load_features()
    rows, costs, counts, similarities = [], [], [], []
    t0 = time.perf_counter()
    tasks = list(hyd.TASKS.items())[:1] if QUICK else hyd.TASKS.items()
    for task, column in tasks:
        (train, selection, calibration, test, values,
         pieces, guards, z, y) = blocked_split(x,profile,column)
        similarities.extend(similarity_audit(z,y,pieces,task))
        for split_name, ids in pieces.items():
            for cls, value in enumerate(values):
                counts.append({"task":task,"split":split_name,
                               "class_value":int(value),
                               "n":int((y[ids]==cls).sum()),
                               "guard_cycles_per_boundary":GAP})
        ty = train[3]
        cw = len(ty)/(len(values)*np.bincount(ty,minlength=len(values)))
        testx, _, _, testy = test
        for seed in SEEDS:
            for method, spec0 in METHODS.items():
                spec = replace(spec0,use_quality=False,train_quality=False)
                model,cost = major.fit(spec,train,selection,cw,seed,
                                       prior=(.2,)*5,epochs=EPOCHS)
                temp = major.fit_temperature(model,calibration,neutral_q=True)
                for fault, modality in (("pressure",0),("vibration",4)):
                    xx, mm, qq, affected = hyd.hydraulic_fault(
                        testx,modality,89000+modality)
                    p, _, _, _, infer = major.predict(model,xx,mm,qq,temp)
                    for subset,take in (("all",np.ones(len(testy),bool)),
                                        ("affected",affected),
                                        ("unaffected",~affected)):
                        for metric,value in risk.all_metrics(testy[take],p[take]).items():
                            rows.append({"task":task,"seed":seed,"method":method,
                                         "fault":fault,"subset":subset,
                                         "metric":metric,"value":value,
                                         "n":int(take.sum())})
                costs.append({"task":task,"seed":seed,"method":method,
                              **cost,"inference_ms_per_observation":1000*infer/len(testy)})
    pd.DataFrame(rows).to_csv(OUT/"major5_hydraulic_blocked_metrics.csv",index=False)
    pd.DataFrame(costs).to_csv(OUT/"major5_hydraulic_blocked_costs.csv",index=False)
    pd.DataFrame(counts).to_csv(OUT/"major5_hydraulic_blocked_counts.csv",index=False)
    pd.DataFrame(similarities).to_csv(OUT/"major5_hydraulic_similarity.csv",index=False)
    (OUT/"major5_hydraulic_blocked_design.json").write_text(json.dumps({
        "split":"within-native-condition acquisition-order blocks",
        "ratios":[.60,.15,.10,.15],
        "guard_cycles_at_each_boundary":GAP,
        "global_chronological_split_rejected":"cooler and accumulator states do not overlap across global time partitions",
        "seeds":list(SEEDS),"methods":list(METHODS),
        "test_data_used_for_selection":False,
        "elapsed_seconds":time.perf_counter()-t0,
    },indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
