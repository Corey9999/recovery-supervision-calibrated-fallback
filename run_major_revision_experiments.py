"""Major-revision experiments requested by the second strict Q2 review.

The script uses fixed test corruptions across optimization seeds and methods. It
adds matched corruption-trained baselines, mechanism ablations, diagnostic-q
controls, leave-one-group-out tests, score/weight audits, and hyperparameter
sensitivity. It never performs post-hoc method selection on the test set.
"""

from __future__ import annotations

import json
import os
import time
import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import run_q2_revision as q2
import run_uci_validation as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
SEEDS = (101,) if os.getenv("MAJOR_QUICK") == "1" else tuple(range(101, 111))
MAX_EPOCHS = 3 if os.getenv("MAJOR_QUICK") == "1" else 60


@dataclass(frozen=True)
class Spec:
    architecture: str = "bounded"  # bounded, gate, attention_gate, expert_router, qmf, entropy, early, uniform, quality
    paired: bool = True
    score_loss: bool = True
    consistency: bool = True
    norm_mono: bool = True
    raw_mono: bool = False
    score_rank: bool = True
    direct_rank: bool = False
    use_quality: bool = True
    train_quality: bool = True
    bound: float = 3.0
    beta: float = 0.25
    w_score: float = 0.20
    w_deg: float = 0.15
    w_cons: float = 0.10
    w_mono: float = 0.05
    w_rank: float = 0.15
    boundary_penalty: float = 0.50
    degraded_auc: bool = False
    confidence_mono: bool = False
    w_auc: float = 0.10
    w_confidence: float = 0.05
    interior_barrier: bool = False
    recovery_distillation: bool = False
    recovery_teacher: str = "best_ce"  # legacy default; formal headline scripts set clean explicitly
    recovery_confidence_threshold: float = 0.0
    recovery_confidence_weighted: bool = False
    teacher_ema_decay: float = 0.0
    qmf_rank_weight: float = 0.10
    qmf_confidence_scale: float = 0.10
    qmf_normalized_fusion: bool = False
    qmf_faithful_rank: bool = False
    qmf_detach_fusion_confidence: bool = False
    modality_dropout: bool = False
    hard_group_sampling: bool = False
    degraded_cvar: bool = False
    brier_regularization: bool = False
    gate_kl_regularization: bool = False
    w_interior: float = 0.03
    w_recovery: float = 0.15
    w_brier: float = 0.05
    w_gate_kl: float = 0.005
    hardness_temperature: float = 1.0
    cvar_fraction: float = 0.30


SPECS = {
    "EF": Spec("early", paired=False, score_loss=False, consistency=False, norm_mono=False,
               score_rank=False, use_quality=True),
    "EF_PD": Spec("early", score_loss=False, consistency=False, norm_mono=False,
                  score_rank=False, use_quality=True),
    "UF": Spec("uniform", paired=False, score_loss=False, consistency=False, norm_mono=False,
               score_rank=False, use_quality=False),
    "UF_PD": Spec("uniform", score_loss=False, consistency=False, norm_mono=False,
                  score_rank=False, use_quality=False),
    "Q_ONLY": Spec("quality", score_loss=False, consistency=False, norm_mono=False,
                   score_rank=False, use_quality=True),
    "CAGF": Spec("gate", score_loss=False, consistency=True, norm_mono=True,
                 score_rank=False, use_quality=True),
    "DWR": Spec("gate", score_loss=False, consistency=True, norm_mono=True,
                score_rank=False, direct_rank=True, use_quality=True),
    "ENTROPY_PD": Spec("entropy", score_loss=False, consistency=True, norm_mono=True,
                       score_rank=False, use_quality=False, train_quality=False),
    "PD_ONLY": Spec("bounded", score_loss=False, consistency=False, norm_mono=False,
                    score_rank=False),
    "PD_MONO": Spec("bounded", score_loss=False, consistency=False, norm_mono=True,
                    score_rank=False),
    "PD_RANK": Spec("bounded", score_loss=False, consistency=False, norm_mono=False,
                    score_rank=True),
    "NO_LSCORE": Spec("bounded", score_loss=False),
    "LSCORE_NO_LRANK": Spec("bounded", score_rank=False),
    "PDRF_NOQ": Spec("bounded", use_quality=False, train_quality=False),
    "PDRF": Spec("bounded"),
}


class Encoder(nn.Module):
    def __init__(self, d: int, classes: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.logits = nn.Linear(32, classes)


class Fusion(nn.Module):
    def __init__(self, modalities: int, d: int, classes: int, spec: Spec):
        super().__init__()
        self.m, self.d, self.c, self.spec = modalities, d, classes, spec
        if spec.architecture == "early":
            self.net = nn.Sequential(nn.Linear(modalities*d + 2*modalities, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, classes))
            return
        self.encoders = nn.ModuleList([Encoder(d, classes) for _ in range(modalities)])
        if spec.architecture == "bounded":
            self.score_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
                for _ in range(modalities)])
        if spec.architecture == "gate":
            self.gate = nn.Sequential(nn.Linear(modalities*32 + 2*modalities, 64), nn.ReLU(),
                                      nn.Linear(64, modalities))
        if spec.architecture == "attention_gate":
            # A compact, parameter-budget-matched group-token attention gate.
            # Sensor identity is retained through learned group embeddings;
            # unavailable groups are masked before self-attention and gating.
            self.group_embedding = nn.Parameter(torch.zeros(1, modalities, 32))
            nn.init.normal_(self.group_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)
            self.group_attention = nn.TransformerEncoder(layer, num_layers=1)
            self.attention_gate = nn.Sequential(
                nn.Linear(34, 16), nn.ReLU(), nn.Linear(16, 1))
        if spec.architecture == "expert_router":
            # Sensor-group adaptation of multi-expert routing.  The candidate
            # set contains one expert per physical group and one joint expert
            # that can model cross-group interactions.  The router may select
            # a single robust group or retain the joint representation.
            context_width = modalities*32 + 2*modalities
            self.joint_expert = nn.Sequential(
                nn.Linear(context_width, 64), nn.ReLU(), nn.Linear(64, classes))
            self.expert_router = nn.Sequential(
                nn.Linear(context_width, 64), nn.ReLU(), nn.Linear(64, modalities+1))

    def forward(self, x, mask, q):
        if torch.any(mask.sum(1) == 0):
            raise ValueError("Fusion is undefined when all sensor groups are unavailable; abstain before model inference.")
        if self.spec.architecture == "early":
            logits = self.net(torch.cat([x.reshape(len(x), -1), mask, q], 1))
            z = torch.zeros_like(mask)
            return logits, z, torch.zeros(len(x), self.m, self.c), z, z
        hs, ls = [], []
        for j, enc in enumerate(self.encoders):
            h = enc.body(x[:, j]); hs.append(h); ls.append(enc.logits(h))
        modal_logits = torch.stack(ls, 1)
        if self.spec.architecture == "bounded":
            raw = torch.cat([head(h) for head, h in zip(self.score_heads, hs)], 1)
            scores = self.spec.bound * torch.tanh(raw / self.spec.bound)
            unnorm = mask * torch.exp(-self.spec.beta * scores)
            if self.spec.use_quality:
                unnorm = unnorm * q.clamp(0.01, 1.0)
        elif self.spec.architecture == "qmf":
            # Sensor-group adaptation of Quality-aware Multimodal Fusion
            # (QMF; Zhang et al., ICML 2023). The public implementation uses
            # confidence = -0.1 * energy, with
            # energy = -logsumexp(group logits), before decision-level fusion.
            energy = -torch.logsumexp(modal_logits, dim=2)
            confidence = (-self.spec.qmf_confidence_scale * energy).clamp_min(1e-6)
            unnorm = mask * confidence
            scores = energy
        elif self.spec.architecture == "entropy":
            # Parameter-free predictive-uncertainty weighting. Entropy is
            # normalized by log(C), then converted to a positive confidence.
            modal_prob = torch.softmax(modal_logits, dim=2).clamp_min(1e-8)
            entropy = -(modal_prob * modal_prob.log()).sum(2) / np.log(self.c)
            unnorm = mask * torch.exp(-2.0 * entropy)
            scores = entropy
        elif self.spec.architecture == "gate":
            g = self.gate(torch.cat(hs + [mask, q], 1)).masked_fill(mask == 0, -1e9)
            unnorm = torch.exp(g - g.max(1, keepdim=True).values) * mask
            scores = torch.zeros_like(mask)
        elif self.spec.architecture == "attention_gate":
            tokens = torch.stack(hs, 1) + self.group_embedding
            attended = self.group_attention(tokens, src_key_padding_mask=(mask == 0))
            gate_input = torch.cat([attended, mask.unsqueeze(-1), q.unsqueeze(-1)], 2)
            g = self.attention_gate(gate_input).squeeze(-1).masked_fill(mask == 0, -1e9)
            unnorm = torch.exp(g - g.max(1, keepdim=True).values) * mask
            scores = torch.zeros_like(mask)
        elif self.spec.architecture == "expert_router":
            context = torch.cat(hs + [mask, q], 1)
            joint_logits = self.joint_expert(context)
            expert_logits = torch.cat([modal_logits, joint_logits[:, None, :]], 1)
            expert_mask = torch.cat(
                [mask, torch.ones(len(mask), 1, dtype=mask.dtype, device=mask.device)], 1)
            gate = self.expert_router(context).masked_fill(expert_mask == 0, -1e9)
            expert_weights = torch.softmax(gate, 1)
            logits = (expert_weights.unsqueeze(-1)*expert_logits).sum(1)
            # Return an effective physical-group weight for the shared
            # monotonicity audit by distributing the joint expert weight over
            # the groups available in that observation.
            joint_share = expert_weights[:, -1:] * mask / mask.sum(1, keepdim=True).clamp_min(1)
            weights = expert_weights[:, :-1] + joint_share
            scores = torch.zeros_like(mask)
            unnorm = weights
            return logits, scores, modal_logits, weights, unnorm
        elif self.spec.architecture == "quality":
            unnorm = mask * q.clamp(0.01, 1.0)
            scores = torch.zeros_like(mask)
        else:
            unnorm = mask
            scores = torch.zeros_like(mask)
        weights = unnorm / unnorm.sum(1, keepdim=True).clamp_min(1e-6)
        if self.spec.architecture == "qmf":
            # Preserve the decisive (unnormalized) logit fusion in the public
            # QMF implementation. Normalized weights are returned only for
            # diagnostics and do not alter the fused decision.
            fusion_weight = weights if self.spec.qmf_normalized_fusion else unnorm
            if self.spec.qmf_detach_fusion_confidence:
                fusion_weight = fusion_weight.detach()
            logits = (fusion_weight.unsqueeze(-1) * modal_logits).sum(1)
        else:
            logits = (weights.unsqueeze(-1) * modal_logits).sum(1)
        return logits, scores, modal_logits, weights, unnorm


def set_seed(seed):
    base.set_seed(seed)


def fit(spec: Spec, train, select, class_weights, seed, prior=None, epochs=60, trace=None):
    set_seed(seed)
    tx, tm, tq, ty = train
    vx, vm, vq, vy = select
    if not spec.train_quality:
        tq = np.ones_like(tq); vq = np.ones_like(vq)
    m, d, c = tx.shape[1], tx.shape[2], len(class_weights)
    model = Fusion(m, d, c, spec)
    ema_model = None
    if spec.teacher_ema_decay > 0:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(tm),
                                      torch.from_numpy(tq), torch.from_numpy(ty),
                                      torch.arange(len(tx))),
                        batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cw = torch.from_numpy(class_weights.astype(np.float32))
    prior_t = torch.tensor(prior or tuple([1/m]*m), dtype=torch.float32)[None, :]
    qmf_loss_history = torch.zeros(len(tx), m, dtype=torch.float32)
    qmf_history_count = torch.zeros(len(tx), m, dtype=torch.float32)
    best, state, ema_state, stale, used_epochs = float("inf"), None, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        used_epochs = epoch + 1
        model.train()
        epoch_losses, epoch_grad_norms = [], []
        for x, mask, q, y, sample_index in loader:
            opt.zero_grad(set_to_none=True)
            logits, scores, modal_logits, weights, unnorm = model(x, mask, q)
            loss = F.cross_entropy(logits, y, weight=cw)
            if spec.architecture == "qmf":
                # QMF jointly supervises the fused and group-specific
                # classifiers. Averaging over available groups keeps the
                # loss scale stable when the number of sensor groups changes.
                per_group_ce = torch.stack([
                    F.cross_entropy(modal_logits[:, j], y, weight=cw,
                                    reduction="none") for j in range(m)], 1)
                loss = loss + (per_group_ce*mask).sum()/mask.sum().clamp_min(1)

                # Training-trajectory ranking control from QMF: historical
                # high-loss group predictions should receive lower energy
                # confidence. Pairs are formed within the shuffled minibatch;
                # only pairs with prior trajectory observations contribute.
                if spec.qmf_faithful_rank:
                    # The official text--image implementation accumulates
                    # per-sample classification loss before forming pairs and
                    # uses the normalized history difference as a margin.
                    with torch.no_grad():
                        qmf_loss_history[sample_index] += per_group_ce.detach()*mask
                        qmf_history_count[sample_index] += mask
                history = qmf_loss_history[sample_index]
                seen = qmf_history_count[sample_index] > 0
                rank_terms = []
                confidence = unnorm
                for j in range(m):
                    valid_pair = ((mask[:, j] > 0) & seen[:, j] &
                                  (torch.roll(mask[:, j], -1) > 0) &
                                  torch.roll(seen[:, j], -1))
                    if valid_pair.any():
                        h1 = history[:, j]
                        h2 = torch.roll(h1, -1)
                        if spec.qmf_faithful_rank:
                            all_history = qmf_loss_history[:, j]
                            hmin, hmax = all_history.min(), all_history.max()
                            denom = (hmax-hmin).clamp_min(1e-8)
                            h1 = (h1-hmin)/denom
                            h2 = (h2-hmin)/denom
                            target = torch.sign(h1-h2)
                        else:
                            target = torch.sign(h2-h1)
                        keep = valid_pair & (target != 0)
                        if keep.any():
                            conf2 = torch.roll(confidence[:, j], -1)
                            rank_target = target
                            if spec.qmf_faithful_rank:
                                margin = torch.abs(h1-h2)
                                target_nonzero = torch.where(
                                    target == 0, torch.ones_like(target), target)
                                conf2 = conf2 + margin/target_nonzero
                                rank_target = -target
                            rank_terms.append(F.margin_ranking_loss(
                                confidence[keep, j], conf2[keep],
                                rank_target[keep], margin=0.0))
                if rank_terms:
                    loss = loss + spec.qmf_rank_weight*torch.stack(rank_terms).mean()
                if not spec.qmf_faithful_rank:
                    with torch.no_grad():
                        old_count = qmf_history_count[sample_index]
                        new_count = old_count + mask
                        qmf_loss_history[sample_index] = torch.where(
                            mask > 0,
                            (qmf_loss_history[sample_index]*old_count +
                             per_group_ce.detach()*mask)/new_count.clamp_min(1),
                            qmf_loss_history[sample_index])
                        qmf_history_count[sample_index] = new_count
            if spec.brier_regularization:
                target_onehot = F.one_hot(y, c).float()
                loss = loss + spec.w_brier*torch.square(torch.softmax(logits, 1)-target_onehot).sum(1).mean()
            if spec.gate_kl_regularization and spec.architecture in ("gate", "attention_gate", "expert_router"):
                # Architecture-matched analogue of the bounded-score interior
                # penalty: discourage a near-one-hot gate without imposing a
                # score geometry that the unrestricted gate does not possess.
                available = mask.sum(1).clamp_min(1)
                gate_kl = (weights.clamp_min(1e-8) *
                           (weights.clamp_min(1e-8).log() + available.log()[:, None])).sum(1)
                loss = loss + spec.w_gate_kl*gate_kl.mean()
            if spec.score_loss and spec.architecture == "bounded":
                per = torch.stack([F.cross_entropy(modal_logits[:, j], y, weight=cw, reduction="none")
                                   for j in range(m)], 1)
                score_core = ((torch.exp(-scores) * per + scores) * mask).sum()/mask.sum().clamp_min(1)
                boundary = (((scores/spec.bound)**4)*mask).sum()/mask.sum().clamp_min(1)
                loss = loss + spec.w_score*score_core + spec.boundary_penalty*boundary
                if spec.interior_barrier:
                    interior = -torch.log((1-torch.square(scores/spec.bound)).clamp_min(1e-4))
                    loss = loss + spec.w_interior*(interior*mask).sum()/mask.sum().clamp_min(1)
            if spec.paired:
                probs = mask * prior_t
                if spec.hard_group_sampling:
                    group_ce = torch.stack([
                        F.cross_entropy(modal_logits[:, j], y, weight=cw, reduction="none")
                        for j in range(m)], 1).detach()
                    centered = group_ce-(group_ce*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
                    probs = probs*torch.exp(spec.hardness_temperature*centered.clamp(-4,4))
                probs = torch.where(probs.sum(1, keepdim=True) > 0, probs, mask)
                chosen = torch.multinomial(probs/probs.sum(1, keepdim=True), 1).squeeze(1)
                degraded = F.one_hot(chosen, m).float()*mask
                rho = 0.15 + 0.35*torch.rand(len(x), 1)
                valid = mask.sum(1) > 1
                if spec.modality_dropout:
                    degraded_mask = mask*(1-degraded)
                    degraded_mask[~valid] = mask[~valid]
                    x2 = x*degraded_mask.unsqueeze(-1)
                    q2v = q*degraded_mask
                else:
                    degraded_mask = mask
                    x2 = x + torch.randn_like(x)*(2.5*(1-rho)).unsqueeze(-1)*degraded.unsqueeze(-1)
                    report = (torch.rand(len(x), 1) > 0.5).float()
                    q2v = q*(1-degraded + degraded*(report*rho + 1-report))
                if not spec.train_quality:
                    q2v = torch.ones_like(q2v)
                l2, s2, modal_logits2, w2, u2 = model(x2, degraded_mask, q2v)
                if spec.degraded_cvar:
                    degraded_losses = F.cross_entropy(l2, y, weight=cw, reduction="none")
                    k = max(1, int(np.ceil(spec.cvar_fraction*len(degraded_losses))))
                    degraded_task = torch.topk(degraded_losses, k).values.mean()
                else:
                    degraded_task = F.cross_entropy(l2, y, weight=cw)
                loss = loss + spec.w_deg*degraded_task
                if spec.architecture == "qmf":
                    per_group_degraded = torch.stack([
                        F.cross_entropy(modal_logits2[:, j], y, weight=cw,
                                        reduction="none") for j in range(m)], 1)
                    loss = loss + spec.w_deg*(per_group_degraded*degraded_mask).sum()/degraded_mask.sum().clamp_min(1)
                if spec.brier_regularization:
                    loss = loss + spec.w_brier*torch.square(torch.softmax(l2, 1)-target_onehot).sum(1).mean()
                if spec.gate_kl_regularization and spec.architecture in ("gate", "attention_gate", "expert_router"):
                    available2 = mask.sum(1).clamp_min(1)
                    gate_kl2 = (w2.clamp_min(1e-8) *
                                (w2.clamp_min(1e-8).log() + available2.log()[:, None])).sum(1)
                    loss = loss + spec.w_gate_kl*gate_kl2.mean()
                if spec.interior_barrier and spec.architecture == "bounded":
                    interior2 = -torch.log((1-torch.square(s2/spec.bound)).clamp_min(1e-4))
                    loss = loss + spec.w_interior*(interior2*mask).sum()/mask.sum().clamp_min(1)
                if spec.degraded_auc:
                    # Smooth one-vs-rest pairwise ranking surrogate on the
                    # deliberately degraded view. This directly targets the
                    # ordering measured by macro-AUROC without using test data.
                    auc_terms = []
                    for cls in range(c):
                        pos = l2[y == cls, cls]
                        neg = l2[y != cls, cls]
                        if len(pos) and len(neg):
                            auc_terms.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
                    if auc_terms:
                        loss = loss + spec.w_auc*torch.stack(auc_terms).mean()
                if spec.confidence_mono:
                    clean_conf = torch.softmax(logits, 1).amax(1).detach()
                    degraded_conf = torch.softmax(l2, 1).amax(1)
                    loss = loss + spec.w_confidence*torch.relu(degraded_conf-clean_conf).mean()
                if valid.any() and (spec.consistency or spec.norm_mono or spec.raw_mono
                                    or spec.recovery_distillation):
                    removed = mask*(1-degraded)
                    removed[~valid] = mask[~valid]
                    lr, _, _, _, _ = model(x*removed.unsqueeze(-1), removed, q)
                    sev = (1-rho[valid]).clamp(0, 1)
                    target = (1-sev)*torch.softmax(logits[valid], 1).detach() + sev*torch.softmax(lr[valid], 1).detach()
                    if spec.consistency:
                        loss = loss + spec.w_cons*F.mse_loss(torch.softmax(l2[valid], 1), target)
                    if spec.recovery_distillation:
                        clean_prob = torch.softmax(logits[valid],1)
                        removal_prob = torch.softmax(lr[valid],1)
                        if spec.recovery_teacher == "best_ce":
                            clean_ce = F.cross_entropy(logits[valid], y[valid], reduction="none")
                            removal_ce = F.cross_entropy(lr[valid], y[valid], reduction="none")
                            choose_removal = (removal_ce < clean_ce).unsqueeze(1)
                        elif spec.recovery_teacher == "clean":
                            choose_removal = torch.zeros((valid.sum(),1),dtype=torch.bool)
                        elif spec.recovery_teacher == "removal":
                            choose_removal = torch.ones((valid.sum(),1),dtype=torch.bool)
                        elif spec.recovery_teacher == "confidence":
                            choose_removal = (removal_prob.max(1).values >
                                              clean_prob.max(1).values).unsqueeze(1)
                        elif spec.recovery_teacher == "clean_removal_agreement":
                            teacher = 0.5*(clean_prob+removal_prob)
                            teacher_keep = clean_prob.argmax(1) == removal_prob.argmax(1)
                            choose_removal = torch.zeros((valid.sum(),1),dtype=torch.bool)
                        elif spec.recovery_teacher == "ce_conflict_gate":
                            # Training labels are already available for the
                            # degraded task loss. Distillation abstains when
                            # the clean teacher assigns greater CE than the
                            # current faulted prediction, preventing a
                            # demonstrably worse teacher from supervising it.
                            clean_ce = F.cross_entropy(
                                logits[valid], y[valid], reduction="none")
                            degraded_ce = F.cross_entropy(
                                l2[valid], y[valid], reduction="none")
                            teacher = clean_prob
                            teacher_keep = clean_ce <= degraded_ce
                            choose_removal = torch.zeros((valid.sum(),1),dtype=torch.bool)
                        elif spec.recovery_teacher in ("ema", "ema_clean_agreement"):
                            if ema_model is None:
                                raise ValueError("EMA recovery teacher requires teacher_ema_decay > 0")
                            with torch.no_grad():
                                ema_logits = ema_model(x, mask, q)[0][valid]
                                ema_prob = torch.softmax(ema_logits, 1)
                            choose_removal = torch.zeros((valid.sum(),1),dtype=torch.bool)
                        else:
                            raise ValueError(f"Unknown recovery teacher: {spec.recovery_teacher}")
                        if spec.recovery_teacher == "ema":
                            teacher = ema_prob.detach()
                            teacher_keep = None
                        elif spec.recovery_teacher == "ema_clean_agreement":
                            teacher = (0.5*(ema_prob+clean_prob)).detach()
                            teacher_keep = ema_prob.argmax(1) == clean_prob.argmax(1)
                        elif spec.recovery_teacher in ("clean_removal_agreement",
                                                      "ce_conflict_gate"):
                            teacher = teacher.detach()
                        else:
                            teacher = torch.where(choose_removal, removal_prob, clean_prob).detach()
                            teacher_keep = None
                        per_recovery = F.kl_div(
                            torch.log_softmax(l2[valid],1), teacher,
                            reduction="none").sum(1)
                        if teacher_keep is not None:
                            if teacher_keep.any():
                                loss = loss + spec.w_recovery*per_recovery[teacher_keep].mean()
                        elif spec.recovery_confidence_threshold > 0:
                            confident = teacher.max(1).values >= spec.recovery_confidence_threshold
                            if confident.any():
                                loss = loss + spec.w_recovery*per_recovery[confident].mean()
                        elif spec.recovery_confidence_weighted:
                            confidence = teacher.max(1).values
                            loss = loss + spec.w_recovery*(per_recovery*confidence).sum()/confidence.sum().clamp_min(1e-6)
                        else:
                            loss = loss + spec.w_recovery*per_recovery.mean()
                    selected_w0 = (weights*degraded).sum(1)[valid]
                    selected_w1 = (w2*degraded).sum(1)[valid]
                    if spec.norm_mono:
                        loss = loss + spec.w_mono*torch.relu(selected_w1-selected_w0).mean()
                    if spec.raw_mono:
                        selected_u0 = (unnorm*degraded).sum(1)[valid]
                        selected_u1 = (u2*degraded).sum(1)[valid]
                        loss = loss + spec.w_mono*torch.relu(selected_u1-selected_u0).mean()
                if valid.any() and spec.score_rank and spec.architecture == "bounded":
                    s0 = (scores*degraded).sum(1)[valid]
                    s1 = (s2*degraded).sum(1)[valid]
                    margin = 0.5*(1-rho.squeeze(1)[valid])/0.85
                    loss = loss + spec.w_rank*torch.relu(margin+s0.detach()-s1).mean()
                if valid.any() and spec.direct_rank:
                    w0 = (weights*degraded).sum(1)[valid]
                    w1 = (w2*degraded).sum(1)[valid]
                    loss = loss + spec.w_rank*torch.relu(0.10+w1-w0).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ema_model is not None:
                decay = spec.teacher_ema_decay
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                        ema_parameter.mul_(decay).add_(parameter, alpha=1-decay)
            epoch_losses.append(float(loss.detach()))
            epoch_grad_norms.append(float(grad_norm.detach()))
        model.eval()
        with torch.no_grad():
            val = F.cross_entropy(model(torch.from_numpy(vx), torch.from_numpy(vm),
                                        torch.from_numpy(vq))[0], torch.from_numpy(vy)).item()
        if trace is not None:
            trace.append({"seed":seed, "epoch":epoch+1,
                          "train_loss":float(np.mean(epoch_losses)),
                          "gradient_norm":float(np.mean(epoch_grad_norms)),
                          "selection_ce":float(val)})
        if val < best - 1e-4:
            best, stale = val, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ema_model is not None:
                ema_state = {k: v.detach().clone()
                             for k, v in ema_model.state_dict().items()}
        else:
            stale += 1
            if stale > 8: break
    model.load_state_dict(state)
    if ema_model is not None:
        ema_model.load_state_dict(ema_state)
        # Expose the checkpoint-matched teacher for held-out agreement audits;
        # it remains excluded from optimization and parameter counts.
        model.ema_teacher = ema_model.eval()
    return model.eval(), {"parameters": sum(p.numel() for p in model.parameters()
                                                    if p.requires_grad),
                          "train_seconds": time.perf_counter()-t0, "epochs": used_epochs}


def fit_temperature(model, calibration, neutral_q=False):
    x, m, q, y = calibration
    if neutral_q: q = np.ones_like(q)
    with torch.no_grad(): logits = model(torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(q))[0].numpy()
    def objective(log_t):
        p = np.exp(logits/np.exp(log_t) - (logits/np.exp(log_t)).max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return log_loss(y, p, labels=np.arange(logits.shape[1]))
    return float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def predict(model, x, m, q, temperature=1.0):
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, scores, _, weights, unnorm = model(torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(q))
        p = torch.softmax(logits/temperature, 1).numpy()
    return p, scores.numpy(), weights.numpy(), unnorm.numpy(), time.perf_counter()-t0


def metrics(y, p):
    c = p.shape[1]; pred = p.argmax(1); yb = label_binarize(y, classes=np.arange(c))
    if c == 2 and yb.shape[1] == 1: yb = np.column_stack([1-yb[:, 0], yb[:, 0]])
    out = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
           "macro_auroc": roc_auc_score(yb, p, average="macro", multi_class="ovr"),
           "macro_auprc": average_precision_score(yb, p, average="macro"),
           "nll": log_loss(y, p, labels=np.arange(c))}
    for j in range(c):
        out[f"class_{j+1}_precision"] = precision_score(y == j, pred == j, zero_division=0)
        out[f"class_{j+1}_recall"] = recall_score(y == j, pred == j, zero_division=0)
        out[f"class_{j+1}_f1"] = f1_score(y == j, pred == j, zero_division=0)
        out[f"class_{j+1}_auroc"] = roc_auc_score(yb[:, j], p[:, j])
        out[f"class_{j+1}_auprc"] = average_precision_score(yb[:, j], p[:, j])
    return {k: float(v) for k, v in out.items()}


def fixed_fault(x, kind="silent", modality=0, scale=3.0, prevalence=0.40, seed=70001):
    rng = np.random.default_rng(seed)
    xx = x.copy(); n, m, d = x.shape
    mask = base.random_mask(n, rng, 0.20)
    affected = rng.random(n) < prevalence
    noise = rng.normal(0, scale, (affected.sum(), d)).astype(np.float32)
    xx[affected, modality] += noise
    q = np.ones((n, m), np.float32)
    if kind == "reported": q[affected, modality] = 0.15
    elif kind == "shuffled":
        q[affected, modality] = 0.15; q = q[rng.permutation(n)]
    elif kind == "misleading":
        q[affected, :] = 0.20; q[affected, modality] = 1.0
    elif kind not in {"silent", "missing_q"}: raise ValueError(kind)
    xx *= mask[:, :, None]
    return xx.astype(np.float32), mask, q, affected


def paired_audit(model, x, modality=0, seed=71001):
    rng = np.random.default_rng(seed); n, m, d = x.shape
    mask = base.random_mask(n, rng, 0.20); q = np.ones((n, m), np.float32)
    affected = (rng.random(n) < 0.40) & (mask[:, modality] > 0)
    xb = x.copy()*mask[:, :, None]; xa = x.copy()
    xa[affected, modality] += rng.normal(0, 3, (affected.sum(), d)).astype(np.float32)
    xa *= mask[:, :, None]
    _, sb, wb, ub, _ = predict(model, xb.astype(np.float32), mask, q)
    _, sa, wa, ua, _ = predict(model, xa.astype(np.float32), mask, q)
    v = affected
    other = np.arange(m) != modality
    return {"n": int(v.sum()), "raw_score_violation": float(np.mean(sa[v, modality] <= sb[v, modality])),
            "unnormalized_weight_violation": float(np.mean(ua[v, modality] >= ub[v, modality])),
            "normalized_weight_violation": float(np.mean(wa[v, modality] >= wb[v, modality])),
            "other_weight_mean_change": float(np.mean(wa[v][:, other]-wb[v][:, other])),
            "score_saturation": float(np.mean(np.abs(sa[v, modality]) >= 0.95*model.spec.bound)),
            "mean_score_before": float(sb[v, modality].mean()), "mean_score_after": float(sa[v, modality].mean())}


def main():
    zip_path = base.ensure_zip(None)
    groups = [list(range(i, i+4)) for i in range(0, 16, 4)]
    split = q2.prepare_grouped(zip_path, groups)
    tx, ty, _ = split["train"]; sx, sy, _ = split["select"]; cx, cy, _ = split["calibration"]
    testx, testy, testb = split["test"]
    metric_rows=[]; pred_rows=[]; cost_rows=[]; audit_rows=[]; q_rows=[]
    models = {}
    for seed in SEEDS:
        train, select, calibration, cw = q2.train_views(split, seed)
        for name, spec in SPECS.items():
            model, costs = fit(spec, train, select, cw, seed, prior=(.45,.25,.18,.12), epochs=MAX_EPOCHS)
            temp = fit_temperature(model, calibration, neutral_q=not spec.train_quality)
            models[(seed, name)] = (model, temp)
            cost_rows.append({"seed":seed,"method":name,**costs,"temperature":temp})
            for fault_i, fault in enumerate(("natural", "silent")):
                if fault == "natural":
                    rng=np.random.default_rng(70000); mm=base.random_mask(len(testx),rng,0.20)
                    xx=testx.copy()*mm[:,:,None]; qq=np.ones((len(testx),4),np.float32); aff=np.zeros(len(testx),bool)
                else: xx,mm,qq,aff=fixed_fault(testx,"silent",seed=70001)
                p,s,w,u,infer=predict(model,xx,mm,qq,temp)
                for k,v in metrics(testy,p).items(): metric_rows.append({"seed":seed,"method":name,"fault":fault,"subset":"all","metric":k,"value":v})
                if fault=="silent":
                    for subset,sel in (("affected",aff),("unaffected",~aff)):
                        for k,v in metrics(testy[sel],p[sel]).items(): metric_rows.append({"seed":seed,"method":name,"fault":fault,"subset":subset,"metric":k,"value":v})
                for i in range(len(testy)):
                    row={"seed":seed,"sample":i,"batch":int(testb[i]),"y":int(testy[i]),"method":name,
                         "fault":fault,"affected":bool(aff[i]),"pred":int(p[i].argmax())}
                    row.update({f"p{j+1}":float(p[i,j]) for j in range(p.shape[1])}); pred_rows.append(row)
                cost_rows[-1][f"inference_ms_{fault}"]=1000*infer/len(testy)
            if name=="PDRF": audit_rows.append({"seed":seed,**paired_audit(model,testx)})

    # q controls use identical silent-fault values and masks.
    for seed in SEEDS:
        for train_name in ("PDRF","PDRF_NOQ"):
            model,temp=models[(seed,train_name)]
            for regime in ("silent","reported","shuffled","misleading","missing_q"):
                xx,mm,qq,aff=fixed_fault(testx,regime,seed=70001)
                p,_,_,_,_=predict(model,xx,mm,qq,temp)
                for k,v in metrics(testy,p).items(): q_rows.append({"seed":seed,"training":train_name,"test_q":regime,"metric":k,"value":v})

    # Fixed PDRF/CAGF fault grids: same indices and noise for every fitted model.
    grid_rows=[]
    for seed in SEEDS:
        for name in ("CAGF","PDRF"):
            model,temp=models[(seed,name)]
            for prevalence in (.10,.40,.70):
                for scale in (1.0,3.0,5.0):
                    for modality in range(4):
                        xx,mm,qq,aff=fixed_fault(testx,"silent",modality,scale,prevalence,72000+modality)
                        p,_,_,_,_=predict(model,xx,mm,qq,temp)
                        for subset,sel in (("all",np.ones(len(aff),bool)),("affected",aff),("unaffected",~aff)):
                            grid_rows.append({"seed":seed,"method":name,"prevalence":prevalence,"scale":scale,
                                              "modality":modality+1,"subset":subset,"macro_auroc":metrics(testy[sel],p[sel])["macro_auroc"]})

    # Leave one group out for the principal comparator and PDRF (five seeds).
    logo_rows=[]
    for held in range(4):
        prior=np.ones(4); prior[held]=0; prior=tuple((prior/prior.sum()).tolist())
        for seed in SEEDS[:5]:
            train, select, calibration, cw=q2.train_views(split,seed)
            for name in ("CAGF","PDRF"):
                model,_=fit(SPECS[name],train,select,cw,seed,prior=prior,epochs=MAX_EPOCHS)
                temp=fit_temperature(model,calibration)
                xx,mm,qq,aff=fixed_fault(testx,"silent",held,3,.40,73000+held)
                p,_,_,_,_=predict(model,xx,mm,qq,temp)
                logo_rows.append({"held_out_group":held+1,"seed":seed,"method":name,
                                  "macro_auroc":metrics(testy,p)["macro_auroc"],"accuracy":metrics(testy,p)["accuracy"]})

    # Compact hyperparameter sensitivity (three seeds; validation-selected defaults are not reselected on test).
    hp_rows=[]; hp_specs=[]
    for b in (1.,2.,3.,4.,5.): hp_specs.append((f"bound_{b:g}",replace(SPECS["PDRF"],bound=b)))
    for beta in (.10,.25,.50,.75): hp_specs.append((f"beta_{beta:g}",replace(SPECS["PDRF"],beta=beta)))
    for wr in (.05,.15,.30): hp_specs.append((f"rank_weight_{wr:g}",replace(SPECS["PDRF"],w_rank=wr)))
    for label,spec in hp_specs:
        for seed in SEEDS[:3]:
            train,select,calibration,cw=q2.train_views(split,seed)
            model,_=fit(spec,train,select,cw,seed,prior=(.45,.25,.18,.12),epochs=MAX_EPOCHS)
            temp=fit_temperature(model,calibration)
            xx,mm,qq,aff=fixed_fault(testx,"silent",seed=70001)
            p,_,_,_,_=predict(model,xx,mm,qq,temp)
            hp_rows.append({"setting":label,"seed":seed,"macro_auroc":metrics(testy,p)["macro_auroc"],
                            "accuracy":metrics(testy,p)["accuracy"],**paired_audit(model,testx)})

    pd.DataFrame(metric_rows).to_csv(OUT/"major_ablation_metrics.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/"major_seed_predictions.csv",index=False)
    pd.DataFrame(cost_rows).to_csv(OUT/"major_compute_costs.csv",index=False)
    pd.DataFrame(audit_rows).to_csv(OUT/"major_weight_audit.csv",index=False)
    pd.DataFrame(q_rows).to_csv(OUT/"major_quality_controls.csv",index=False)
    pd.DataFrame(grid_rows).to_csv(OUT/"major_fault_grid.csv",index=False)
    pd.DataFrame(logo_rows).to_csv(OUT/"major_leave_group_out.csv",index=False)
    pd.DataFrame(hp_rows).to_csv(OUT/"major_hyperparameter_sensitivity.csv",index=False)
    (OUT/"major_revision_design.json").write_text(json.dumps({"seeds":SEEDS,"methods":list(SPECS),
        "fixed_test_fault_seed":70001,"confirmatory":{"dataset":"UCI gas array batches 8-10",
        "fault":"silent Gaussian, prevalence 0.40, scale 3","metric":"macro-AUROC",
        "comparator":"CAGF"}},indent=2),encoding="utf-8")


if __name__ == "__main__": main()
