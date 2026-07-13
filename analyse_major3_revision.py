"""Tables, statistics and Python-only figures for the third strict-Q1 revision."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt
import run_q1_risk_sensitive as risk

ROOT = Path(__file__).resolve().parent
SRC, TAB, FIG = ROOT/"source_data", ROOT/"tables", ROOT/"figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none", "pdf.fonttype":42, "font.size":7.2,
    "axes.titlesize":8.2, "axes.labelsize":7.2, "xtick.labelsize":6.5,
    "ytick.labelsize":6.5, "legend.fontsize":6.3,
    "axes.spines.top":False, "axes.spines.right":False,
    "legend.frameon":False,
})
COL = {"CAGF":"#8F8F8F", "RO-CAGF":"#D98256",
       "PDRF":"#4C78A8", "RO-PDRF":"#2F8F6B",
       "EF-PD":"#0072B2", "RO-PDRF-NOQ":"#E69F00",
       "GroupDRO-EF-PD":"#009E73", "HGB-current":"#CC79A7"}


def ci_bootstrap(values, reps=10000, seed=93001):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (reps, len(values)), replace=True).mean(1)
    return float(values.mean()), *map(float, np.quantile(draws, [.025, .975]))


def paired(m, left, right, dataset, subset="affected", metric="macro_auroc",
           environment=None):
    z = m[(m.dataset==dataset)&(m.subset==subset)&(m.metric==metric)]
    if environment is not None:
        z = z[z.environment==environment]
    a = z[z.method==left].sort_values(["task","environment","seed"]).value.to_numpy()
    b = z[z.method==right].sort_values(["task","environment","seed"]).value.to_numpy()
    d = a-b
    # Hydraulic task x fault environments, not optimization seeds, are the
    # external replication unit for cross-environment inference.
    if dataset == "hydraulic" and environment is None:
        zp = z[z.method.isin([left,right])].pivot_table(
            index=["task","environment","seed"],columns="method",values="value")
        d = (zp[left]-zp[right]).groupby(["task","environment"]).mean().to_numpy()
    stat, p = wilcoxon(d, method="exact") if len(d) <= 25 else wilcoxon(d, method="approx")
    mean, lo, hi = ci_bootstrap(d)
    return {"dataset":dataset, "subset":subset, "metric":metric,
            "environment":environment or "all_environments",
            "contrast":f"{left} - {right}", "n":len(d),
            "mean_difference":mean, "ci_low":lo, "ci_high":hi,
            "wins":int((d>0).sum()), "losses":int((d<0).sum()),
            "wilcoxon_p":float(p)}


def pm(mean, sd):
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def display_name(value):
    """Expand the legacy internal Full-model identifier in reader-facing output."""
    return str(value).replace("RO-PDRF", "RO-PDRF-Full")


def save(fig, name):
    fig.tight_layout(pad=.8)
    for ext, kwargs in (("svg",{}), ("pdf",{}), ("png",{"dpi":300}), ("tiff",{"dpi":600})):
        fig.savefig(FIG/f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def objective_analysis():
    m = pd.read_csv(SRC/"major3_objective_matched_metrics.csv")
    c = pd.read_csv(SRC/"major3_objective_matched_costs.csv")
    main = m[(m.dataset=="chemical")&(m.environment=="type:silent_gaussian")]
    methods = ["CAGF","RO-CAGF","PDRF","RO-PDRF"]
    metrics = [("all","macro_auroc"),("affected","macro_auroc"),
               ("affected","macro_auprc"),("affected","nll"),("affected","ece15")]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Method & All AUROC & Aff. AUROC & Aff. AUPRC & Aff. NLL & Aff. ECE \\", r"\midrule"]
    for method in methods:
        vals = []
        for subset, metric in metrics:
            z = main[(main.method==method)&(main.subset==subset)&(main.metric==metric)].value
            vals.append(z.mean())
        lines.append(display_name(method)+" & "+" & ".join(f"{v:.3f}" for v in vals)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_2x2_chemical.tex").write_text("\n".join(lines),encoding="utf-8")

    effects = [
        paired(m,"RO-PDRF","PDRF","chemical",environment="type:silent_gaussian"),
        paired(m,"RO-CAGF","CAGF","chemical",environment="type:silent_gaussian"),
        paired(m,"RO-PDRF","RO-CAGF","chemical",environment="type:silent_gaussian"),
        paired(m,"PDRF","CAGF","chemical",environment="type:silent_gaussian"),
    ]
    # Difference-in-differences on the same seeds.
    z = main[(main.subset=="affected")&(main.metric=="macro_auroc")]
    pivot = z.pivot(index="seed",columns="method",values="value")
    interaction = (pivot["RO-PDRF"]-pivot["PDRF"])-(pivot["RO-CAGF"]-pivot["CAGF"])
    stat, p = wilcoxon(interaction, method="exact")
    mean, lo, hi = ci_bootstrap(interaction)
    effects.append({"dataset":"chemical","subset":"affected","metric":"macro_auroc",
                    "environment":"type:silent_gaussian",
                    "contrast":"interaction: bounded recovery gain - gate recovery gain",
                    "n":len(interaction),"mean_difference":mean,"ci_low":lo,"ci_high":hi,
                    "wins":int((interaction>0).sum()),"losses":int((interaction<0).sum()),
                    "wilcoxon_p":float(p)})
    effects.extend([
        paired(m,"RO-PDRF","RO-CAGF","hydraulic"),
        paired(m,"RO-PDRF","PDRF","hydraulic"),
        paired(m,"RO-CAGF","CAGF","hydraulic"),
    ])
    eff = pd.DataFrame(effects)
    eff.to_csv(SRC/"major3_objective_matched_effects.csv",index=False)
    ni=eff[(eff.contrast=="RO-PDRF - RO-CAGF")].copy()
    ni["noninferiority_margin"]=-.01
    ni["descriptive_noninferiority_met"]=ni.ci_low>ni.noninferiority_margin
    ni.to_csv(SRC/"major3_descriptive_noninferiority.csv",index=False)
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",
           r"Dataset & Contrast & Mean $\Delta$ AUROC (95\% CI) & Wins & Losses & $P$ \\",r"\midrule"]
    for _,x in eff.iterrows():
        lines.append(f"{x.dataset.capitalize()} & {display_name(x.contrast.replace('_',' '))} & "
                     f"{x.mean_difference:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}] & "
                     f"{int(x.wins)} & {int(x.losses)} & {x.wilcoxon_p:.4f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_2x2_effects.tex").write_text("\n".join(lines),encoding="utf-8")

    # Correlated stress environments are summarized descriptively at the
    # environment level; they are not treated as 170 independent replicates.
    envz=m[(m.dataset=="chemical")&(m.subset=="affected")&(m.metric=="macro_auroc")]
    envp=envz.pivot_table(index=["environment","seed"],columns="method",values="value")
    erows=[]
    for left,right in (("RO-PDRF","RO-CAGF"),("RO-PDRF","PDRF"),("RO-CAGF","CAGF")):
        d=(envp[left]-envp[right]).groupby("environment").mean()
        stat,p=wilcoxon(d,method="exact");mean,lo,hi=ci_bootstrap(d)
        erows.append({"contrast":f"{left} - {right}","environments":len(d),
                      "mean_difference":mean,"ci_low":lo,"ci_high":hi,
                      "positive_environments":int((d>0).sum()),
                      "negative_environments":int((d<0).sum()),
                      "descriptive_wilcoxon_p":float(p)})
    env_eff=pd.DataFrame(erows)
    env_eff.to_csv(SRC/"major3_chemical_environment_effects.csv",index=False)
    lines=[r"\begin{tabular}{lrrrr}",r"\toprule",
           r"Contrast & Environment mean $\Delta$ (95\% CI) & Positive & Negative & Descriptive $P$ \\",r"\midrule"]
    for _,x in env_eff.iterrows():
        lines.append(f"{display_name(x.contrast)} & {x.mean_difference:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}] & "
                     f"{int(x.positive_environments)} & {int(x.negative_environments)} & "
                     f"{x.descriptive_wilcoxon_p:.4f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_chemical_environment_effects.tex").write_text("\n".join(lines),encoding="utf-8")

    hyd = m[(m.dataset=="hydraulic")&(m.subset=="affected")&(m.metric=="macro_auroc")]
    hp = hyd.groupby(["task","fault_type","method"]).value.agg(["mean","std"]).reset_index()
    lines=[r"\begin{tabular}{llrrr}",r"\toprule",
           r"Task & Fault & RO-CAGF & RO-PDRF-Full & Difference \\",r"\midrule"]
    hyd_diff=[]
    available_tasks=[x for x in ("cooler","valve","pump","accumulator") if x in set(hp.task)]
    for task in available_tasks:
        for fault in ("pressure","vibration"):
            a=hp[(hp.task==task)&(hp.fault_type==fault)&(hp.method=="RO-CAGF")].iloc[0]
            b=hp[(hp.task==task)&(hp.fault_type==fault)&(hp.method=="RO-PDRF")].iloc[0]
            d=b["mean"]-a["mean"];hyd_diff.append((task,fault,d))
            lines.append(f"{task.capitalize()} & {fault.capitalize()} & {pm(a['mean'],a['std'])} & "
                         f"{pm(b['mean'],b['std'])} & {d:+.3f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_2x2_hydraulic.tex").write_text("\n".join(lines),encoding="utf-8")

    # The guarded blocked split is the main hydraulic result. The original
    # stratified table above is retained only for supplementary protocol
    # transparency.
    blocked = pd.read_csv(SRC/"major5_hydraulic_blocked_metrics.csv")
    blocked = blocked[(blocked.subset=="affected")&
                      (blocked.metric=="macro_auroc")]
    bp = blocked.groupby(["task","fault","method"]).value.agg(
        ["mean","std"]).reset_index()
    lines=[r"\begin{tabular}{llrrr}",r"\toprule",
           r"Task & Fault & RO-CAGF & RO-PDRF-Full & Difference \\",r"\midrule"]
    blocked_diff=[]
    for task in available_tasks:
        for fault in ("pressure","vibration"):
            a=bp[(bp.task==task)&(bp.fault==fault)&
                 (bp.method=="RO-CAGF")].iloc[0]
            b=bp[(bp.task==task)&(bp.fault==fault)&
                 (bp.method=="RO-PDRF")].iloc[0]
            d=b["mean"]-a["mean"]
            blocked_diff.append((task,fault,d))
            lines.append(f"{task.capitalize()} & {fault.capitalize()} & "
                         f"{pm(a['mean'],a['std'])} & {pm(b['mean'],b['std'])} & "
                         f"{d:+.3f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major7_hydraulic_blocked_main.tex").write_text(
        "\n".join(lines),encoding="utf-8")

    cost = c.groupby(["dataset","method"]).agg(
        parameters=("parameters","first"), train_mean=("train_seconds","mean"),
        train_sd=("train_seconds","std"),
        infer_mean=("inference_ms_per_observation","mean"),
        infer_sd=("inference_ms_per_observation","std")).reset_index()
    cost.to_csv(SRC/"major3_objective_matched_cost_summary.csv",index=False)
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",
           r"Dataset & Method & Parameters & Train time (s) & Inference (ms/obs.) & Training paths \\",r"\midrule"]
    for _,x in cost.iterrows():
        lines.append(f"{x.dataset.capitalize()} & {display_name(x.method)} & {int(x.parameters):,} & "
                     f"{x.train_mean:.1f} $\\pm$ {x.train_sd:.1f} & "
                     f"{x.infer_mean:.4f} $\\pm$ {x.infer_sd:.4f} & 3"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_2x2_cost.tex").write_text("\n".join(lines),encoding="utf-8")

    # Figure contract: objective matching identifies whether recovery benefit
    # belongs to the architecture or to the training objective.
    fig, axs = plt.subplots(2,2,figsize=(7.2,6.0))
    ax=axs[0,0]
    for arch, pair, color, marker in [
        ("Unrestricted gate",("CAGF","RO-CAGF"),"#D98256","o"),
        ("Bounded factorized",("PDRF","RO-PDRF"),"#2F8F6B","s")]:
        means=[];ses=[]
        for method in pair:
            v=z[z.method==method].value
            means.append(v.mean());ses.append(v.std())
        ax.errorbar([0,1],means,yerr=ses,marker=marker,capsize=2,color=color,label=arch)
    ax.set_xticks([0,1],["Original objective","Recovery objective"])
    ax.set_ylabel("Affected macro-AUROC")
    ax.set_title("a  Recovery-objective-matched 2x2",loc="left",fontweight="bold")
    ax.legend()

    ax=axs[0,1]
    env=m[(m.dataset=="chemical")&(m.subset=="affected")&(m.metric=="macro_auroc")]
    ep=env.pivot_table(index=["environment","seed"],columns="method",values="value")
    es=(ep["RO-PDRF"]-ep["RO-CAGF"]).groupby("environment").agg(["mean","std"]).sort_values("mean")
    def stress_label(value):
        if value.startswith("type:"):
            return "Fault type: "+value.split(":",1)[1].replace("_"," ")
        if value.startswith("prevalence:"):
            return f"Affected prevalence: {float(value.split(':',1)[1]):.2f}"
        if value.startswith("group:"):
            return "Affected group: "+value.split(":",1)[1]
        if value.startswith("scale:"):
            return "Gaussian scale: "+value.split(":",1)[1]
        return value.replace("_"," ")
    y=np.arange(len(es));ax.errorbar(es["mean"],y,xerr=es["std"],fmt="o",ms=3,color="#4C78A8",capsize=1.5)
    ax.axvline(0,color="#555",ls="--",lw=.7);ax.set_yticks(y,[stress_label(x) for x in es.index]);ax.tick_params(axis="y",labelsize=5.7)
    ax.set_xlabel("RO-PDRF-Full minus RO-CAGF AUROC");ax.set_title("b  Correlated chemical stress conditions",loc="left",fontweight="bold",fontsize=7.5)

    ax=axs[1,0]
    h=pd.DataFrame(blocked_diff,columns=["task","fault","difference"]).pivot(index="task",columns="fault",values="difference").reindex(available_tasks)
    im=ax.imshow(h.values,cmap="RdBu_r",vmin=-max(.02,np.abs(h.values).max()),vmax=max(.02,np.abs(h.values).max()),aspect="auto")
    ax.set_xticks(range(len(h.columns)),[x.capitalize() for x in h.columns]);ax.set_yticks(range(len(h)),[x.capitalize() for x in h.index])
    for i in range(len(h)):
        for j in range(2): ax.text(j,i,f"{h.values[i,j]:+.3f}",ha="center",va="center",fontsize=6)
    ax.set_title("c  Guarded blocked hydraulic split",loc="left",fontweight="bold")
    # The signed values are printed in every cell; omit a vertical colorbar
    # label to preserve separation from panel d in double-column export.
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)

    ax=axs[1,1]
    strat=eff[(eff.dataset=="hydraulic")&
              (eff.contrast=="RO-PDRF - RO-CAGF")].iloc[0]
    blocked_summary=json.loads((SRC/"major5_hydraulic_blocked_summary.json").read_text())
    means=np.array([strat.mean_difference,blocked_summary["mean"]])
    lo=np.array([strat.ci_low,blocked_summary["ci_low"]])
    hi=np.array([strat.ci_high,blocked_summary["ci_high"]])
    xpos=np.arange(2)
    ax.errorbar(xpos,means,yerr=np.vstack([means-lo,hi-means]),fmt="o",
                ms=5,color="#4C78A8",capsize=3)
    ax.axhline(0,color="#555",ls="--",lw=.7)
    ax.set_xticks(xpos,["Random\nstratified","Guarded\nblocked"])
    ax.set_ylabel("RO-PDRF-Full minus RO-CAGF AUROC")
    ax.set_title("d  Hydraulic split sensitivity",loc="left",fontweight="bold")
    save(fig,"figure7_objective_matched")


def ahu_analysis():
    m = pd.read_csv(SRC/"major3_ahu_adaptation_metrics.csv")
    shift = pd.read_csv(SRC/"major3_ahu_global_shift.csv")
    prior = pd.read_csv(SRC/"major3_ahu_prior_shift.csv")
    feat = pd.read_csv(SRC/"major3_ahu_feature_shift.csv")
    if "feature" not in feat:
        feat["feature"] = [f"group_{g}_position_{p}" for g,p in zip(feat.group,feat.position)]
    pca = pd.read_csv(SRC/"major3_ahu_pca.csv")
    metric = m[m.metric.isin(["macro_auroc","macro_auprc","reversed_auroc","nll","ece15"])]
    summary=metric.groupby(["held_out_building","method","regime","metric"]).value.agg(["mean","std"]).reset_index()
    summary.to_csv(SRC/"major3_ahu_adaptation_summary.csv",index=False)
    shown=[("EF-PD","source_norm"),("GroupDRO-EF-PD","source_only"),
           ("HGB-current","source_only"),("RO-PDRF-NOQ","source_norm"),
           ("RO-PDRF-NOQ","target_norm"),("RO-PDRF-NOQ","coral_adapt")]
    lines=[r"\begin{tabular}{lllrrr}",r"\toprule",
           r"Held-out & Method & Regime & AUROC & Reverse AUROC & AUPRC \\",r"\midrule"]
    for building in ("auditorium","hospital","office"):
        for method,regime in shown:
            vals=[]
            for met in ("macro_auroc","reversed_auroc","macro_auprc"):
                z=summary[(summary.held_out_building==building)&(summary.method==method)&
                          (summary.regime==regime)&(summary.metric==met)]
                vals.append((z.iloc[0]["mean"],z.iloc[0]["std"]))
            lines.append(f"{building.capitalize()} & {method} & {regime.replace('_',' ')} & "+
                         " & ".join(pm(a,b) for a,b in vals)+r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_ahu_adaptation.tex").write_text("\n".join(lines),encoding="utf-8")

    lines=[r"\begin{tabular}{lrrr}",r"\toprule",
           r"Held-out building & Linear MMD & RBF MMD & CORAL distance \\",r"\midrule"]
    for _,x in shift.iterrows():
        lines.append(f"{x.held_out_building.capitalize()} & {x.linear_mmd:.3f} & {x.rbf_mmd:.3f} & {x.coral_distance:.5f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_ahu_shift_distance.tex").write_text("\n".join(lines),encoding="utf-8")

    hosp=feat[feat.held_out_building=="hospital"].copy()
    hosp["abs_smd"]=hosp.standardized_mean_difference.abs()
    top=hosp.sort_values(["abs_smd","ks_statistic"],ascending=False).head(8)
    lines=[r"\begin{tabular}{lrrrr}",r"\toprule",
           r"Feature & Standardized mean difference & KS & Wasserstein & Target outside source 1--99\% \\",r"\midrule"]
    for _,x in top.iterrows():
        lines.append(f"{x.feature.replace('_',' ')} & {x.standardized_mean_difference:+.2f} & "
                     f"{x.ks_statistic:.2f} & {x.wasserstein:.2f} & {100*x.target_outside_source_1_99:.1f}\\%"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_hospital_feature_shift.tex").write_text("\n".join(lines),encoding="utf-8")

    # Shift figure contract: identify the dominant hospital shift and test
    # whether source-only robustness or unlabeled adaptation repairs it.
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.7))
    ax=axs[0,0];x=np.arange(3);w=.25
    for j,col in enumerate(["linear_mmd","rbf_mmd","coral_distance"]):
        v=shift.set_index("held_out_building").reindex(["auditorium","hospital","office"])[col]
        vv=v/v.max()
        ax.bar(x+(j-1)*w,vv,w,label=col.replace("_"," "))
    ax.set_xticks(x,["Auditorium","Hospital","Office"],rotation=15,ha="right");ax.set_ylabel("Distance normalized to maximum")
    ax.legend(fontsize=5.5);ax.set_title("a  Cross-building shift magnitude",loc="left",fontweight="bold")

    ax=axs[0,1];z=pca[pca.held_out_building=="hospital"]
    for domain,color in (("source","#8F8F8F"),("target","#D55E00")):
        q=z[z.domain==domain];ax.scatter(q.pc1,q.pc2,s=2,alpha=.20,color=color,label=domain)
    ax.set_xlabel("PC1");ax.set_ylabel("PC2");ax.legend();ax.set_title("b  Hospital source-target embedding",loc="left",fontweight="bold")

    ax=axs[0,2]
    hs=summary[(summary.held_out_building=="hospital")&(summary.metric.isin(["macro_auroc","reversed_auroc"]))]
    order=shown
    labels=[];normal=[];reverse=[]
    for method,regime in order:
        q=hs[(hs.method==method)&(hs.regime==regime)]
        labels.append((method+"\n"+regime.replace("_"," ")).replace(
            "RO-PDRF-NOQ", "RO-PDRF-Full (no q)"))
        normal.append(q[q.metric=="macro_auroc"]["mean"].iloc[0]);reverse.append(q[q.metric=="reversed_auroc"]["mean"].iloc[0])
    short=["EF ERM","GroupDRO","HGB","RO ERM","RO target norm.","RO CORAL"]
    yy=np.arange(len(order));ax.barh(yy-.18,normal,.36,color="#4C78A8",label="Original");ax.barh(yy+.18,reverse,.36,color="#D98256",label="Reversed")
    ax.axvline(.5,color="#555",ls="--",lw=.7);ax.set_yticks(yy,short);ax.invert_yaxis();ax.set_xlabel("AUROC");ax.legend()
    ax.set_title("c  Hospital ranking reversal",loc="left",fontweight="bold")

    ax=axs[1,0]
    src=summary[(summary.metric=="macro_auroc")&(((summary.method=="EF-PD")&(summary.regime=="source_norm"))|
        ((summary.method=="RO-PDRF-NOQ")&(summary.regime=="source_norm"))|
        ((summary.method=="GroupDRO-EF-PD")&(summary.regime=="source_only"))|
        ((summary.method=="HGB-current")&(summary.regime=="source_only")))]
    builds=["auditorium","hospital","office"];methods=["EF-PD","GroupDRO-EF-PD","HGB-current","RO-PDRF-NOQ"];w=.19;xx=np.arange(3)
    for j,method in enumerate(methods):
        q=src[src.method==method].set_index("held_out_building").reindex(builds)
        ax.bar(xx+(j-1.5)*w,q["mean"],w,yerr=q["std"],capsize=1,color=COL[method],label=method.replace("-NOQ",""))
    ax.set_xticks(xx,[x.capitalize() for x in builds],rotation=15,ha="right");ax.set_ylim(0,1.05);ax.set_ylabel("Macro-AUROC");ax.legend(fontsize=5.2,ncol=2,loc="lower left")
    ax.set_title("d  Source-only generalization",loc="left",fontweight="bold")

    ax=axs[1,1]
    for method,color in (("EF-PD","#0072B2"),("RO-PDRF-NOQ","#E69F00")):
        q=summary[(summary.held_out_building=="hospital")&(summary.method==method)&
                  (summary.metric=="macro_auroc")].set_index("regime").reindex(["source_norm","target_norm","coral_adapt"])
        ax.plot(range(3),q["mean"],marker="o",color=color,label=method.replace("-NOQ",""))
    ax.axhline(.5,color="#555",ls="--",lw=.7);ax.set_xticks(range(3),["Source norm.","Target norm.","CORAL"],rotation=15)
    ax.set_ylabel("Hospital macro-AUROC");ax.legend();ax.set_title("e  Unlabeled adaptation",loc="left",fontweight="bold")

    ax=axs[1,2]
    q=prior[prior.scope.isin(["source_pool","target_test"])].copy()
    builds=["auditorium","hospital","office"];xx=np.arange(3);w=.35
    s=q[q.scope=="source_pool"].set_index("held_out_building").reindex(builds)
    t=q[q.scope=="target_test"].set_index("held_out_building").reindex(builds)
    ax.bar(xx-w/2,s.fault_prevalence,w,color="#8F8F8F",label="Source pool")
    ax.bar(xx+w/2,t.fault_prevalence,w,color="#D55E00",label="Target")
    ax.set_xticks(xx,[x.capitalize() for x in builds],rotation=15,ha="right");ax.set_ylabel("Fault prevalence");ax.legend()
    ax.set_title("f  Label-prior shift (audit only)",loc="left",fontweight="bold")
    save(fig,"figureS2_ahu_shift_adaptation")

    # Compact main-text version: the field experiment is a negative boundary,
    # so the figure foregrounds source-only failure and ranking reversal.
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.5))
    ax=axs[0,0]
    for j,method in enumerate(methods):
        q=src[src.method==method].set_index("held_out_building").reindex(builds)
        ax.bar(xx+(j-1.5)*w,q["mean"],w,yerr=q["std"],capsize=1,
               color=COL[method],label=method.replace("-NOQ",""))
    ax.set_xticks(xx,[x.capitalize() for x in builds],rotation=12,ha="right")
    ax.set_ylim(0,1.05);ax.set_ylabel("Macro-AUROC");ax.legend(fontsize=5.5,ncol=2,loc="lower left")
    ax.set_title("a  Source-only leave-one-building-out",loc="left",fontweight="bold")

    ax=axs[0,1]
    ax.barh(yy-.18,normal,.36,color="#4C78A8",label="Original")
    ax.barh(yy+.18,reverse,.36,color="#D98256",label="Reversed")
    ax.axvline(.5,color="#555",ls="--",lw=.7);ax.set_yticks(yy,short);ax.invert_yaxis()
    ax.set_xlabel("Hospital AUROC");ax.legend()
    ax.set_title("b  Stable ranking reversal",loc="left",fontweight="bold")

    ax=axs[1,0]
    for method,color in (("EF-PD","#0072B2"),("RO-PDRF-NOQ","#E69F00")):
        q=summary[(summary.held_out_building=="hospital")&(summary.method==method)&
                  (summary.metric=="macro_auroc")].set_index("regime").reindex(["source_norm","target_norm","coral_adapt"])
        ax.errorbar(range(3),q["mean"],yerr=q["std"],marker="o",capsize=2,color=color,label=method.replace("-NOQ",""))
    ax.axhline(.5,color="#555",ls="--",lw=.7);ax.set_xticks(range(3),["Source norm.","Target norm.","CORAL"])
    ax.set_ylabel("Hospital macro-AUROC");ax.legend();ax.set_title("c  Unlabeled adaptation did not repair RO",loc="left",fontweight="bold")

    ax=axs[1,1];x=np.arange(3);ww=.25
    for j,col in enumerate(["linear_mmd","rbf_mmd","coral_distance"]):
        v=shift.set_index("held_out_building").reindex(builds)[col];vv=v/v.max()
        ax.bar(x+(j-1)*ww,vv,ww,label=col.replace("_"," "))
    ax.set_xticks(x,[x.capitalize() for x in builds],rotation=12,ha="right")
    ax.set_ylabel("Distance normalized to maximum");ax.legend(fontsize=5.5)
    ax.set_title("d  Hospital has the largest aggregate shift",loc="left",fontweight="bold")
    save(fig,"figure8_ahu_shift_main")


def ablation_analysis():
    teacher=pd.read_csv(SRC/"major3_teacher_ablation.csv")
    loss=pd.read_csv(SRC/"major3_loss_ablation.csv")
    ts=teacher[(teacher.subset=="affected")&teacher.metric.isin(["macro_auroc","macro_auprc","nll"])]
    ta=ts.groupby(["teacher","metric"]).value.agg(["mean","std"]).reset_index()
    order=["clean_only","best_ce","removal_only","confidence_selected"]
    lines=[r"\begin{tabular}{lrrr}",r"\toprule",
           r"Teacher selection & Aff. AUROC & Aff. AUPRC & Aff. NLL \\",r"\midrule"]
    for method in order:
        vals=[]
        for metric in ("macro_auroc","macro_auprc","nll"):
            x=ta[(ta.teacher==method)&(ta.metric==metric)].iloc[0]
            vals.append(pm(x["mean"],x["std"]))
        lines.append(method.replace("_"," ")+" & "+" & ".join(vals)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_teacher_ablation.tex").write_text("\n".join(lines),encoding="utf-8")

    l=loss[(loss.subset=="affected")&(loss.metric=="macro_auroc")]
    lp=l.pivot(index="seed",columns="variant",values="value")
    rows=[]
    for variant in ("minus_recovery","minus_brier","minus_interior","minus_auc"):
        d=lp["full"]-lp[variant]
        stat,p=wilcoxon(d,method="exact")
        mean,lo,hi=ci_bootstrap(d)
        rows.append({"omitted":variant.replace("minus_",""),"mean_difference":mean,
                     "ci_low":lo,"ci_high":hi,"wins":int((d>0).sum()),
                     "losses":int((d<0).sum()),"p":float(p)})
    ld=pd.DataFrame(rows);ld.to_csv(SRC/"major3_loss_ablation_effects.csv",index=False)
    lines=[r"\begin{tabular}{lrrrr}",r"\toprule",
           r"Omitted term & Full-minus-ablated AUROC (95\% CI) & Wins & Losses & $P$ \\",r"\midrule"]
    for _,x in ld.iterrows():
        lines.append(f"{x.omitted.capitalize()} & {x.mean_difference:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}] & "
                     f"{int(x.wins)} & {int(x.losses)} & {x.p:.4f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major3_loss_ablation.tex").write_text("\n".join(lines),encoding="utf-8")

    fig,axs=plt.subplots(1,2,figsize=(7.2,2.7))
    ax=axs[0]
    vals=[];err=[]
    for method in order:
        x=ta[(ta.teacher==method)&(ta.metric=="macro_auroc")].iloc[0]
        vals.append(x["mean"]);err.append(x["std"])
    ax.bar(range(4),vals,yerr=err,color=["#2F8F6B","#4C78A8","#D98256","#8F8F8F"],capsize=2)
    ax.set_xticks(range(4),["Clean only\n(formal)","Label-assisted\nbest view","Removal only","Confidence\nselected"],rotation=10)
    ax.set_ylabel("Affected macro-AUROC");ax.set_title("a  Recovery-teacher audit",loc="left",fontweight="bold")
    ax=axs[1];yy=np.arange(len(ld))
    ax.errorbar(ld.mean_difference,yy,xerr=np.vstack([ld.mean_difference-ld.ci_low,ld.ci_high-ld.mean_difference]),
                fmt="o",color="#2F8F6B",capsize=2)
    ax.axvline(0,color="#555",ls="--",lw=.7);ax.set_yticks(yy,[x.capitalize() for x in ld.omitted])
    ax.set_xlabel("Full − term-ablated affected AUROC");ax.set_title("b  Leave-one-loss-out audit",loc="left",fontweight="bold")
    save(fig,"figureS3_recovery_objective_ablation")


def protocol_regularizer_and_ensemble_analysis():
    """Resolve protocol duplication and summarize the clean-teacher headline run."""
    main = pd.read_csv(SRC/"major3_objective_matched_metrics.csv")
    q1 = pd.read_csv(SRC/"q1_risk_metrics.csv")
    archived = pd.read_csv(SRC/"archived_bestce_seed85000_metrics.csv")

    principal = main[(main.dataset=="chemical") &
                     (main.environment=="type:silent_gaussian")]
    old_main = q1[(q1.subset=="affected") & (q1.metric=="macro_auroc")]
    old_rerun = archived[(archived.dataset=="chemical") &
                         (archived.environment=="type:silent_gaussian") &
                         (archived.subset=="affected") &
                         (archived.metric=="macro_auroc")]
    new_main = principal[(principal.subset=="affected") &
                         (principal.metric=="macro_auroc")]
    protocol_rows=[]
    for label, frame, mapping in (
        ("Original formal run", old_main,
         {"CAGF":"CAGF","PDRF":"PDRF","RO-PDRF":"PDRF_RCIA_R30"}),
        ("Superseded recovery-objective rerun", old_rerun,
         {"CAGF":"CAGF","PDRF":"PDRF","RO-PDRF":"RO-PDRF"}),
        ("Unified headline protocol", new_main,
         {"CAGF":"CAGF","PDRF":"PDRF","RO-PDRF":"RO-PDRF"}),
    ):
        for display, stored in mapping.items():
            z=frame[frame.method==stored].value
            if len(z):
                protocol_rows.append({"protocol":label,"method":display,
                                      "mean":z.mean(),"std":z.std(),"n":len(z)})
    pd.DataFrame(protocol_rows).to_csv(SRC/"major4_protocol_run_summary.csv",index=False)

    lines=[r"\begin{tabular}{llll}",r"\toprule",
           r"Protocol field & Original formal run & Superseded rerun & Unified headline \\",r"\midrule",
           r"Purpose & recovery/calibration audit & initial 2$\times$2 audit & sole main-paper protocol \\",
           r"Recovery teacher & label-assisted best view & label-assisted best view & clean view \\",
           r"Chemical seeds & 101--110 & 101--110 & 101--110 \\",
           r"Data split indices & batches 1--6 / 7 / 8--10 & same & same \\",
           r"Initialization & PyTorch seed 101--110 & same & same \\",
           r"Test-fault seed & 70001 & 85000 & 70001 \\",
           r"Fault / group & Gaussian scale 3 / group 1 & Gaussian scale 3 / group 1 & Gaussian scale 3 / group 1 \\",
           r"Affected / missing & 40\% / 20\% & 40\% / 20\% & 40\% / 20\% \\",
           r"Quality at test & neutral ($q=1$) & neutral ($q=1$) & neutral ($q=1$) \\",
           r"Checkpoint & natural selection CE & natural selection CE & natural selection CE \\",
           r"Training / inference paths & 3 / 1 & 3 / 1 & 3 / 1 \\",
           r"Evaluation subset & fixed affected rows & different fixed affected rows & fixed affected rows \\",
           r"Status & supplementary sensitivity & archived / superseded & headline \\",
           r"\bottomrule",r"\end{tabular}"]
    (TAB/"major4_protocol_reconciliation.tex").write_text("\n".join(lines),encoding="utf-8")

    # Check exact reproducibility for unchanged CAGF/PDRF after aligning the
    # fixed fault. Any non-zero difference would indicate a remaining branch.
    rec=[]
    for method in ("CAGF","PDRF"):
        a=old_main[old_main.method==method].set_index("seed").value
        b=new_main[new_main.method==method].set_index("seed").value
        common=a.index.intersection(b.index);d=b.loc[common]-a.loc[common]
        rec.append({"method":method,"n":len(common),"max_abs_difference":abs(d).max(),
                    "mean_difference":d.mean()})
    pd.DataFrame(rec).to_csv(SRC/"major4_protocol_reproducibility_check.csv",index=False)

    # Architecture-specific regularizer sensitivity; the formal .005 rows are
    # reused from the headline fit to avoid duplicate training.
    sens=pd.read_csv(SRC/"major4_regularizer_metrics.csv")
    s=sens[(sens.subset=="affected")&(sens.metric=="macro_auroc")][["seed","method","value"]]
    add=[]
    for method,newname in (("RO-CAGF","RO-CAGF KL=.005"),
                           ("RO-PDRF","RO-PDRF interior=.005")):
        z=new_main[new_main.method==method][["seed","value"]].copy();z["method"]=newname;add.append(z)
    s=pd.concat([s]+add,ignore_index=True)
    summary=s.groupby("method").value.agg(["mean","std"]).reset_index()
    summary.to_csv(SRC/"major4_regularizer_summary.csv",index=False)
    order_c=["RO-CAGF KL=0","RO-CAGF KL=.001","RO-CAGF KL=.005","RO-CAGF KL=.020"]
    order_p=["RO-PDRF interior=0","RO-PDRF interior=.001","RO-PDRF interior=.005","RO-PDRF interior=.020"]
    pivot=s.pivot(index="seed",columns="method",values="value")
    matrix=[]
    for pmeth in order_p:
        for cmeth in order_c:
            d=pivot[pmeth]-pivot[cmeth];_,p=wilcoxon(d,method="exact")
            mean,lo,hi=ci_bootstrap(d)
            matrix.append({"pdrf_setting":pmeth,"cagf_setting":cmeth,
                           "mean_difference":mean,"ci_low":lo,"ci_high":hi,
                           "wins":int((d>0).sum()),"losses":int((d<0).sum()),"p":float(p)})
    mat=pd.DataFrame(matrix);mat.to_csv(SRC/"major4_regularizer_contrast_matrix.csv",index=False)
    lines=[r"\begin{tabular}{lrr}",r"\toprule",
           r"Recovery model / regularizer weight & Aff. AUROC & SD \\",r"\midrule"]
    for method in order_c+order_p:
        x=summary[summary.method==method].iloc[0]
        lines.append(f"{display_name(method)} & {x['mean']:.3f} & {x['std']:.3f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major4_regularizer_sensitivity.tex").write_text("\n".join(lines),encoding="utf-8")

    # Clean-teacher ensemble and probability-quality audit.
    pred=pd.read_csv(SRC/"major3_objective_matched_predictions.csv")
    pcols=[x for x in pred.columns if x.startswith("p") and x[1:].isdigit()]
    ens_rows=[];div_rows=[];ensemble={}
    for method in ("CAGF","RO-CAGF","PDRF","RO-PDRF"):
        z=pred[pred.method==method]
        arr=np.stack([g.sort_values("sample")[pcols].to_numpy()
                      for _,g in z.groupby("seed")])
        meta=z[z.seed==z.seed.min()].sort_values("sample")
        y=meta.y.to_numpy();affected=meta.affected.astype(bool).to_numpy()
        probability=arr.mean(0);ensemble[method]=probability
        for subset,take in (("all",np.ones(len(y),bool)),("affected",affected),
                            ("unaffected",~affected)):
            for metric,value in risk.all_metrics(y[take],probability[take]).items():
                ens_rows.append({"method":method,"subset":subset,"metric":metric,"value":value})
        disagreements=[]
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                disagreements.append((arr[i].argmax(1)!=arr[j].argmax(1)).mean())
        div_rows.append({"method":method,"pairwise_disagreement":np.mean(disagreements)})
    pd.DataFrame(ens_rows).to_csv(SRC/"major4_headline_ensemble_metrics.csv",index=False)
    pd.DataFrame(div_rows).to_csv(SRC/"major4_headline_ensemble_diversity.csv",index=False)
    rng=np.random.default_rng(94001);boot=[]
    meta=pred[(pred.seed==pred.seed.min())&(pred.method=="RO-PDRF")].sort_values("sample")
    y=meta.y.to_numpy();affected=meta.affected.astype(bool).to_numpy();classes=np.arange(len(pcols))
    for subset,take in (("all",np.ones(len(y),bool)),("affected",affected)):
        idx=np.flatnonzero(take);yb=label_binarize(y[idx],classes=classes)
        for _ in range(5000):
            draw=rng.integers(0,len(idx),len(idx));yy=yb[draw]
            a=roc_auc_score(yy,ensemble["RO-PDRF"][idx][draw],average="macro",multi_class="ovr")
            b=roc_auc_score(yy,ensemble["RO-CAGF"][idx][draw],average="macro",multi_class="ovr")
            boot.append({"subset":subset,"difference":a-b})
    boot=pd.DataFrame(boot);boot.to_csv(SRC/"major4_headline_ensemble_bootstrap.csv",index=False)
    bs=boot.groupby("subset").difference.agg(
        mean="mean",ci_low=lambda x:x.quantile(.025),ci_high=lambda x:x.quantile(.975)).reset_index()
    bs.to_csv(SRC/"major4_headline_ensemble_effects.csv",index=False)

    mechanism=pd.read_csv(SRC/"major3_objective_matched_mechanism.csv")
    pd.DataFrame([{
        "outer_5pct":mechanism.outer_5pct.mean(),
        "weight_reduction_rate":(mechanism.weight_reduction>0).mean(),
        "failure_despite_weight_reduction":((mechanism.weight_reduction>0)&(~mechanism.fault_correct)).mean(),
        "mean_score_change":mechanism.score_change.mean(),
    }]).to_csv(SRC/"major4_headline_mechanism_summary.csv",index=False)

    # Supplementary visual: regularizer response and all architecture contrasts.
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    ax=axs[0]
    for order,label,color,marker in ((order_c,"RO-CAGF","#D98256","o"),
                                     (order_p,"RO-PDRF-Full","#2F8F6B","s")):
        vals=[summary[summary.method==m].iloc[0] for m in order]
        positions=np.arange(4)
        ax.errorbar(positions,[x['mean'] for x in vals],yerr=[x['std'] for x in vals],
                    marker=marker,capsize=2,label=label,color=color)
    ax.set_xticks(range(4),["0",".001",".005",".020"])
    ax.set_xlabel("Architecture-specific regularizer weight")
    ax.set_ylabel("Affected macro-AUROC");ax.set_title("a  Regularizer sensitivity",loc="left",fontweight="bold");ax.legend()
    ax=axs[1]
    hm=mat.pivot(index="pdrf_setting",columns="cagf_setting",values="mean_difference").reindex(index=order_p,columns=order_c)
    vmax=max(.01,np.abs(hm.values).max());im=ax.imshow(hm.values,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(4),["0",".001",".005",".020"]);ax.set_yticks(range(4),["0",".001",".005",".020"])
    ax.set_xlabel("RO-CAGF KL weight");ax.set_ylabel("RO-PDRF-Full interior weight")
    for i in range(4):
        for j in range(4):ax.text(j,i,f"{hm.values[i,j]:+.3f}",ha="center",va="center",fontsize=6)
    ax.set_title("b  RO-PDRF-Full minus RO-CAGF",loc="left",fontweight="bold")
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Affected AUROC difference")
    save(fig,"figureS4_regularizer_sensitivity")


def main():
    objective_analysis()
    ahu_analysis()
    ablation_analysis()
    protocol_regularizer_and_ensemble_analysis()
    print("Major-3 tables, statistics and Python figures written.")


if __name__=="__main__":
    main()
