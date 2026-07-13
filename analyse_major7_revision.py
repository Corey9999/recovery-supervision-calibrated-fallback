"""Analyse the manuscript-15 strong-baseline and teacher-remedy experiments."""
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


ROOT = Path(__file__).resolve().parent
SRC, TAB, FIG = ROOT/"source_data", ROOT/"tables", ROOT/"figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],
    "svg.fonttype":"none", "pdf.fonttype":42, "font.size":7.2,
    "axes.titlesize":8.2, "axes.labelsize":7.2,
    "xtick.labelsize":6.5, "ytick.labelsize":6.5,
    "legend.fontsize":6.3, "axes.spines.top":False,
    "axes.spines.right":False, "legend.frameon":False,
})
BLUE="#4C78A8"; GREEN="#2F8F6B"; ORANGE="#D98256"
PURPLE="#7A68A6"; GREY="#777777"; RED="#C94C4C"


def save(fig, name):
    fig.tight_layout(pad=.9)
    for ext, kwargs in (("svg",{}),("pdf",{}),("png",{"dpi":300}),
                        ("tiff",{"dpi":600})):
        fig.savefig(FIG/f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def t_interval(values):
    x=np.asarray(values,float)
    if len(x)<2:
        return float(x.mean()), np.nan, np.nan
    margin=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return float(x.mean()),float(x.mean()-margin),float(x.mean()+margin)


def two_level_interval(frame, difference, reps=5000, seed=15001):
    """Resample realizations and crossed optimization-seed clusters."""
    rng=np.random.default_rng(seed)
    p=frame.pivot_table(index=["fault_realization","seed"],
                        columns="method",values="value")
    d=(p[difference[0]]-p[difference[1]]).unstack("seed")
    raw=d.to_numpy(dtype=float)
    if np.isnan(raw).any():
        raise ValueError("The prespecified realization-by-seed design is incomplete")
    n_real,n_seed=raw.shape
    # Balanced crossed-factor bootstrap, vectorized over replicates.  The same
    # fitted seed is reused across fault realizations, so one seed resample is
    # shared across all selected realizations in a replicate.
    rid=rng.integers(0,n_real,size=(reps,n_real))
    sid=rng.integers(0,n_seed,size=(reps,n_seed))
    selected=raw[rid]
    sampled=np.take_along_axis(selected,sid[:,None,:],axis=2)
    draws=sampled.mean(axis=(1,2))
    mean=float(raw.mean())
    lo,hi=np.quantile(draws,[.025,.975])
    return mean,float(lo),float(hi)


def three_level_interval(frame, difference, reps=5000, seed=15002):
    """Resample fault type, realization within type, and seed within realization."""
    rng=np.random.default_rng(seed)
    p=frame.pivot_table(index=["fault_type","fault_realization","seed"],
                        columns="method",values="value")
    p=(p[difference[0]]-p[difference[1]]).rename("difference").reset_index()
    types=sorted(p.fault_type.unique())
    realizations=sorted(p.fault_realization.unique())
    seeds=sorted(p.seed.unique())
    cube=(p.pivot_table(index=["fault_type","fault_realization"],
                        columns="seed",values="difference")
            .reindex(pd.MultiIndex.from_product([types,realizations],
                                                names=["fault_type","fault_realization"]))
            .reindex(columns=seeds).to_numpy(dtype=float)
            .reshape(len(types),len(realizations),len(seeds)))
    if np.isnan(cube).any():
        raise ValueError("The prespecified fault-type-by-realization-by-seed design is incomplete")
    n_type,n_real,n_seed=cube.shape
    # Vectorized multi-level bootstrap: fault type, realization within the
    # sampled type, and crossed optimization-seed cluster.  Sharing the seed
    # resample across all fault cells preserves the fact that one fitted model
    # is evaluated under every controlled intervention.
    tid=rng.integers(0,n_type,size=(reps,n_type))
    rid=rng.integers(0,n_real,size=(reps,n_type,n_real))
    sid=rng.integers(0,n_seed,size=(reps,n_seed))
    real_sample=cube[tid[:,:,None],rid,:]
    seed_sample=np.take_along_axis(real_sample,sid[:,None,None,:],axis=3)
    draws=seed_sample.mean(axis=(1,2,3))
    mean=float(cube.mean());lo,hi=np.quantile(draws,[.025,.975])
    return mean,float(lo),float(hi)


def principal_method_summary():
    records=[]
    base=pd.read_csv(SRC/"major3_objective_matched_metrics.csv")
    base=base[(base.dataset=="chemical")&
              (base.environment=="type:silent_gaussian")]
    sources=[
        (base,["RO-CAGF","RO-PDRF"]),
        (pd.read_csv(SRC/"major6_strong_control_metrics.csv"),["RO-AT-GATE"]),
        (pd.read_csv(SRC/"major6_ema_metrics.csv"),["RO-PDRF-EMA"]),
        (pd.read_csv(SRC/"major7_qmf_metrics.csv"),["QMF-PD"]),
        (pd.read_csv(SRC/"major7_teacher_agreement_metrics.csv"),
         ["RO-PDRF-CRA","RO-PDRF-ECA"]),
    ]
    for frame, methods in sources:
        z=frame[(frame.subset=="affected")&frame.method.isin(methods)]
        for method in methods:
            row={"method":method}
            for metric in ("macro_auroc","nll","brier","ece15"):
                v=z[(z.method==method)&(z.metric==metric)].value
                row[metric]=float(v.mean())
                row[metric+"_sd"]=float(v.std())
            records.append(row)
    out=pd.DataFrame(records)

    cost_sources=[
        pd.read_csv(SRC/"major3_objective_matched_costs.csv"),
        pd.read_csv(SRC/"major6_strong_control_costs.csv"),
        pd.read_csv(SRC/"major6_ema_costs.csv"),
        pd.read_csv(SRC/"major7_qmf_costs.csv"),
        pd.read_csv(SRC/"major7_teacher_agreement_costs.csv"),
    ]
    costs=pd.concat(cost_sources,ignore_index=True)
    if "dataset" in costs:
        costs=costs[(costs.dataset.isna())|(costs.dataset=="chemical")]
    params=costs.groupby("method").parameters.first()
    out["parameters"]=out.method.map(params)
    out.to_csv(SRC/"major7_principal_method_summary.csv",index=False)

    lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",
           r"Method & Parameters & Affected AUROC & NLL & Brier & ECE \\",r"\midrule"]
    for _,x in out.iterrows():
        lines.append(f"{x.method} & {int(x.parameters):,} & {x.macro_auroc:.3f} & "
                     f"{x.nll:.3f} & {x.brier:.3f} & {x.ece15:.3f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major7_literature_teacher_controls.tex").write_text(
        "\n".join(lines),encoding="utf-8")
    return out


def consistency_effect_table():
    ladder=pd.read_csv(SRC/"major5_consistency_metrics.csv")
    ladder=ladder[(ladder.fault_realization==70001)&
                  (ladder.batch.astype(str)=="all")&
                  (ladder.subset=="affected")&
                  (ladder.metric=="macro_auroc")]
    basic=pd.read_csv(SRC/"major5_basic_distillation_metrics.csv")
    basic=basic[(basic.fault_realization==70001)&
                (basic.batch.astype(str)=="all")&
                (basic.subset=="affected")&
                (basic.metric=="macro_auroc")]
    head=pd.read_csv(SRC/"major3_objective_matched_metrics.csv")
    head=head[(head.dataset=="chemical")&
              (head.environment=="type:silent_gaussian")&
              (head.subset=="affected")&(head.metric=="macro_auroc")]
    rows=[]
    for family in ("EF-PD","UF-PD","CAGF","PDRF"):
        if family in ("EF-PD","UF-PD"):
            source_name=family+"+CD"
            pdv=ladder[ladder.method==source_name].set_index("seed").value
            cd=basic[basic.method==source_name].set_index("seed").value
        else:
            pdv=head[head.method==family].set_index("seed").value
            cd=ladder[ladder.method==family+"+CD"].set_index("seed").value
        stages=[("Paired degradation",pdv)]
        stages.append(("Clean distillation",cd))
        if family in ("CAGF","PDRF"):
            stages.append(("Full recovery",
                           head[head.method=="RO-"+family].set_index("seed").value))
        for stage,v in stages:
            common=v.index.intersection(cd.index)
            delta=(v.loc[common]-cd.loc[common]).to_numpy()
            mean,lo,hi=t_interval(delta)
            p="--" if stage=="Clean distillation" else f"{wilcoxon(delta).pvalue:.4f}"
            rows.append({"architecture":family,"objective":stage,
                         "mean":v.mean(),"sd":v.std(),"delta":mean,
                         "lo":lo,"hi":hi,"p":p})
    out=pd.DataFrame(rows);out.to_csv(
        SRC/"major7_consistency_effects.csv",index=False)
    lines=[r"\begin{tabular}{llrrr}",r"\toprule",
           r"Architecture & Model/training condition & Affected AUROC & $\Delta$ vs clean distillation (95\% CI) & $P$ \\",r"\midrule"]
    display_condition={
        ("EF-PD","Paired degradation"):"EF-PD",
        ("EF-PD","Clean distillation"):"EF-PD + clean distillation",
        ("UF-PD","Paired degradation"):"UF-PD",
        ("UF-PD","Clean distillation"):"UF-PD + clean distillation",
        ("CAGF","Paired degradation"):"CAGF",
        ("CAGF","Clean distillation"):"CAGF + clean distillation",
        ("CAGF","Full recovery"):"RO-CAGF",
        ("PDRF","Paired degradation"):"PDRF",
        ("PDRF","Clean distillation"):"RO-PDRF-Lite",
        ("PDRF","Full recovery"):"RO-PDRF-Full",
    }
    for _,x in out.iterrows():
        effect="Reference" if x.objective=="Clean distillation" else (
            f"{x.delta:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}]")
        condition=display_condition[(x.architecture,x.objective)]
        lines.append(f"{x.architecture} & {condition} & "
                     f"{x['mean']:.3f} $\\pm$ {x.sd:.3f} & {effect} & {x.p}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major7_consistency_ladder_effects.tex").write_text(
        "\n".join(lines),encoding="utf-8")


def conditional_rates():
    old=pd.read_csv(SRC/"major5_teacher_fault_predictions.csv")
    old=old[(old.fault_realization==70001)&old.seed.isin(range(101,106))]
    baseline=old[old.method=="PDRF"].set_index(["seed","sample"])
    standard=old[old.method=="RO-PDRF"].set_index(["seed","sample"])
    frames=[]
    s=pd.DataFrame({
        "teacher_correct":standard.clean_correct,
        "baseline_correct":baseline.fault_correct,
        "model_correct":standard.fault_correct,
        "active":True,
    }).dropna()
    frames.append(("Clean teacher",s))

    new=pd.read_csv(SRC/"major7_teacher_agreement_predictions.csv")
    for method,label in (("RO-PDRF-CRA","Clean/removal agreement"),
                         ("RO-PDRF-ECA","Student/EMA agreement")):
        view=new[(new.method==method)&new.seed.isin(range(101,106))]
        view=view.set_index(["seed","sample"])
        b=baseline.reindex(view.index)
        c=pd.DataFrame({
            "teacher_correct":view.teacher_correct,
            "baseline_correct":b.fault_correct,
            "model_correct":view.fault_correct,
            "active":view.valid_removal & view.teacher_agreement,
        }).dropna()
        frames.append((label,c))

    rows=[]
    for method,z in frames:
        active=z.active.astype(bool)
        recovery=active&z.teacher_correct.astype(bool)&~z.baseline_correct.astype(bool)
        transfer=active&~z.teacher_correct.astype(bool)&z.baseline_correct.astype(bool)
        recovered=recovery&z.model_correct.astype(bool)
        lost=transfer&~z.model_correct.astype(bool)
        rows.append({
            "teacher_rule":method,"active_coverage":float(active.mean()),
            "recovery_eligible":int(recovery.sum()),
            "recovered":int(recovered.sum()),
            "recovery_rate":float(recovered.sum()/max(1,recovery.sum())),
            "transfer_eligible":int(transfer.sum()),
            "lost":int(lost.sum()),
            "error_transfer_rate":float(lost.sum()/max(1,transfer.sum())),
        })
    out=pd.DataFrame(rows);out.to_csv(
        SRC/"major7_teacher_conditional_rates.csv",index=False)
    lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",
           r"Teacher rule & Active coverage & Recovery & Recovery rate & Error transfer & Transfer rate \\",r"\midrule"]
    for _,x in out.iterrows():
        lines.append(f"{x.teacher_rule} & {100*x.active_coverage:.1f}\% & "
                     f"{int(x.recovered)}/{int(x.recovery_eligible)} & "
                     f"{100*x.recovery_rate:.1f}\% & "
                     f"{int(x.lost)}/{int(x.transfer_eligible)} & "
                     f"{100*x.error_transfer_rate:.1f}\%"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major7_teacher_agreement.tex").write_text(
        "\n".join(lines),encoding="utf-8")
    return out


def fault_type_analysis():
    raw=pd.read_csv(SRC/"major7_fault_type_ema_metrics.csv")
    z=raw[(raw.subset=="affected")&
          raw.metric.isin(["macro_auroc","nll","ece15"])]
    summary=z.groupby(["fault_type","method","metric"]).value.agg(
        ["mean","std"]).reset_index()
    summary.to_csv(SRC/"major7_fault_type_ema_summary.csv",index=False)
    delta_rows=[]
    for metric in ("macro_auroc","nll","ece15"):
        for fault_type in sorted(z.fault_type.unique()):
            zz=z[(z.metric==metric)&(z.fault_type==fault_type)]
            mean,lo,hi=two_level_interval(
                zz,("RO-PDRF-EMA99","RO-PDRF"),seed=15100+len(delta_rows))
            delta_rows.append({"fault_type":fault_type,"metric":metric,
                               "contrast":"RO-PDRF-EMA99 - RO-PDRF",
                               "mean_difference":mean,"ci_low":lo,"ci_high":hi})
    for metric in ("macro_auroc","nll","ece15"):
        zz=z[z.metric==metric]
        mean,lo,hi=three_level_interval(
            zz,("RO-PDRF-EMA99","RO-PDRF"),seed=15200+len(delta_rows))
        delta_rows.append({"fault_type":"all four types","metric":metric,
                           "contrast":"RO-PDRF-EMA99 - RO-PDRF",
                           "mean_difference":mean,"ci_low":lo,"ci_high":hi})
    delta=pd.DataFrame(delta_rows);delta.to_csv(
        SRC/"major7_fault_type_hierarchical_effects.csv",index=False)

    lines=[r"\begin{tabular}{lrrrrrr}",r"\toprule",
           r"Fault type & RO-CAGF AUROC & RO-PDRF-Full AUROC & EMA99 AUROC & EMA99 NLL & EMA99 ECE & $\Delta$ AUROC vs Full \\",r"\midrule"]
    for fault_type in sorted(z.fault_type.unique()):
        def val(method,metric):
            return summary[(summary.fault_type==fault_type)&
                           (summary.method==method)&
                           (summary.metric==metric)].iloc[0]["mean"]
        d=delta[(delta.fault_type==fault_type)&
                (delta.metric=="macro_auroc")].iloc[0]
        lines.append(f"{fault_type.replace('_','-').capitalize()} & "
                     f"{val('RO-CAGF','macro_auroc'):.3f} & "
                     f"{val('RO-PDRF','macro_auroc'):.3f} & "
                     f"{val('RO-PDRF-EMA99','macro_auroc'):.3f} & "
                     f"{val('RO-PDRF-EMA99','nll'):.3f} & "
                     f"{val('RO-PDRF-EMA99','ece15'):.3f} & "
                     f"{d.mean_difference:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major7_fault_type_ema.tex").write_text(
        "\n".join(lines),encoding="utf-8")

    principal=z[(z.fault_type=="gaussian")&
                (z.fault_realization==70001)&
                z.method.isin(["RO-PDRF-EMA95","RO-PDRF-EMA99",
                               "RO-PDRF-EMA995"])]
    sensitivity=principal.groupby(["method","metric"]).value.agg(
        ["mean","std"]).reset_index()
    sensitivity.to_csv(SRC/"major7_ema_decay_sensitivity.csv",index=False)
    lines=[r"\begin{tabular}{lrrr}",r"\toprule",
           r"EMA decay & Affected AUROC & NLL & ECE \\",r"\midrule"]
    for method,decay in (("RO-PDRF-EMA95","0.95"),("RO-PDRF-EMA99","0.99"),
                         ("RO-PDRF-EMA995","0.995")):
        def s(metric):
            x=sensitivity[(sensitivity.method==method)&
                          (sensitivity.metric==metric)].iloc[0]
            return f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}"
        lines.append(f"{decay} & {s('macro_auroc')} & {s('nll')} & {s('ece15')}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TAB/"major7_ema_decay.tex").write_text(
        "\n".join(lines),encoding="utf-8")
    return z,summary,delta


def figures(methods, conditional, fault_raw, fault_summary, fault_delta):
    # Main figure contract: no tested method dominates ranking and calibration;
    # agreement filtering reduces the set of active teacher targets and exposes
    # the recovery/error-transfer trade-off.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),
                         gridspec_kw={"width_ratios":[1.18,1.0]})
    ax=axs[0]
    colors={"RO-CAGF":ORANGE,"RO-PDRF":GREEN,"RO-AT-GATE":PURPLE,
            "RO-PDRF-EMA":BLUE,"QMF-PD":GREY,"RO-PDRF-CRA":"#1B9E77",
            "RO-PDRF-ECA":"#7570B3"}
    offsets={"RO-CAGF":(6,-10),"RO-PDRF":(7,-5),"RO-AT-GATE":(6,8),
             "RO-PDRF-EMA":(-88,8),"QMF-PD":(6,6),
             "RO-PDRF-CRA":(-88,8),"RO-PDRF-ECA":(7,-8)}
    for _,r in methods.iterrows():
        ax.scatter(r.macro_auroc,r.ece15,s=30,color=colors.get(r.method,BLUE),
                   edgecolor="white",linewidth=.5,zorder=3)
        distant=r.method in {"RO-PDRF-EMA","RO-PDRF-CRA"}
        display="RO-PDRF-Full" if r.method=="RO-PDRF" else r.method
        ax.annotate(display,(r.macro_auroc,r.ece15),
                    xytext=offsets.get(r.method,(4,4)),textcoords="offset points",
                    fontsize=6.1,
                    arrowprops=({"arrowstyle":"-","color":colors.get(r.method,BLUE),
                                 "lw":.55,"shrinkA":2,"shrinkB":2}
                                if distant else None))
    frontier=[]
    for _,r in methods.iterrows():
        dominated=((methods.macro_auroc>=r.macro_auroc)&
                   (methods.ece15<=r.ece15)&
                   ((methods.macro_auroc>r.macro_auroc)|
                    (methods.ece15<r.ece15))).any()
        if not dominated: frontier.append((r.macro_auroc,r.ece15))
    frontier=sorted(frontier)
    if len(frontier)>1:
        ax.plot(*zip(*frontier),color="#444",ls="--",lw=.8,zorder=1)
    ax.set_xlabel("Affected macro-AUROC (higher is better)")
    ax.set_ylabel("Affected ECE (lower is better)")
    ax.set_xlim(.747,.817);ax.set_ylim(.202,.388)
    ax.set_title("a  Ranking--calibration Pareto audit",loc="left",fontweight="bold")

    ax=axs[1]
    labels=["Active teacher\ncoverage","Conditional\nrecovery","Conditional\nerror transfer"]
    x=np.arange(3);width=.24
    n_rules=len(conditional)
    for j,(_,r) in enumerate(conditional.iterrows()):
        vals=[r.active_coverage,r.recovery_rate,r.error_transfer_rate]
        offset=(j-(n_rules-1)/2)*width
        ax.bar(x+offset,vals,width,color=[BLUE,GREEN,PURPLE][j],
               label=r.teacher_rule)
        for k,v in enumerate(vals):
            ax.text(x[k]+offset,v+.012+.018*j,f"{100*v:.1f}%",
                    ha="center",va="bottom",fontsize=5.9)
    ax.set_xticks(x,labels);ax.set_ylim(0,1.08);ax.set_ylabel("Fraction")
    ax.set_title("b  Multi-view teacher agreement",loc="left",fontweight="bold")
    ax.legend(loc="upper right",fontsize=5.6)
    save(fig,"figure5_pareto_teacher")

    # Supplementary figure: four prespecified fault mechanisms, eight
    # realizations and five optimization seeds. Error bars follow the
    # realization--seed hierarchy and do not imply hardware replication.
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.1));axs=axs.ravel()
    types=["gaussian","offset","drift","stuck_at"]
    labels=["Gaussian","Offset","Drift","Stuck-at"]
    ax=axs[0]
    for method,color,marker in (("RO-CAGF",ORANGE,"o"),("RO-PDRF",GREEN,"s"),
                                ("RO-PDRF-EMA99",BLUE,"^")):
        means=[];sds=[]
        for ft in types:
            r=fault_summary[(fault_summary.fault_type==ft)&
                            (fault_summary.method==method)&
                            (fault_summary.metric=="macro_auroc")].iloc[0]
            means.append(r["mean"]);sds.append(r["std"])
        display="RO-PDRF-Full" if method=="RO-PDRF" else method
        ax.errorbar(range(4),means,yerr=sds,marker=marker,color=color,
                    capsize=2,label=display)
    ax.set_xticks(range(4),labels,rotation=15,ha="right")
    ax.set_ylabel("Affected macro-AUROC")
    ax.set_title("a  Fault-type discrimination",loc="left",fontweight="bold")
    ax.legend()
    for panel,(metric,title,ylabel) in enumerate((
            ("macro_auroc","b  EMA effect on ranking","Delta AUROC"),
            ("nll","c  EMA effect on NLL","Delta NLL"),
            ("ece15","d  EMA effect on calibration","Delta ECE")),start=1):
        ax=axs[panel]
        z=fault_delta[(fault_delta.metric==metric)&
                      (fault_delta.fault_type!="all four types")].set_index("fault_type").reindex(types)
        mean=z.mean_difference.to_numpy();lo=z.ci_low.to_numpy();hi=z.ci_high.to_numpy()
        ax.errorbar(range(4),mean,yerr=np.vstack([mean-lo,hi-mean]),fmt="o",
                    color=BLUE,capsize=2)
        ax.axhline(0,color=GREY,ls="--",lw=.8)
        ax.set_xticks(range(4),labels,rotation=15,ha="right")
        ax.set_ylabel(ylabel);ax.set_title(title,loc="left",fontweight="bold")
    save(fig,"figureS8_fault_type_ema")


def main():
    methods=principal_method_summary()
    consistency_effect_table()
    conditional=conditional_rates()
    fault_raw,fault_summary,fault_delta=fault_type_analysis()
    figures(methods,conditional,fault_raw,fault_summary,fault_delta)
    overall=fault_delta[fault_delta.fault_type=="all four types"]
    print(json.dumps({
        "qmf_affected_auroc":float(methods.loc[methods.method=="QMF-PD","macro_auroc"].iloc[0]),
        "agreement":conditional.to_dict("records"),
        "ema_all_fault_types":overall.to_dict("records")},indent=2))


if __name__=="__main__":
    main()
