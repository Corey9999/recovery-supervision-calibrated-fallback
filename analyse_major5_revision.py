"""Analyse manuscript-12 controls and generate submission tables/figures."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch

import run_major_revision_experiments as major

ROOT=Path(__file__).resolve().parent; SRC=ROOT/"source_data"; TAB=ROOT/"tables"; FIG=ROOT/"figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.titlesize":8,
 "axes.labelsize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
BLUE="#4C78A8"; ORANGE="#D98256"; GREEN="#2F8F6B"; GREY="#777777"; PURPLE="#7A68A6"


def save(fig,name):
    fig.tight_layout(pad=.8)
    for ext,kw in (("svg",{}),("pdf",{}),("png",{"dpi":300}),("tiff",{"dpi":600})):
        fig.savefig(FIG/f"{name}.{ext}",bbox_inches="tight",**kw)
    plt.close(fig)


def paired_p(a,b):
    d=np.asarray(a)-np.asarray(b)
    return float(wilcoxon(d,alternative="two-sided").pvalue) if np.any(d) else 1.0


def ci(x):
    return np.quantile(np.asarray(x),[.025,.975])


def hierarchical_bootstrap(pivot,n=5000,seed=991):
    """Resample top-level rows, then available seed columns within each row."""
    rng=np.random.default_rng(seed); a=pivot.to_numpy(float); out=np.empty(n)
    for k in range(n):
        rid=rng.integers(0,len(a),len(a)); vals=[]
        for r in rid:
            v=a[r][np.isfinite(a[r])]
            vals.extend(v[rng.integers(0,len(v),len(v))])
        out[k]=np.mean(vals)
    return out


def latex_table(df,columns,headers,align,name,formats=None):
    lines=[f"\\begin{{tabular}}{{{align}}}","\\toprule"," & ".join(headers)+r" \\","\\midrule"]
    formats=formats or {}
    for _,r in df.iterrows():
        vals=[]
        for c in columns:
            v=r[c]
            vals.append(formats[c](v) if c in formats else str(v))
        lines.append(" & ".join(vals)+r" \\")
    lines += ["\\bottomrule","\\end{tabular}"]
    (TAB/name).write_text("\n".join(lines),encoding="utf-8")


def consistency_ladder():
    old=pd.read_csv(SRC/"major_ablation_metrics.csv")
    old=old[(old.fault=="silent")&(old.subset=="affected")&(old.metric=="macro_auroc")]
    new=pd.read_csv(SRC/"major5_consistency_metrics.csv")
    corrected=SRC/"major5_basic_distillation_metrics.csv"
    if corrected.exists():
        basic=pd.read_csv(corrected)
        names=set(basic.method.unique())
        new=pd.concat([new[~new.method.isin(names)],basic],ignore_index=True)
    new=new[(new.fault_realization==70001)&(new.batch.astype(str)=="all")&
            (new.subset=="affected")&(new.metric=="macro_auroc")]
    head=pd.read_csv(SRC/"major3_objective_matched_metrics.csv")
    head=head[(head.dataset=="chemical")&(head.environment=="type:silent_gaussian")&
              (head.subset=="affected")&(head.metric=="macro_auroc")]
    base_map={"EF-PD":"EF_PD","UF-PD":"UF_PD","CAGF":"CAGF","PDRF":"PDRF"}
    rows=[]; seed_values={}
    for family,old_name in base_map.items():
        v=old[old.method==old_name].set_index("seed").value
        seed_values[(family,"Paired degradation")]=v
        rows.append((family,"Paired degradation",v.mean(),v.std(),np.nan))
        cd_name=family+"+CD"
        v2=new[new.method==cd_name].set_index("seed").value
        seed_values[(family,"Clean distillation")]=v2
        rows.append((family,"Clean distillation",v2.mean(),v2.std(),paired_p(v2.loc[v.index],v)))
        if family in ("CAGF","PDRF"):
            full_name="RO-"+family
            v3=head[head.method==full_name].set_index("seed").value
            seed_values[(family,"Full recovery")]=v3
            rows.append((family,"Full recovery",v3.mean(),v3.std(),paired_p(v3.loc[v2.index],v2)))
    for name in ("PDRF+CD-T07","ModDrop-SD"):
        v=new[new.method==name].set_index("seed").value
        rows.append((name,"Control",v.mean(),v.std(),np.nan))
    strong_path=SRC/"major6_strong_control_metrics.csv"
    ema_path=SRC/"major6_ema_metrics.csv"
    if strong_path.exists() and ema_path.exists():
        strong=pd.concat([pd.read_csv(strong_path),pd.read_csv(ema_path)],ignore_index=True)
        strong=strong[(strong.subset=="affected")&(strong.metric=="macro_auroc")]
        additions=(("RO-AT-GATE","Attention gate + full recovery",None),
                   ("RO-PDRF-EMA","Full recovery + EMA","RO-PDRF"),
                   ("RO-PDRF-CAL","Full recovery + 4x Brier","RO-PDRF"))
        for name,stage,reference in additions:
            v=strong[strong.method==name].set_index("seed").value
            if reference is None:
                p=np.nan
            else:
                ref=head[head.method==reference].set_index("seed").value
                p=paired_p(v.loc[ref.index],ref)
            rows.append((name,stage,v.mean(),v.std(),p))
    out=pd.DataFrame(rows,columns=["family","stage","mean","sd","p_vs_previous"])
    out.to_csv(SRC/"major5_consistency_ladder_summary.csv",index=False)
    latex_table(out,["family","stage","mean","sd","p_vs_previous"],
                ["Architecture","Training objective","Affected AUROC","SD","$P$ vs previous"],
                "llrrr","major5_consistency_ladder.tex",
                {"mean":lambda x:f"{x:.3f}","sd":lambda x:f"{x:.3f}",
                 "p_vs_previous":lambda x:"--" if pd.isna(x) else f"{x:.4f}"})
    return out,seed_values


def fault_realization_analysis():
    raw=pd.read_csv(SRC/"major5_consistency_metrics.csv")
    raw=raw[(raw.subset=="affected")&(raw.metric=="macro_auroc")&
            raw.method.isin(["CAGF","RO-CAGF","PDRF","RO-PDRF"])]
    d=raw[raw.batch.astype(str)=="all"]
    summary=(d.groupby(["method","fault_realization"]).value.mean().reset_index()
             .groupby("method").value.agg(["mean","std","min","max"]).reset_index())
    pv=d.pivot_table(index=["fault_realization","seed"],columns="method",values="value")
    contrasts=[]
    for label,a,b in (("RO-PDRF - RO-CAGF","RO-PDRF","RO-CAGF"),
                      ("RO-PDRF - PDRF","RO-PDRF","PDRF"),
                      ("RO-CAGF - CAGF","RO-CAGF","CAGF")):
        diff=(pv[a]-pv[b]).unstack("seed")
        boot=hierarchical_bootstrap(diff); lo,hi=ci(boot)
        realization_means=diff.mean(1)
        contrasts.append((label,diff.to_numpy().mean(),lo,hi,
                          int((realization_means>0).sum()),len(realization_means)))
    con=pd.DataFrame(contrasts,columns=["contrast","mean","ci_low","ci_high","positive_realizations","n_realizations"])
    con.to_csv(SRC/"major5_fault_realization_effects.csv",index=False)
    summary.to_csv(SRC/"major5_fault_realization_summary.csv",index=False)
    batch=raw[raw.batch.astype(str)!="all"].pivot_table(
        index=["fault_realization","batch","seed"],columns="method",values="value")
    batch["difference"]=batch["RO-PDRF"]-batch["RO-CAGF"]
    batch_summary=batch.groupby("batch").difference.agg(["mean","std","min","max"]).reset_index()
    batch_summary.to_csv(SRC/"major5_batch_stratified_effects.csv",index=False)
    latex_table(batch_summary,["batch","mean","std","min","max"],
                ["Acquisition batch","Mean $\\Delta$","SD","Minimum","Maximum"],
                "lrrrr","major5_batch_stratified_effects.tex",
                {c:(lambda x:f"{x:+.3f}") for c in ["mean","std","min","max"]})
    latex_table(con,["contrast","mean","ci_low","ci_high","positive_realizations","n_realizations"],
                ["Contrast","Mean $\\Delta$ AUROC","95\\% hierarchical CI","","Positive","Realizations"],
                "lrrrrr","major5_fault_realization_effects.tex",
                {"mean":lambda x:f"{x:+.3f}","ci_low":lambda x:f"[{x:+.3f},",
                 "ci_high":lambda x:f"{x:+.3f}]"})
    per=d.groupby(["fault_realization","method"]).value.mean().unstack()
    return con,per


def teacher_analysis():
    p=pd.read_csv(SRC/"major5_teacher_fault_predictions.csv")
    ro=p[p.method=="RO-PDRF"].copy()
    q=(ro.groupby(["clean_correct","fault_correct"]).size().rename("n").reset_index())
    q["fraction"]=q.n/q.n.sum();q.to_csv(SRC/"major5_teacher_quadrants.csv",index=False)
    qdisplay=q.copy();qdisplay["clean_state"]=qdisplay.clean_correct.map({True:"correct",False:"wrong"})
    qdisplay["fault_state"]=qdisplay.fault_correct.map({True:"correct",False:"wrong"})
    latex_table(qdisplay,["clean_state","fault_state","n","fraction"],
                ["Clean teacher","Faulted prediction","Rows","Fraction"],"llrr",
                "major5_teacher_quadrants.tex",{"fraction":lambda x:f"{x:.3f}"})
    a=ro.merge(p[p.method=="PDRF"],on=["seed","fault_realization","sample","batch","y"],suffixes=("_ro","_base"))
    a["ro_minus_base_true_probability"]=a.fault_true_probability_ro-a.fault_true_probability_base
    a["confidence_bin"]=pd.qcut(a.clean_confidence_ro,5,duplicates="drop")
    conf=(a.groupby("confidence_bin",observed=True)
          .agg(n=("sample","size"),clean_confidence=("clean_confidence_ro","mean"),
               ro_minus_base_true_probability=("ro_minus_base_true_probability","mean"),
               ro_fault_accuracy=("fault_correct_ro","mean"),base_fault_accuracy=("fault_correct_base","mean"))
          .reset_index())
    conf.to_csv(SRC/"major5_teacher_confidence.csv",index=False)
    confdisplay=conf.copy();confdisplay["confidence_bin"]=confdisplay.confidence_bin.astype(str)
    latex_table(confdisplay,["confidence_bin","n","clean_confidence","ro_minus_base_true_probability","ro_fault_accuracy","base_fault_accuracy"],
                ["Teacher-confidence quintile","Rows","Mean confidence","RO-base $\\Delta p_y$","RO accuracy","Base accuracy"],"lrrrrr",
                "major5_teacher_confidence.tex",
                {c:(lambda x:f"{x:.3f}") for c in ["clean_confidence","ro_minus_base_true_probability","ro_fault_accuracy","base_fault_accuracy"]})
    eligible=(~a.clean_correct_ro)&a.fault_correct_base
    destroyed=eligible&(~a.fault_correct_ro)
    recovery_eligible=a.clean_correct_ro&(~a.fault_correct_base)
    recovered=recovery_eligible&a.fault_correct_ro
    transfer={
        "correct_clean_teacher_rows":int(a.clean_correct_ro.sum()),
        "baseline_wrong_with_correct_ro_teacher":int(recovery_eligible.sum()),
        "baseline_error_corrected_under_ro":int(recovered.sum()),
        "conditional_recovery_rate":float(recovered.sum()/recovery_eligible.sum()) if recovery_eligible.sum() else np.nan,
        "wrong_clean_teacher_rows":int((~a.clean_correct_ro).sum()),
        "wrong_teacher_but_ro_recovers":int(((~a.clean_correct_ro)&a.fault_correct_ro).sum()),
        "baseline_correct_with_wrong_ro_teacher":int(eligible.sum()),
        "baseline_correction_lost_under_ro":int(destroyed.sum()),
        "conditional_error_transfer_rate":float(destroyed.sum()/eligible.sum()) if eligible.sum() else np.nan,
    }
    (SRC/"major5_teacher_error_transfer.json").write_text(json.dumps(transfer,indent=2),encoding="utf-8")
    conditional=pd.DataFrame([
        {"teacher_state":"Correct","baseline_fault_state":"Wrong",
         "eligible":int(recovery_eligible.sum()),"changed":int(recovered.sum()),
         "conditional_rate":transfer["conditional_recovery_rate"],
         "interpretation":"Recovered by RO-PDRF"},
        {"teacher_state":"Wrong","baseline_fault_state":"Correct",
         "eligible":int(eligible.sum()),"changed":int(destroyed.sum()),
         "conditional_rate":transfer["conditional_error_transfer_rate"],
         "interpretation":"Correction lost under RO-PDRF"},
    ])
    conditional.to_csv(SRC/"major5_teacher_conditional_rates.csv",index=False)
    latex_table(conditional,["teacher_state","baseline_fault_state","eligible","changed","conditional_rate","interpretation"],
                ["Clean teacher","Baseline fault prediction","Eligible","Changed","Conditional rate","Outcome"],
                "llrrrl","major5_teacher_conditional_rates.tex",
                {"conditional_rate":lambda x:f"{x:.3f}"})
    return q,conf,transfer


def hydraulic_analysis():
    blocked=pd.read_csv(SRC/"major5_hydraulic_blocked_metrics.csv")
    blocked=blocked[(blocked.subset=="affected")&(blocked.metric=="macro_auroc")]
    pv=blocked.pivot_table(index=["task","fault","seed"],columns="method",values="value")
    pv["difference"]=pv["RO-PDRF"]-pv["RO-CAGF"]
    condition=pv.groupby(["task","fault"]).difference.mean().reset_index()
    mat=pv.difference.unstack("seed")
    boot=hierarchical_bootstrap(mat,seed=992);lo,hi=ci(boot)
    loo=[]
    for task in sorted(condition.task.unique()):
        loo.append({"left_out_task":task,"mean_difference":condition[condition.task!=task].difference.mean()})
    pd.DataFrame(loo).to_csv(SRC/"major5_hydraulic_leave_one_task_out.csv",index=False)
    summary={"mean":float(condition.difference.mean()),"median":float(condition.difference.median()),
             "minimum":float(condition.difference.min()),"maximum":float(condition.difference.max()),
             "ci_low":float(lo),"ci_high":float(hi),
             "positive_conditions":int((condition.difference>0).sum()),"n_conditions":len(condition)}
    (SRC/"major5_hydraulic_blocked_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    sim=pd.read_csv(SRC/"major5_hydraulic_similarity.csv")
    simsum=sim.groupby("task").agg(n=("test_index","size"),same_class_nearest=("same_class","mean"),
                                    median_distance=("euclidean_distance","median"),
                                    median_cosine=("cosine_similarity","median")).reset_index()
    simsum.to_csv(SRC/"major5_hydraulic_similarity_summary.csv",index=False)
    latex_table(simsum,["task","n","same_class_nearest","median_distance","median_cosine"],
                ["Task","Test cycles","Same-class nearest","Median distance","Median cosine"],"lrrrr",
                "major5_hydraulic_similarity.tex",
                {c:(lambda x:f"{x:.3f}") for c in ["same_class_nearest","median_distance","median_cosine"]})
    latex_table(condition,["task","fault","difference"],["Task","Fault","Blocked-split $\\Delta$ AUROC"],
                "llr","major5_hydraulic_blocked.tex",{"difference":lambda x:f"{x:+.3f}"})
    return condition,summary


def multiclass_ece(y,p,bins=15):
    conf=p.max(1); pred=p.argmax(1); correct=pred==y; edges=np.linspace(0,1,bins+1); out=0
    for lo,hi in zip(edges[:-1],edges[1:]):
        take=(conf>lo)&(conf<=hi)
        if take.any(): out+=take.mean()*abs(correct[take].mean()-conf[take].mean())
    return float(out)


def classwise_ece(y,p,bins=10):
    out=[]
    for cls in range(p.shape[1]):
        yy=y==cls; pp=p[:,cls]; e=0
        for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
            take=(pp>lo)&(pp<=hi)
            if take.any(): e+=take.mean()*abs(yy[take].mean()-pp[take].mean())
        out.append(e)
    return np.asarray(out)


def calibration_analysis():
    d=pd.read_csv(SRC/"major3_objective_matched_predictions.csv")
    d=d[d.method.isin(["RO-CAGF","RO-PDRF"])]
    pcols=[c for c in d if c.startswith("p")]
    rows=[]; ensemble={}
    for method in ("RO-CAGF","RO-PDRF"):
        z=d[d.method==method]
        idx=["sample","batch","y","affected"]
        prob=z.groupby(idx)[pcols].mean().reset_index()
        prob_values=prob[pcols].to_numpy()
        prob[pcols]=prob_values/prob_values.sum(1,keepdims=True)
        logp=z.copy();logp[pcols]=np.log(logp[pcols].clip(1e-12))
        geom=logp.groupby(idx)[pcols].mean().reset_index()
        g=np.exp(geom[pcols].to_numpy());g/=g.sum(1,keepdims=True);geom[pcols]=g
        for aggregation,dd in (("probability mean",prob),("logit mean",geom)):
            for subset,take in (("all",np.ones(len(dd),bool)),("affected",dd.affected.to_numpy(bool))):
                y=dd.y.to_numpy()[take];p=dd[pcols].to_numpy()[take]
                yb=label_binarize(y,classes=np.arange(len(pcols)))
                rows.append({"method":method,"aggregation":aggregation,"subset":subset,
                             "macro_auroc":roc_auc_score(yb,p,average="macro",multi_class="ovr"),
                             "nll":log_loss(y,p,labels=np.arange(len(pcols))),
                             "ece":multiclass_ece(y,p),"classwise_ece":classwise_ece(y,p).mean()})
            if aggregation=="probability mean": ensemble[method]=prob
    out=pd.DataFrame(rows);out.to_csv(SRC/"major5_ensemble_calibration.csv",index=False)
    # Classwise reliability source data and selective-risk source data.
    rel=[];risk=[]
    for method,dd in ensemble.items():
        take=dd.affected.to_numpy(bool);y=dd.y.to_numpy()[take];p=dd[pcols].to_numpy()[take]
        for cls in range(len(pcols)):
            pp=p[:,cls]; yy=y==cls
            for b,(lo,hi) in enumerate(zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:])):
                keep=(pp>lo)&(pp<=hi)
                if keep.any(): rel.append({"method":method,"class":cls+1,"bin":b,"n":int(keep.sum()),
                                           "confidence":pp[keep].mean(),"frequency":yy[keep].mean()})
        conf=p.max(1);correct=p.argmax(1)==y
        order=np.argsort(-conf)
        for coverage in np.linspace(.2,1,9):
            n=max(1,int(np.floor(coverage*len(y))));keep=order[:n]
            risk.append({"method":method,"coverage":coverage,"risk":1-correct[keep].mean(),"n":n})
    rel=pd.DataFrame(rel);risk=pd.DataFrame(risk)
    rel.to_csv(SRC/"major5_classwise_reliability.csv",index=False)
    risk.to_csv(SRC/"major5_selective_risk.csv",index=False)
    latex_table(out,["method","aggregation","subset","macro_auroc","nll","ece","classwise_ece"],
                ["Method","Aggregation","Subset","AUROC","NLL","ECE","Classwise ECE"],
                "lllrrrr","major5_ensemble_calibration.tex",
                {c:(lambda x:f"{x:.3f}") for c in ["macro_auroc","nll","ece","classwise_ece"]})
    return out,rel,risk


def complexity_analysis():
    rows=[]
    for m in (2,4,8,16):
        for method,spec in (("CAGF",major.SPECS["CAGF"]),("PDRF",major.SPECS["PDRF"])):
            model=major.Fusion(m,32,6,spec).eval(); macs=[0];acts=[0]
            hooks=[]
            def hook(module,inputs,output):
                macs[0]+=int(output.numel()*module.in_features)
                acts[0]+=int(output.numel()*4)
            for module in model.modules():
                if isinstance(module,torch.nn.Linear):hooks.append(module.register_forward_hook(hook))
            with torch.no_grad():model(torch.zeros(1,m,32),torch.ones(1,m),torch.ones(1,m))
            for h in hooks:h.remove()
            params=sum(p.numel() for p in model.parameters())
            rows.append({"method":method,"groups":m,"parameters":params,
                         "inference_macs":macs[0],"inference_flops":2*macs[0],
                         "state_kib":params*4/1024,"forward_activation_kib":acts[0]/1024})
    out=pd.DataFrame(rows);out.to_csv(SRC/"major5_complexity_scaling.csv",index=False)
    latex_table(out,["method","groups","parameters","inference_flops","state_kib","forward_activation_kib"],
                ["Method","Groups","Parameters","FLOPs","State (KiB)","Activations (KiB)"],
                "lrrrrr","major5_complexity_scaling.tex",
                {"state_kib":lambda x:f"{x:.1f}","forward_activation_kib":lambda x:f"{x:.1f}"})
    stability=pd.read_csv(SRC/"major5_training_stability.csv")
    stable=(stability.groupby("method").agg(epochs=("epoch","max"),
             median_gradient_norm=("gradient_norm","median"),
             maximum_gradient_norm=("gradient_norm","max"),
             final_selection_ce=("selection_ce","last")).reset_index())
    stable.to_csv(SRC/"major5_training_stability_summary.csv",index=False)
    latex_table(stable,["method","epochs","median_gradient_norm","maximum_gradient_norm","final_selection_ce"],
                ["Method","Max epochs","Median gradient norm","Maximum gradient norm","Final selection CE"],"lrrrr",
                "major5_training_stability.tex",
                {c:(lambda x:f"{x:.3f}") for c in ["median_gradient_norm","maximum_gradient_norm","final_selection_ce"]})
    return out


def figures(ladder,seed_values,per_fault,quadrants,confidence,hyd_condition,cal,rel,risk,complexity):
    # Figure contract: clean-to-fault distillation explains most of the gain;
    # repeated corruptions preserve a positive direction; Full adds little
    # ranking over Lite at substantially higher measured CPU cost; and a wrong
    # clean teacher exposes a distinct error-transfer boundary.
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.8))
    ax=axs[0,0]
    x=np.arange(3)
    for family,color,marker in (("CAGF",ORANGE,"o"),("PDRF",GREEN,"s")):
        vals=[seed_values[(family,s)].mean() for s in ("Paired degradation","Clean distillation","Full recovery")]
        sd=[seed_values[(family,s)].std() for s in ("Paired degradation","Clean distillation","Full recovery")]
        ax.errorbar(x,vals,yerr=sd,color=color,marker=marker,capsize=2,label=family)
    ax.set_xticks(x,["Paired\ndegradation","+ clean\ndistillation","Full\nrecovery"]);ax.set_ylabel("Affected macro-AUROC")
    ax.set_title("a  Distillation ladder",loc="left",fontweight="bold");ax.legend()
    ax=axs[0,1]
    diff=per_fault["RO-PDRF"]-per_fault["RO-CAGF"]
    ax.axhline(0,color=GREY,ls="--",lw=.8);ax.scatter(range(1,len(diff)+1),diff,color=BLUE)
    ax.plot(range(1,len(diff)+1),diff,color=BLUE,lw=.8);ax.set_xlabel("Independent fixed fault realization")
    ax.set_ylabel("RO-PDRF-Full minus RO-CAGF AUROC");ax.set_title("b  Fault-realization robustness",loc="left",fontweight="bold")
    ax=axs[1,0]
    principal=pd.read_csv(SRC/"major8_principal_method_summary.csv").set_index("method")
    trade=principal.loc[["RO-PDRF-Lite","RO-PDRF-Full"]]
    colors=[BLUE,GREEN]
    for (name,row),color in zip(trade.iterrows(),colors):
        ax.errorbar(row.macro_auroc,row.train_seconds,
                    xerr=row.macro_auroc_sd,yerr=row.train_seconds_sd,
                    fmt="o",ms=6,capsize=2.5,color=color,zorder=3)
        dx=-34 if name.endswith("Lite") else 7
        dy=7 if name.endswith("Lite") else -12
        ax.annotate(name.replace("RO-PDRF-", ""),
                    (row.macro_auroc,row.train_seconds),xytext=(dx,dy),
                    textcoords="offset points",fontsize=7,color=color,
                    fontweight="bold")
    lite,full=trade.loc["RO-PDRF-Lite"],trade.loc["RO-PDRF-Full"]
    ax.plot([lite.macro_auroc,full.macro_auroc],
            [lite.train_seconds,full.train_seconds],color=GREY,lw=.9,ls="--",zorder=1)
    ax.text(.04,.96,
            f"$\\Delta$AUROC {full.macro_auroc-lite.macro_auroc:+.4f}\n"
            f"CPU time {full.train_seconds/lite.train_seconds:.2f}$\\times$",
            transform=ax.transAxes,ha="left",va="top",fontsize=6.8,
            bbox={"facecolor":"white","edgecolor":"#B8B8B8","pad":2.0})
    ax.set_xlim(.790,.814);ax.set_ylim(0,32)
    ax.set_xlabel("Affected macro-AUROC");ax.set_ylabel("Mean CPU training time (s)")
    ax.set_title("c  Lite--Full cost--ranking trade-off",loc="left",fontweight="bold")
    ax=axs[1,1]
    order=[(True,True),(True,False),(False,True),(False,False)]
    labels=["Teacher correct\nFault correct","Teacher correct\nFault wrong","Teacher wrong\nFault correct","Teacher wrong\nFault wrong"]
    qmap={(bool(r.clean_correct),bool(r.fault_correct)):r.fraction for _,r in quadrants.iterrows()}
    vals=[qmap.get(k,0) for k in order];ax.bar(range(4),vals,color=[GREEN,ORANGE,BLUE,GREY])
    ax.set_xticks(range(4),labels,rotation=18,ha="right");ax.set_ylabel("Fraction of affected predictions")
    ax.set_ylim(0,.58)
    conditional=pd.read_csv(SRC/"major5_teacher_conditional_rates.csv")
    recovery=float(conditional.loc[
        conditional.interpretation=="Recovered by RO-PDRF","conditional_rate"].iloc[0])
    transfer=float(conditional.loc[
        conditional.interpretation=="Correction lost under RO-PDRF","conditional_rate"].iloc[0])
    ax.text(.98,.95,f"Conditional recovery {100*recovery:.1f}%\nError transfer {100*transfer:.1f}%",
            transform=ax.transAxes,ha="right",va="top",fontsize=6.4,
            bbox={"facecolor":"white","edgecolor":"#B8B8B8","pad":2.2})
    ax.set_title("d  Clean-teacher error quadrants",loc="left",fontweight="bold")
    save(fig,"figure4_revision_controls")

    fig,axs=plt.subplots(2,3,figsize=(7.2,4.5),sharex=True,sharey=True);axs=axs.ravel()
    for cls,ax in enumerate(axs,1):
        ax.plot([0,1],[0,1],color=GREY,ls="--",lw=.7)
        for method,color in (("RO-CAGF",ORANGE),("RO-PDRF",GREEN)):
            z=rel[(rel["class"]==cls)&(rel.method==method)]
            label="RO-PDRF-Full" if method=="RO-PDRF" else method
            ax.plot(z.confidence,z.frequency,marker="o",ms=2.5,color=color,label=label)
        ax.set_title(f"Class {cls}");ax.set_xlim(0,1);ax.set_ylim(0,1)
    axs[3].set_xlabel("Predicted probability");axs[4].set_xlabel("Predicted probability")
    axs[0].set_ylabel("Observed frequency");axs[3].set_ylabel("Observed frequency");axs[0].legend()
    save(fig,"figureS5_classwise_reliability")

    fig,axs=plt.subplots(1,3,figsize=(7.2,2.55))
    ax=axs[0]
    for method,color in (("RO-CAGF",ORANGE),("RO-PDRF",GREEN)):
        z=risk[risk.method==method]
        label="RO-PDRF-Full" if method=="RO-PDRF" else method
        ax.plot(z.coverage,z.risk,marker="o",color=color,label=label)
    ax.set_xlabel("Coverage");ax.set_ylabel("Selective classification risk");ax.set_title("a  Affected-row abstention",loc="left",fontweight="bold");ax.legend()
    ax=axs[1]
    for method,color in (("CAGF",ORANGE),("PDRF",GREEN)):
        z=complexity[complexity.method==method];ax.plot(z.groups,z.inference_flops/1e3,marker="o",color=color,label=method)
    ax.set_xlabel("Sensor groups");ax.set_ylabel("Inference FLOPs (thousands)");ax.set_title("b  Computational scaling",loc="left",fontweight="bold")
    ax=axs[2]
    stability=pd.read_csv(SRC/"major5_training_stability.csv")
    for method,color in (("PDRF+CD",GREEN),("CAGF+CD",ORANGE)):
        z=stability[stability.method==method].groupby("epoch").gradient_norm.mean()
        ax.plot(z.index,z.values,color=color,label=method)
    ax.set_xlabel("Epoch");ax.set_ylabel("Mean gradient norm");ax.set_title("c  Training stability",loc="left",fontweight="bold");ax.legend()
    save(fig,"figureS6_deployment_and_complexity")


def main():
    ladder,seed_values=consistency_ladder()
    _,per_fault=fault_realization_analysis()
    quadrants,confidence,transfer=teacher_analysis()
    hyd_condition,hyd_summary=hydraulic_analysis()
    cal,rel,risk=calibration_analysis()
    complexity=complexity_analysis()
    figures(ladder,seed_values,per_fault,quadrants,confidence,hyd_condition,cal,rel,risk,complexity)
    print(json.dumps({"teacher_error_transfer":transfer,"hydraulic":hyd_summary},indent=2))


if __name__=="__main__":main()
