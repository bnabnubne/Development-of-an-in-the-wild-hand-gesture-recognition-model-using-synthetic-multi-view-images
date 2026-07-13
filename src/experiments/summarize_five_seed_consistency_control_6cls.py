from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest,ttest_rel
from sklearn.metrics import f1_score,classification_report,confusion_matrix

R=Path(__file__).resolve().parent/"results";O=R/"defense_experiments_6cls";N=R/"consistency_finetune_6cls";OUT=N/"five_seed_control";OUT.mkdir(exist_ok=True)
S=[0,1,7,21,42];C=["ok","paper","rock","scissors","the-finger","thumb"];COL=["prob_"+c for c in C]
groups={"ce_checkpoint":[O/f"gru_wrist_middle_views8_lambda0p0_seed{s}/predictions_droh_raw.csv" for s in S],"low_lr_ce_control":[N/f"lambda0p0_lr1em4_seed{s}/predictions_droh_raw.csv" for s in S],"low_lr_consistency_0p3":[N/f"lambda0p3_lr1em4_seed{s}/predictions_droh_raw.csv" for s in S]}
rows=[];ens={};probs={}
for name,paths in groups.items():
 fs=[pd.read_csv(p) for p in paths]
 for s,f in zip(S,fs):rows.append({"group":name,"seed":s,"correct":int(f.correct.sum()),"rows":len(f),"accuracy":f.correct.mean()})
 p=np.mean([f[COL].to_numpy() for f in fs],0);truth=fs[0].true_class.to_numpy();pred=np.asarray(C)[p.argmax(1)];cor=pred==truth;ens[name]=cor;probs[name]=p
 out=fs[0][["action","sample_id","true_class"]].copy();out["predicted_class"]=pred;out["correct"]=cor;out["confidence"]=p.max(1)
 for i,c in enumerate(C):out["prob_"+c]=p[:,i]
 out.to_csv(OUT/f"predictions_{name}_droh.csv",index=False);np.save(OUT/f"confmat_{name}_droh.npy",confusion_matrix(truth,pred,labels=C))
 (OUT/f"classification_report_{name}_droh.txt").write_text(classification_report(truth,pred,labels=C,digits=6,zero_division=0))
pd.DataFrame(rows).to_csv(OUT/"all_runs.csv",index=False)
runs=pd.DataFrame(rows);agg=runs.groupby("group").accuracy.agg(["mean","std"]).reset_index();agg.to_csv(OUT/"aggregate.csv",index=False)
ensemble=pd.DataFrame([{"group":n,"correct":int(c.sum()),"rows":len(c),"accuracy":c.mean(),"macro_f1":f1_score(pd.read_csv(groups[n][0]).true_class,np.asarray(C)[probs[n].argmax(1)],average="macro")} for n,c in ens.items()]);ensemble.to_csv(OUT/"ensembles.csv",index=False)
left=ens["low_lr_ce_control"];right=ens["low_lr_consistency_0p3"];a=int((left&~right).sum());b=int((~left&right).sum());rng=np.random.default_rng(42);idx=rng.integers(0,len(left),(10000,len(left)));d=(right[idx].mean(1)-left[idx].mean(1))*100
control=runs[runs.group=="low_lr_ce_control"].set_index("seed").loc[S].accuracy.to_numpy();con=runs[runs.group=="low_lr_consistency_0p3"].set_index("seed").loc[S].accuracy.to_numpy()
stats={"five_seed_mean_difference_pp":100*(con.mean()-control.mean()),"paired_t_p":ttest_rel(con,control).pvalue,"ensemble_control_accuracy":left.mean(),"ensemble_consistency_accuracy":right.mean(),"ensemble_difference_pp":100*(right.mean()-left.mean()),"control_only_correct":a,"consistency_only_correct":b,"mcnemar_p":binomtest(min(a,b),a+b,.5).pvalue,"bootstrap_95ci_pp":[float(x) for x in np.quantile(d,[.025,.975])]}
ce=ens["ce_checkpoint"];a2=int((ce&~right).sum());b2=int((~ce&right).sum());idx2=rng.integers(0,len(ce),(10000,len(ce)));d2=(right[idx2].mean(1)-ce[idx2].mean(1))*100
stats["consistency_vs_original_ce_ensemble"]={"ce_accuracy":ce.mean(),"consistency_accuracy":right.mean(),"difference_pp":100*(right.mean()-ce.mean()),"ce_only_correct":a2,"consistency_only_correct":b2,"mcnemar_p":binomtest(min(a2,b2),a2+b2,.5).pvalue,"bootstrap_95ci_pp":[float(x) for x in np.quantile(d2,[.025,.975])]}
(OUT/"statistics.json").write_text(json.dumps(stats,indent=2))

# Internal Salux test check for the same five-member ensembles. This is reported
# separately from DrOh and is never used to tune a checkpoint.
salux_groups={
    "ce_checkpoint":[O/f"gru_wrist_middle_views8_lambda0p0_seed{s}/predictions_salux_raw.csv" for s in S],
    "low_lr_ce_control":[N/f"lambda0p0_lr1em4_seed{s}/predictions_salux_raw.csv" for s in S],
    "low_lr_consistency_0p3":[N/f"lambda0p3_lr1em4_seed{s}/predictions_salux_raw.csv" for s in S],
}
salux_rows=[]
for name,paths in salux_groups.items():
    fs=[pd.read_csv(p) for p in paths]
    key=fs[0][["action","sample_id","true_class"]]
    assert all(len(f)==983 and f[["action","sample_id","true_class"]].equals(key) for f in fs)
    p=np.mean([f[COL].to_numpy() for f in fs],axis=0)
    truth=fs[0].true_class.to_numpy();pred=np.asarray(C)[p.argmax(1)]
    salux_rows.append({"group":name,"correct":int((pred==truth).sum()),"rows":len(truth),
                       "accuracy":float((pred==truth).mean()),
                       "macro_f1":float(f1_score(truth,pred,average="macro"))})
pd.DataFrame(salux_rows).to_csv(OUT/"salux_ensembles.csv",index=False)

# Per-class DrOh comparison for the matched control and consistency ensembles.
truth=pd.read_csv(groups["low_lr_ce_control"][0]).true_class.to_numpy()
per_class=[]
for class_name in C:
    mask=truth==class_name
    row={"class":class_name,"samples":int(mask.sum())}
    for name in ["low_lr_ce_control","low_lr_consistency_0p3"]:
        row[name+"_correct"]=int(ens[name][mask].sum())
        row[name+"_accuracy"]=float(ens[name][mask].mean())
    row["consistency_gain_pp"]=100*(row["low_lr_consistency_0p3_accuracy"]-row["low_lr_ce_control_accuracy"])
    per_class.append(row)
pd.DataFrame(per_class).to_csv(OUT/"per_class_droh.csv",index=False)
print(agg.to_string(index=False));print(ensemble.to_string(index=False));print(json.dumps(stats,indent=2))
