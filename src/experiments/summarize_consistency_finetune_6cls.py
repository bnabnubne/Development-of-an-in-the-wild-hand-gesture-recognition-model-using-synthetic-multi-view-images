from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT=Path(__file__).resolve().parent
IN=ROOT/"results/consistency_finetune_6cls"; BASE=ROOT/"results/defense_experiments_6cls"; OUT=IN/"summary";OUT.mkdir(exist_ok=True)
WEIGHTS=[0.,.01,.03,.05,.1,.2,.3];SEEDS=[0,1,42];CLASSES=["ok","paper","rock","scissors","the-finger","thumb"]
def tag(w,s):return f"lambda{str(w).replace('.','p')}_lr1em4_seed{s}"
rows=[];ensembles={};cols=[f"prob_{c}" for c in CLASSES]
for w in WEIGHTS:
 fs=[]
 for s in SEEDS:
  d=IN/tag(w,s);q=json.loads((d/"summary.json").read_text());p=pd.read_csv(d/"predictions_droh_raw.csv")
  rows.append({"lambda":w,"seed":s,"val_all9":q["best_salux_val_all9_accuracy"],"val_anchor":q["selected_salux_val"]["anchor_accuracy"],"val_inconsistency":q["selected_salux_val"]["feature_inconsistency"],"droh_accuracy":q["evaluations"]["droh"]["accuracy"],"droh_macro_f1":q["evaluations"]["droh"]["macro_f1"],"correct":int(p.correct.sum())})
  fs.append(p)
 prob=np.mean([f[cols].to_numpy() for f in fs],0);truth=fs[0].true_class.map({c:i for i,c in enumerate(CLASSES)}).to_numpy();pred=prob.argmax(1);cor=truth==pred
 ensembles[w]=cor
runs=pd.DataFrame(rows);agg=runs.groupby("lambda").agg(val_all9_mean=("val_all9","mean"),val_all9_std=("val_all9","std"),val_inconsistency_mean=("val_inconsistency","mean"),droh_accuracy_mean=("droh_accuracy","mean"),droh_accuracy_std=("droh_accuracy","std"),droh_macro_f1_mean=("droh_macro_f1","mean"),droh_macro_f1_std=("droh_macro_f1","std")).reset_index()
ens=pd.DataFrame([{"lambda":w,"correct":int(c.sum()),"rows":len(c),"accuracy":c.mean()} for w,c in ensembles.items()])
selected=float(agg.loc[agg.val_all9_mean.idxmax(),"lambda"])
base=pd.read_csv(BASE/"ensembles/predictions_mv_ce_only_3seed_droh_raw.csv").correct.to_numpy(bool);right=ensembles[selected]
lo=int(np.sum(base&~right));ro=int(np.sum(~base&right));rng=np.random.default_rng(42);idx=rng.integers(0,len(base),(10000,len(base)));delta=(right[idx].mean(1)-base[idx].mean(1))*100
comparison={"selected_lambda":selected,"ce_accuracy":float(base.mean()),"consistency_accuracy":float(right.mean()),"difference_pp":float(100*(right.mean()-base.mean())),"ce_only_correct":lo,"consistency_only_correct":ro,"mcnemar_p":float(binomtest(min(lo,ro),lo+ro,.5).pvalue),"bootstrap_95ci_pp":[float(x) for x in np.quantile(delta,[.025,.975])]}
runs.to_csv(OUT/"all_runs.csv",index=False);agg.to_csv(OUT/"aggregate.csv",index=False);ens.to_csv(OUT/"ensembles.csv",index=False);(OUT/"selection_and_comparison.json").write_text(json.dumps(comparison,indent=2))
lines=["# Consistency Fine-tuning Results","","CE-only Blender8 checkpoint warm-up; cosine-consistency fine-tuning at lr=1e-4; selection by Salux validation all9 accuracy.","","| lambda | Val all9 | DrOh accuracy | DrOh macro-F1 | Ensemble |","|---:|---:|---:|---:|---:|"]
for _,x in agg.iterrows():
 e=ens[ens["lambda"]==x["lambda"]].iloc[0];lines.append(f"| {x['lambda']:g} | {100*x.val_all9_mean:.2f} +/- {100*x.val_all9_std:.2f} | {100*x.droh_accuracy_mean:.2f} +/- {100*x.droh_accuracy_std:.2f} | {100*x.droh_macro_f1_mean:.2f} +/- {100*x.droh_macro_f1_std:.2f} | {100*e.accuracy:.2f} ({int(e.correct)}/675) |")
lines += ["",f"Validation-selected lambda: **{selected:g}**.",f"Selected consistency ensemble versus CE-only ensemble: {comparison['consistency_accuracy']*100:.2f}% vs {comparison['ce_accuracy']*100:.2f}% ({comparison['difference_pp']:+.2f} pp), p={comparison['mcnemar_p']:.4g}, CI [{comparison['bootstrap_95ci_pp'][0]:.2f}, {comparison['bootstrap_95ci_pp'][1]:.2f}] pp."]
(OUT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
