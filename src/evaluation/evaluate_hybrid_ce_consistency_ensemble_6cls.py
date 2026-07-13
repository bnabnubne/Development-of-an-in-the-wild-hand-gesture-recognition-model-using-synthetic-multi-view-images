from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

ROOT=Path(__file__).resolve().parent/"results";OLD=ROOT/"defense_experiments_6cls";NEW=ROOT/"consistency_finetune_6cls";OUT=NEW/"hybrid_ensemble";OUT.mkdir(exist_ok=True)
C=["ok","paper","rock","scissors","the-finger","thumb"];COL=["prob_"+c for c in C];summary={}
for ds in ["salux_raw","droh_raw"]:
 fs=[pd.read_csv(OLD/f"gru_wrist_middle_views8_lambda0p0_seed{s}/predictions_{ds}.csv") for s in [0,1,42]]
 fs += [pd.read_csv(NEW/f"lambda0p3_lr1em4_seed{s}/predictions_{ds}.csv") for s in [0,1,42]]
 for f in fs[1:]:
  assert f.sample_id.equals(fs[0].sample_id) and f.true_class.equals(fs[0].true_class)
 prob=np.mean([f[COL].to_numpy() for f in fs],0);pred=np.asarray(C)[prob.argmax(1)];truth=fs[0].true_class.to_numpy();correct=pred==truth
 out=fs[0][["action","sample_id","true_class"]].copy();out["predicted_class"]=pred;out["correct"]=correct;out["confidence"]=prob.max(1)
 for i,c in enumerate(C):out["prob_"+c]=prob[:,i]
 out.to_csv(OUT/f"predictions_{ds}.csv",index=False);np.save(OUT/f"confmat_{ds}.npy",confusion_matrix(truth,pred,labels=C))
 summary[ds]={"accuracy":accuracy_score(truth,pred),"balanced_accuracy":balanced_accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,average="macro"),"correct":int(correct.sum()),"rows":len(correct)}
 if ds=="droh_raw":
  ce=pd.read_csv(OLD/"ensembles/predictions_mv_ce_only_3seed_droh_raw.csv").correct.to_numpy(bool);a=int(np.sum(ce&~correct));b=int(np.sum(~ce&correct))
  summary["paired_vs_ce_ensemble"]={"ce_only_correct_hybrid_wrong":a,"ce_only_wrong_hybrid_correct":b,"difference_pp":100*(correct.mean()-ce.mean()),"mcnemar_p":binomtest(min(a,b),a+b,.5).pvalue}
summary["protocol"]={"members":"three CE-only seeds + three lambda=0.3 consistency-finetuned seeds","fusion":"equal arithmetic mean of six probability vectors","single_view_inference":True,"no_DrOh_weight_tuning":True}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
