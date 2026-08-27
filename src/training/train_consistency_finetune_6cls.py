
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import train_defense_ablation_6cls as base


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "results/defense_experiments_6cls"
OUT_ROOT = ROOT / "results/consistency_finetune_6cls"
WEIGHTS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 7, 21, 42]
LR = 1e-4
MAX_EPOCHS = 50
PATIENCE = 12


def tag(weight, seed):
    return f"lambda{str(weight).replace('.', 'p')}_lr1em4_seed{seed}"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def all_view_metrics(model, loader):
    model.eval(); anchor_correct = 0; view_correct = np.zeros(8, np.int64); total = 0; con_sum = 0.0
    with torch.no_grad():
        for anchor, views, y in loader:
            anchor, views, y = anchor.to(base.DEVICE), views.to(base.DEVICE), y.to(base.DEVICE)
            anchor_logits, anchor_feature = model(anchor)
            b, v, j, c = views.shape
            view_logits, view_feature = model(views.reshape(b*v, j, c))
            view_logits = view_logits.reshape(b, v, -1); view_feature = view_feature.reshape(b, v, -1)
            anchor_correct += int((anchor_logits.argmax(1) == y).sum())
            view_correct += (view_logits.argmax(2) == y[:, None]).sum(0).cpu().numpy()
            con_sum += float((1-F.cosine_similarity(anchor_feature[:,None,:],view_feature,dim=2)).sum())
            total += b
    anchor_acc = anchor_correct / total; view_acc = view_correct / total
    return {
        "anchor_accuracy": anchor_acc, "mean_view_accuracy": float(view_acc.mean()),
        "all9_accuracy": float((anchor_acc + view_acc.sum()) / 9),
        "worst_view_accuracy": float(view_acc.min()),
        "feature_inconsistency": con_sum / (total * 8),
        **{f"view_{i}_accuracy": float(value) for i, value in enumerate(view_acc)},
    }


def single_metrics(model, dataset, out_dir, name):
    loader = DataLoader(dataset, batch_size=base.BATCH_SIZE, shuffle=False)
    truth, prediction, probability = base.evaluate(model, loader)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    frame = dataset.frame[["action", "sample_id"]].copy().reset_index(drop=True)
    frame["true_class"] = [base.CLASS_NAMES[i] for i in truth]
    frame["predicted_class"] = [base.CLASS_NAMES[i] for i in prediction]
    frame["correct"] = truth == prediction
    frame["confidence"] = probability.max(1)
    for i, class_name in enumerate(base.CLASS_NAMES): frame[f"prob_{class_name}"] = probability[:,i]
    frame.to_csv(out_dir/f"predictions_{name}.csv",index=False)
    np.save(out_dir/f"confmat_{name}.npy",confusion_matrix(truth,prediction,labels=range(6)))
    return {
        "accuracy":accuracy_score(truth,prediction),
        "balanced_accuracy":balanced_accuracy_score(truth,prediction),
        "macro_f1":f1_score(truth,prediction,average="macro"),"rows":len(truth),
    }


def run(weight, seed, salux, mv, droh):
    out = OUT_ROOT / tag(weight, seed); summary_path = out/"summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "complete":
        print(f"[skip] {tag(weight,seed)}",flush=True); return
    out.mkdir(parents=True,exist_ok=True); seed_all(seed)
    parts = {split:salux[salux.split==split].copy() for split in ["train","val","test"]}
    mv_parts = {split:mv[mv.split==split].copy() for split in ["train","val"]}
    train_ds = base.MultiViewDataset(mv_parts["train"],list(range(8)),"wrist_middle")
    val_ds = base.MultiViewDataset(mv_parts["val"],list(range(8)),"wrist_middle")
    val_loader = DataLoader(val_ds,batch_size=128,shuffle=False)
    salux_test = base.SingleDataset(parts["test"],"input_path","wrist_middle")
    droh_test = base.SingleDataset(droh,"input_path","wrist_middle")
    shuffle = torch.Generator().manual_seed(seed+20000)
    train_loader = DataLoader(train_ds,batch_size=base.BATCH_SIZE,shuffle=True,generator=shuffle)
    source = SOURCE_ROOT/f"gru_wrist_middle_views8_lambda0p0_seed{seed}/best.pt"
    checkpoint = torch.load(source,map_location=base.DEVICE,weights_only=False)
    model = base.make_model("gru").to(base.DEVICE); model.load_state_dict(checkpoint["model_state_dict"])
    initial_val = all_view_metrics(model,val_loader)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR); criterion=nn.CrossEntropyLoss()
    best=-1.0; best_epoch=-1; stale=0; history=[]; start=time.perf_counter(); best_path=out/"best.pt"
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); total_loss=total_cls=total_con=0.0
        for anchor,views,y in train_loader:
            anchor,views,y=anchor.to(base.DEVICE),views.to(base.DEVICE),y.to(base.DEVICE)
            optimizer.zero_grad(); alogit,afeat=model(anchor); b,v,j,c=views.shape
            vlogit,vfeat=model(views.reshape(b*v,j,c)); vlogit=vlogit.reshape(b,v,-1); vfeat=vfeat.reshape(b,v,-1)
            classification=(criterion(alogit,y)+sum(criterion(vlogit[:,i],y) for i in range(v)))/(v+1)
            consistency=(1-F.cosine_similarity(afeat[:,None,:],vfeat,dim=2)).mean()
            loss=classification+weight*consistency; loss.backward(); optimizer.step()
            total_loss+=loss.item();total_cls+=classification.item();total_con+=consistency.item()
        val=all_view_metrics(model,val_loader)
        row={"epoch":epoch,"loss":total_loss/len(train_loader),"classification_loss":total_cls/len(train_loader),
             "consistency_loss":total_con/len(train_loader),**{f"val_{k}":v for k,v in val.items()}}
        history.append(row)
        if val["all9_accuracy"] > best + 1e-6:
            best=val["all9_accuracy"];best_epoch=epoch;stale=0
            torch.save({"model_state_dict":model.state_dict(),"lambda":weight,"seed":seed,"lr":LR,
                        "source_checkpoint":str(source),"selection":"Salux validation all9 accuracy","val_metrics":val},best_path)
        else: stale+=1
        print(f"lambda={weight:g} seed={seed} epoch={epoch:02d} cls={row['classification_loss']:.5f} con={row['consistency_loss']:.5f} all9={val['all9_accuracy']:.5f} best={best:.5f}",flush=True)
        if stale>=PATIENCE: break
    selected=torch.load(best_path,map_location=base.DEVICE,weights_only=False);model.load_state_dict(selected["model_state_dict"])
    evaluations={"salux":single_metrics(model,salux_test,out,"salux_raw"),"droh":single_metrics(model,droh_test,out,"droh_raw")}
    summary={"status":"complete","protocol":"CE checkpoint warm-up then cosine-consistency fine-tuning",
             "lambda":weight,"seed":seed,"learning_rate":LR,"source_checkpoint":str(source),
             "initial_salux_val":initial_val,"best_epoch":best_epoch,"best_salux_val_all9_accuracy":best,
             "selected_salux_val":selected["val_metrics"],"evaluations":evaluations,
             "droh_policy":"external evaluation only","elapsed_seconds":time.perf_counter()-start}
    summary_path.write_text(json.dumps(summary,indent=2));(out/"history.json").write_text(json.dumps(history,indent=2))
    print(f"[done] {tag(weight,seed)} DrOh={evaluations['droh']['accuracy']:.6f}",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,choices=SEEDS,required=True);parser.add_argument("--only-lambda",type=float,choices=WEIGHTS,default=None);args=parser.parse_args()
    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    salux=pd.read_csv(base.SALUX_CSV);mv=pd.read_csv(base.MV_CSV);droh=pd.read_csv(base.DROH_CSV)
    weights=WEIGHTS if args.only_lambda is None else [args.only_lambda]
    for weight in weights: run(weight,args.seed,salux,mv,droh)


if __name__=="__main__":main()
