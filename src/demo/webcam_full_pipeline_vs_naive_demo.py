
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2

from inference_core import (
    CAMERA_SINGLE_5_CHECKPOINTS,
    CLASS_NAMES,
    DISPLAY_NAMES,
    FINAL_CONSISTENCY_5_CHECKPOINTS,
    PROJECT_ROOT,
    EnsembleGestureRecognizer,
    MediaPipeHandExtractor,
    draw_comparison_result,
)


METRICS_PATH = PROJECT_ROOT / "model/results/full_pipeline_vs_naive_demo_6cls/metrics.json"


def require_files(paths):
    missing=[str(path) for path in paths if not Path(path).is_file()]
    if missing: raise FileNotFoundError("Missing demo files:\n"+"\n".join(missing))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--no-mirror",action="store_true")
    parser.add_argument("--record",default=None)
    parser.add_argument("--smoothing-window",type=int,default=1)
    args=parser.parse_args()
    require_files(CAMERA_SINGLE_5_CHECKPOINTS+FINAL_CONSISTENCY_5_CHECKPOINTS+[METRICS_PATH])
    metrics=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    baseline_accuracy=100*metrics["naive_baseline"]["accuracy"]
    proposed_accuracy=100*metrics["full_proposed"]["accuracy"]
    baseline=EnsembleGestureRecognizer(CAMERA_SINGLE_5_CHECKPOINTS,args.smoothing_window)
    proposed=EnsembleGestureRecognizer(FINAL_CONSISTENCY_5_CHECKPOINTS,args.smoothing_window)
    extractor=MediaPipeHandExtractor(static_image_mode=False)
    capture=cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        extractor.close();raise RuntimeError(f"Cannot open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    ground_truth=None;bh=deque(maxlen=90);ph=deque(maxlen=90);rescues=reverses=0
    writer=None;last_time=perf_counter()
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            if not args.no_mirror:frame=cv2.flip(frame,1)
            extraction=extractor.process_bgr(frame);bp=pp=None
            if extraction is None:
                baseline.reset();proposed.reset()
            else:
                bp=baseline.predict_skeleton(extraction["camera_skeleton"],smooth=True)
                pp=proposed.predict_skeleton(extraction["skeleton"],smooth=True)
                if ground_truth is not None:
                    b_ok=bp["class"]==ground_truth;p_ok=pp["class"]==ground_truth
                    bh.append(b_ok);ph.append(p_ok);rescues+=int((not b_ok) and p_ok);reverses+=int(b_ok and (not p_ok))
            display=draw_comparison_result(
                frame,extraction,bp,pp,
                baseline_title="NAIVE: CAMERA SKELETON + SINGLE-VIEW",
                proposed_title="FULL: CANONICAL + BLENDER8 + CONSISTENCY",
                baseline_accuracy=baseline_accuracy,proposed_accuracy=proposed_accuracy,
            )
            now=perf_counter();fps=1/max(now-last_time,1e-8);last_time=now
            h,w=display.shape[:2];overlay=display.copy();cv2.rectangle(overlay,(8,h-160),(min(w-8,1080),h-8),(8,10,14),-1);display=cv2.addWeighted(overlay,.82,display,.18,0)
            cv2.putText(display,"SYSTEM-LEVEL COMPARISON (not a multiview-only ablation)",(18,h-134),cv2.FONT_HERSHEY_SIMPLEX,.52,(90,205,255),2,cv2.LINE_AA)
            cv2.putText(display,"Same RGB + MediaPipe | baseline keeps camera orientation | full pipeline canonicalizes",(18,h-106),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,235),1,cv2.LINE_AA)
            if ground_truth is None:
                line="Press 1-6 to set ground truth, then rotate the hand | R: reset | Q: quit";color=(90,205,255)
            elif bp is None:
                line=f"GT: {DISPLAY_NAMES[ground_truth]} | no hand detected";color=(90,205,255)
            else:
                b_ok=bp["class"]==ground_truth;p_ok=pp["class"]==ground_truth
                line=f"GT: {DISPLAY_NAMES[ground_truth]} | naive: {'CORRECT' if b_ok else 'WRONG'} | full: {'CORRECT' if p_ok else 'WRONG'}";color=(90,235,130) if p_ok and not b_ok else (235,235,235)
            cv2.putText(display,line,(18,h-78),cv2.FONT_HERSHEY_SIMPLEX,.47,color,2,cv2.LINE_AA)
            if bh:
                br=100*sum(bh)/len(bh);pr=100*sum(ph)/len(ph);rolling=f"Rolling {len(bh)}/90: naive {br:.1f}% | full {pr:.1f}% | rescue {rescues} | reverse {reverses}"
            else:rolling="Rolling robustness starts after ground truth selection"
            cv2.putText(display,rolling,(18,h-49),cv2.FONT_HERSHEY_SIMPLEX,.45,(90,235,130),2,cv2.LINE_AA)
            cv2.putText(display,f"Locked post-filter DrOh: {baseline_accuracy:.2f}% -> {proposed_accuracy:.2f}% ({metrics['difference_pp']:+.2f} pp) | FPS {fps:.1f}",(18,h-20),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,235,235),1,cv2.LINE_AA)
            if ground_truth is not None and bp is not None and bp["class"]!=ground_truth and pp["class"]==ground_truth:
                cv2.rectangle(display,(max(8,w//2-325),278),(min(w-8,w//2+325),330),(25,105,45),-1)
                cv2.putText(display,"FULL PIPELINE RESCUE",(max(18,w//2-215),314),cv2.FONT_HERSHEY_SIMPLEX,.9,(105,255,145),3,cv2.LINE_AA)
            if args.record and writer is None:writer=cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc(*"mp4v"),20.0,(w,h))
            if writer is not None:writer.write(display)
            cv2.imshow("Full Pipeline vs Naive Baseline - press Q to quit",display)
            key=cv2.waitKey(1)&0xFF
            if key in (ord("q"),27):break
            if ord("1")<=key<=ord("6"):
                ground_truth=CLASS_NAMES[key-ord("1")];baseline.reset();proposed.reset();bh.clear();ph.clear();rescues=reverses=0
            elif key==ord("r"):
                baseline.reset();proposed.reset();bh.clear();ph.clear();rescues=reverses=0
    finally:
        capture.release();extractor.close()
        if writer is not None:writer.release()
        cv2.destroyAllWindows()


if __name__=="__main__":main()
