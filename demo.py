"""
demo.py — Intelligent Biometric System — Unified Entry Point
=============================================================
Works three ways:

  1. Positional command (simplest):
       py demo.py train
       py demo.py eval
       py demo.py train_eval
       py demo.py verify --fp1 ... --vein1 ... --fp1b ... --vein1b ... --fp2 ... --vein2 ...
       py demo.py verify_quick --fp1 ... --vein1 ... --fp2 ... --vein2 ...

  2. Flag style (also works):
       py demo.py --mode train

  3. No arguments → shows interactive menu

Verify example (Windows ^ line continuation):
  py demo.py verify ^
      --fp1    "Session1\Fingerprint\001\001_1.bmp" ^
      --vein1  "Session1\FingerVein\001\001_1.bmp"  ^
      --fp1b   "Session2\Fingerprint\001\001_1.bmp" ^
      --vein1b "Session2\FingerVein\001\001_1.bmp"  ^
      --fp2    "Session1\Fingerprint\002\002_1.bmp" ^
      --vein2  "Session1\FingerVein\002\002_1.bmp"
"""

import os
import sys
import csv
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED PATHS — edit these
# ══════════════════════════════════════════════════════════════════════════════
DATASET_ROOT  = r"D:\Intelligent_Biometric_System\NUPT-FPV-main\image"
MODEL_PATH    = r"D:\Intelligent_Biometric_System\best_biometric_model.pth"
MODEL_PATH_V1 = r"D:\Intelligent_Biometric_System\best_biometric_model.pth"
OUTPUT_DIR    = r"D:\Intelligent_Biometric_System\test_results"
LOSS_CSV_PATH = r"D:\Intelligent_Biometric_System\loss_curves.csv"
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_BLUE   = "#378ADD"
C_RED    = "#E24B4A"
C_GREEN  = "#1D9E75"
C_ORANGE = "#F5A623"
C_GREY   = "#888780"
C_PURPLE = "#7F77DD"

VALID_MODES = ["train", "eval", "train_eval", "verify", "verify_quick"]


# ──────────────────────────────────────────────────────────────────────────────
#  Flexible argument parser
#  Accepts:   py demo.py train
#             py demo.py --mode train
#             py demo.py              (interactive menu)
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    import argparse

    # If first real arg is a bare mode word, inject --mode before it
    args = sys.argv[1:]
    if args and args[0] in VALID_MODES:
        args = ["--mode"] + args

    parser = argparse.ArgumentParser(
        prog="demo.py",
        description="Biometric System — unified entry point",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  py demo.py train
  py demo.py eval
  py demo.py train_eval
  py demo.py verify ^
      --fp1    "Session1\\Fingerprint\\001\\001_1.bmp" ^
      --vein1  "Session1\\FingerVein\\001\\001_1.bmp"  ^
      --fp1b   "Session2\\Fingerprint\\001\\001_1.bmp" ^
      --vein1b "Session2\\FingerVein\\001\\001_1.bmp"  ^
      --fp2    "Session1\\Fingerprint\\002\\002_1.bmp" ^
      --vein2  "Session1\\FingerVein\\002\\002_1.bmp"
  py demo.py verify_quick --fp1 a.bmp --vein1 b.bmp --fp2 c.bmp --vein2 d.bmp
""")

    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help="train | eval | train_eval | verify | verify_quick"
    )
    parser.add_argument("--model",  default=None,
                        help="Override model .pth path")
    parser.add_argument("--fp1",    default=None)
    parser.add_argument("--vein1",  default=None)
    parser.add_argument("--fp1b",   default=None)
    parser.add_argument("--vein1b", default=None)
    parser.add_argument("--fp2",    default=None)
    parser.add_argument("--vein2",  default=None)

    return parser.parse_args(args)


# ──────────────────────────────────────────────────────────────────────────────
#  Interactive menu (shown when no args given)
# ──────────────────────────────────────────────────────────────────────────────
def interactive_menu():
    sep = "═" * 52
    print(f"\n{sep}")
    print("  Intelligent Biometric System")
    print(f"{sep}")
    print("  [1] train         — train the model")
    print("  [2] eval          — full evaluation (15 plots)")
    print("  [3] train_eval    — train then evaluate")
    print("  [4] verify        — live demo  (3 image pairs)")
    print("  [5] verify_quick  — quick 2-image comparison")
    print("  [0] exit")
    print(f"{sep}")

    choice = input("  Select [0-5]: ").strip()
    mapping = {
        "1": "train", "2": "eval", "3": "train_eval",
        "4": "verify", "5": "verify_quick", "0": None
    }
    if choice not in mapping or mapping[choice] is None:
        print("  Exiting.")
        sys.exit(0)

    mode = mapping[choice]

    # Collect image paths for verify modes interactively
    class _Args:
        model  = None
        fp1    = None; vein1  = None
        fp1b   = None; vein1b = None
        fp2    = None; vein2  = None

    a = _Args()

    if mode == "verify":
        print(f"\n  Enter image paths (press Enter after each):")
        a.fp1    = input("  Person 1 base  — fingerprint path : ").strip().strip('"')
        a.vein1  = input("  Person 1 base  — vein path        : ").strip().strip('"')
        a.fp1b   = input("  Person 1 (2nd) — fingerprint path : ").strip().strip('"')
        a.vein1b = input("  Person 1 (2nd) — vein path        : ").strip().strip('"')
        a.fp2    = input("  Person 2       — fingerprint path : ").strip().strip('"')
        a.vein2  = input("  Person 2       — vein path        : ").strip().strip('"')

    elif mode == "verify_quick":
        print(f"\n  Enter image paths:")
        a.fp1   = input("  Person 1 — fingerprint path : ").strip().strip('"')
        a.vein1 = input("  Person 1 — vein path        : ").strip().strip('"')
        a.fp2   = input("  Person 2 — fingerprint path : ").strip().strip('"')
        a.vein2 = input("  Person 2 — vein path        : ").strip().strip('"')

    return mode, a


# ──────────────────────────────────────────────────────────────────────────────
#  Module imports
# ──────────────────────────────────────────────────────────────────────────────
def _import_modules():
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    from dataset import BiometricDataset
    from model   import FusionModel
    return BiometricDataset, FusionModel


# ──────────────────────────────────────────────────────────────────────────────
#  Image loading
# ──────────────────────────────────────────────────────────────────────────────
_val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_pair(fp_path, vein_path):
    fp = cv2.imread(fp_path)
    if fp is None:
        raise IOError(f"Cannot read fingerprint: {fp_path}")
    fp = cv2.cvtColor(fp, cv2.COLOR_BGR2RGB)
    vn = cv2.imread(vein_path)
    if vn is None:
        raise IOError(f"Cannot read vein: {vein_path}")
    vn = cv2.cvtColor(_clahe.apply(cv2.cvtColor(vn, cv2.COLOR_BGR2GRAY)),
                      cv2.COLOR_GRAY2RGB)
    return _val_tf(fp).unsqueeze(0), _val_tf(vn).unsqueeze(0)


def embed(model, fp_path, vein_path, device):
    fp_t, vn_t = load_pair(fp_path, vein_path)
    model.eval()
    with torch.no_grad():
        e = model(fp_t.to(device), vn_t.to(device))
    return e.cpu().numpy().squeeze()


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ──────────────────────────────────────────────────────────────────────────────
#  Model loader
# ──────────────────────────────────────────────────────────────────────────────
def load_model(FusionModel, path=None):
    if path is None:
        path = MODEL_PATH if os.path.isfile(MODEL_PATH) else MODEL_PATH_V1
    if not os.path.isfile(path):
        print(f"\n[ERROR] No model found.")
        print(f"  Looked for : {MODEL_PATH}")
        print(f"  And also   : {MODEL_PATH_V1}")
        print(f"\n  Run first  : py demo.py train")
        sys.exit(1)
    model = FusionModel(embed_dim=256, num_heads=8,
                        num_layers=2, output_dim=512).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Model : {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Confidence bar (ASCII)
# ──────────────────────────────────────────────────────────────────────────────
def confidence_bar(score, threshold, width=44):
    pos = int(np.clip((score + 1) / 2 * width, 0, width - 1))
    thr = int(np.clip((threshold + 1) / 2 * width, 0, width - 1))
    bar = ["-"] * width
    bar[thr] = "|"
    bar[pos] = "█"
    return "".join(bar)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: train
# ══════════════════════════════════════════════════════════════════════════════
def mode_train():
    print("\n" + "═"*60)
    print("  MODE: TRAIN  (verification-focused v2)")
    print("═"*60 + "\n")
    try:
        import train
        train.train()
    except ImportError:
        print("[ERROR] train.py not found.")
        print("  Ensure train.py, losses.py, sampler.py are present.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: eval
# ══════════════════════════════════════════════════════════════════════════════
def mode_eval(model_override=None):
    print("\n" + "═"*60)
    print("  MODE: EVAL  (15 plots + CSV metrics)")
    print("═"*60 + "\n")
    try:
        import test as test_module
        test_module.MODEL_PATH    = model_override or (
            MODEL_PATH if os.path.isfile(MODEL_PATH) else MODEL_PATH_V1)
        test_module.DATASET_ROOT  = DATASET_ROOT
        test_module.OUTPUT_DIR    = OUTPUT_DIR
        test_module.LOSS_CSV_PATH = LOSS_CSV_PATH
        test_module.main()
    except ImportError:
        print("[ERROR] test.py not found.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Calibrate threshold from dataset
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_threshold(model, dataset, device):
    dataset.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    all_emb, all_lbl = [], []
    model.eval()
    with torch.no_grad():
        for fp, vn, labels in loader:
            e = model(fp.to(device), vn.to(device))
            all_emb.append(e.cpu().numpy())
            all_lbl.append(labels.numpy())

    emb = np.vstack(all_emb)
    lbl = np.concatenate(all_lbl)
    n   = emb.shape[0]
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    e   = emb / np.clip(nrm, 1e-8, None)
    sim = e @ e.T

    genuine, impostor = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (genuine if lbl[i] == lbl[j] else impostor).append(sim[i, j])

    genuine  = np.array(genuine)
    impostor = np.array(impostor)

    best_eer, best_t = 1.0, 0.5
    for t in np.linspace(-1, 1, 4000):
        far = np.mean(impostor >= t)
        frr = np.mean(genuine  <  t)
        if (far + frr) / 2 < best_eer:
            best_eer, best_t = (far + frr) / 2, t

    tar_results = {}
    for far_target in [0.001, 0.01, 0.1]:
        for t in np.sort(impostor)[::-1]:
            if np.mean(impostor >= t) <= far_target:
                tar_results[far_target] = float(np.mean(genuine >= t))
                break
        else:
            tar_results[far_target] = 0.0

    return {
        "threshold":       best_t,
        "eer":             best_eer,
        "genuine_mean":    float(genuine.mean()),
        "genuine_std":     float(genuine.std()),
        "impostor_mean":   float(impostor.mean()),
        "impostor_std":    float(impostor.std()),
        "genuine_scores":  genuine,
        "impostor_scores": impostor,
        "far_at_thr":      float(np.mean(impostor >= best_t)),
        "frr_at_thr":      float(np.mean(genuine  <  best_t)),
        "tar_results":     tar_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Verify plot
# ══════════════════════════════════════════════════════════════════════════════
def save_verify_plot(scores, cal, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(cal["impostor_scores"], bins=60, density=True, alpha=0.28,
            color=C_RED,  label="Dataset impostors")
    ax.hist(cal["genuine_scores"],  bins=60, density=True, alpha=0.28,
            color=C_BLUE, label="Dataset genuines")
    try:
        from scipy.stats import gaussian_kde
        for data, color in [(cal["impostor_scores"], C_RED),
                            (cal["genuine_scores"],  C_BLUE)]:
            xs = np.linspace(data.min() - 0.05, data.max() + 0.05, 300)
            ax.plot(xs, gaussian_kde(data)(xs), color=color, linewidth=2)
    except ImportError:
        pass
    ax.axvline(cal["threshold"],      color=C_ORANGE, lw=2,   ls="--",
               label=f"EER threshold = {cal['threshold']:.4f}")
    ax.axvline(scores["p1_vs_p1b"],   color=C_GREEN,  lw=2.5, ls="-",
               label=f"Genuine pair = {scores['p1_vs_p1b']:.4f}")
    ax.axvline(scores["p1_vs_p2"],    color=C_PURPLE, lw=2.5, ls="-",
               label=f"P1 vs P2 = {scores['p1_vs_p2']:.4f}")
    if scores.get("p1b_vs_p2") is not None:
        ax.axvline(scores["p1b_vs_p2"], color=C_GREY, lw=1.5, ls=":",
                   label=f"P1b vs P2 = {scores['p1b_vs_p2']:.4f}")
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Your Pairs vs Dataset Score Distribution", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2  = axes[1]
    thr  = cal["threshold"]
    bar_labels = ["P1 base vs\nP1 same-person",
                  "P1 base vs\nP2 (impostor)",
                  "P1 (2nd) vs\nP2 (impostor)"]
    vals = [scores["p1_vs_p1b"], scores["p1_vs_p2"],
            scores.get("p1b_vs_p2", 0)]
    colors_bar = [
        C_GREEN if vals[0] >= thr else C_RED,   # genuine: green=match
        C_GREEN if vals[1] <  thr else C_RED,   # impostor: green=correctly rejected
        C_GREEN if vals[2] <  thr else C_RED,
    ]
    bars = ax2.barh(bar_labels, vals, color=colors_bar,
                    height=0.35, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax2.text(min(v + 0.01, 1.08),
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
    ax2.axvline(thr, color=C_ORANGE, lw=2, ls="--",
                label=f"Threshold = {thr:.4f}")
    ax2.set_xlim(-0.05, 1.15)
    ax2.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax2.set_title("Verification Verdicts\n(green = correct outcome)",
                  fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="x", alpha=0.3)

    v_genuine  = scores["p1_vs_p1b"] >= thr
    v_impostor = scores["p1_vs_p2"]  <  thr
    result_txt = ("✅ SYSTEM CORRECT" if (v_genuine and v_impostor)
                  else "⚠️  CHECK RESULTS")
    ax2.text(0.5, -0.16, result_txt, transform=ax2.transAxes,
             ha="center", fontsize=11, fontweight="bold",
             color=C_GREEN if (v_genuine and v_impostor) else C_RED)

    fig.suptitle("Live Verification Demo — Biometric System",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: verify  (full — with dataset calibration)
# ══════════════════════════════════════════════════════════════════════════════
def mode_verify(args, model_override=None):
    BiometricDataset, FusionModel = _import_modules()

    needed = [("--fp1",    args.fp1),   ("--vein1",  args.vein1),
              ("--fp1b",   args.fp1b),  ("--vein1b", args.vein1b),
              ("--fp2",    args.fp2),   ("--vein2",  args.vein2)]
    missing = [n for n, v in needed if not v]
    if missing:
        print(f"\n[ERROR] verify mode needs: {', '.join(missing)}\n")
        print("  py demo.py verify ^")
        print('      --fp1    "Session1\\Fingerprint\\001\\001_1.bmp" ^')
        print('      --vein1  "Session1\\FingerVein\\001\\001_1.bmp"  ^')
        print('      --fp1b   "Session2\\Fingerprint\\001\\001_1.bmp" ^')
        print('      --vein1b "Session2\\FingerVein\\001\\001_1.bmp"  ^')
        print('      --fp2    "Session1\\Fingerprint\\002\\002_1.bmp" ^')
        print('      --vein2  "Session1\\FingerVein\\002\\002_1.bmp"')
        sys.exit(1)

    for n, p in needed:
        if not os.path.isfile(p):
            print(f"[ERROR] File not found for {n}: {p}")
            sys.exit(1)

    print("\n" + "═"*64)
    print("  MODE: VERIFY — Live Biometric Demo")
    print("═"*64)

    model = load_model(FusionModel, model_override)

    print(f"\n  Embedding on {DEVICE} …")
    print(f"  [P1 base]  {os.path.basename(args.fp1)} + {os.path.basename(args.vein1)}")
    emb_p1  = embed(model, args.fp1,  args.vein1,  DEVICE)
    print(f"  [P1 2nd]   {os.path.basename(args.fp1b)} + {os.path.basename(args.vein1b)}")
    emb_p1b = embed(model, args.fp1b, args.vein1b, DEVICE)
    print(f"  [Person 2] {os.path.basename(args.fp2)} + {os.path.basename(args.vein2)}")
    emb_p2  = embed(model, args.fp2,  args.vein2,  DEVICE)

    scores = {
        "p1_vs_p1b": cosine_sim(emb_p1, emb_p1b),
        "p1_vs_p2":  cosine_sim(emb_p1, emb_p2),
        "p1b_vs_p2": cosine_sim(emb_p1b, emb_p2),
    }

    print("\n  Calibrating threshold from full dataset …")
    dataset = BiometricDataset(root_dir=DATASET_ROOT)
    cal     = calibrate_threshold(model, dataset, DEVICE)

    _print_verify_results(scores, cal)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path  = os.path.join(OUTPUT_DIR, "verify_demo_result.csv")
    plot_path = os.path.join(OUTPUT_DIR, "verify_demo_plot.png")
    _save_verify_csv(scores, cal, csv_path)
    save_verify_plot(scores, cal, plot_path)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: verify_quick  (no dataset, fixed threshold)
# ══════════════════════════════════════════════════════════════════════════════
def mode_verify_quick(args, model_override=None):
    BiometricDataset, FusionModel = _import_modules()

    needed = [("--fp1",  args.fp1), ("--vein1", args.vein1),
              ("--fp2",  args.fp2), ("--vein2", args.vein2)]
    missing = [n for n, v in needed if not v]
    if missing:
        print(f"\n[ERROR] verify_quick needs: {', '.join(missing)}")
        print("  py demo.py verify_quick ^")
        print('      --fp1 "fp1.bmp" --vein1 "vein1.bmp" ^')
        print('      --fp2 "fp2.bmp" --vein2 "vein2.bmp"')
        sys.exit(1)

    for n, p in needed:
        if not os.path.isfile(p):
            print(f"[ERROR] File not found for {n}: {p}")
            sys.exit(1)

    print("\n" + "═"*60)
    print("  MODE: VERIFY QUICK  (fixed threshold = 0.50)")
    print("═"*60)
    print("  Tip: use 'verify' mode for calibrated FAR/FRR stats.\n")

    model = load_model(FusionModel, model_override)

    print(f"  Person 1 : {os.path.basename(args.fp1)} + {os.path.basename(args.vein1)}")
    emb1 = embed(model, args.fp1, args.vein1, DEVICE)
    print(f"  Person 2 : {os.path.basename(args.fp2)} + {os.path.basename(args.vein2)}")
    emb2 = embed(model, args.fp2, args.vein2, DEVICE)

    score = cosine_sim(emb1, emb2)
    thr   = 0.50

    print(f"\n{'─'*52}")
    print(f"  Cosine similarity : {score:.6f}")
    print(f"  Score bar         : [{confidence_bar(score, thr)}]")
    print(f"                       (| = threshold {thr:.2f})")
    match = score >= thr
    if match:
        conf = "Very high" if score > 0.85 else "High" if score > 0.70 else "Moderate"
        print(f"  Verdict           : ✅  MATCH  ({conf} confidence)")
    else:
        conf = "Very high" if score < 0.20 else "High" if score < 0.35 else "Moderate"
        print(f"  Verdict           : ❌  NO MATCH  ({conf} confidence)")
    print(f"{'─'*52}")


# ──────────────────────────────────────────────────────────────────────────────
#  Print verify results
# ──────────────────────────────────────────────────────────────────────────────
def _print_verify_results(scores, cal):
    thr = cal["threshold"]
    sep = "─" * 64

    print(f"\n{sep}")
    print("  DATASET CALIBRATION")
    print(f"{sep}")
    print(f"  Genuine  : mean={cal['genuine_mean']:.4f}  std={cal['genuine_std']:.4f}")
    print(f"  Impostor : mean={cal['impostor_mean']:.4f}  std={cal['impostor_std']:.4f}")
    print(f"  EER threshold : {thr:.4f}   (EER = {cal['eer']*100:.3f}%)")
    for ft, tar in cal["tar_results"].items():
        print(f"  TAR @ FAR={ft*100:.1f}%  : {tar*100:.2f}%")

    print(f"\n{sep}")
    print("  RESULTS")
    print(f"{sep}")

    # ① Genuine pair
    s1 = scores["p1_vs_p1b"]
    v1 = s1 >= thr
    print(f"\n  ① P1-base  vs  P1-other  (same person, different instance)")
    print(f"     Similarity : {s1:.6f}")
    print(f"     Bar        : [{confidence_bar(s1, thr)}]  (| = {thr:.4f})")
    if v1:
        pct = float(np.mean(cal["genuine_scores"] <= s1) * 100)
        print(f"     Verdict    : ✅  MATCH — same person confirmed")
        print(f"     Percentile : top {100-pct:.1f}% of genuine pairs in dataset")
    else:
        pct = float(np.mean(cal["genuine_scores"] <= s1) * 100)
        print(f"     Verdict    : ❌  NO MATCH  ← False Rejection (FRR event)")
        print(f"     Percentile : {pct:.1f}th of genuine pairs (score too low)")
        print(f"     Fix        : train more epochs / raise ArcFace margin")

    # ② Impostor pair
    s2 = scores["p1_vs_p2"]
    v2 = s2 >= thr
    print(f"\n  ② P1-base  vs  Person 2  (should be rejected)")
    print(f"     Similarity : {s2:.6f}")
    print(f"     Bar        : [{confidence_bar(s2, thr)}]")
    if not v2:
        pct = float(np.mean(cal["impostor_scores"] >= s2) * 100)
        print(f"     Verdict    : ✅  CORRECTLY REJECTED")
        print(f"     Percentile : lower than {pct:.1f}% of dataset impostors")
    else:
        print(f"     Verdict    : ⚠️   FALSE ACCEPT  ← FAR event")

    # ③ Bonus
    s3 = scores.get("p1b_vs_p2")
    if s3 is not None:
        v3 = s3 >= thr
        print(f"\n  ③ P1-other vs  Person 2  (bonus check)")
        print(f"     Similarity : {s3:.6f}")
        print(f"     Bar        : [{confidence_bar(s3, thr)}]")
        print(f"     Verdict    : {'✅  CORRECTLY REJECTED' if not v3 else '⚠️   FALSE ACCEPT'}")

    # FAR / FRR
    far = cal["far_at_thr"]; frr = cal["frr_at_thr"]
    print(f"\n{sep}")
    print("  FAR / FRR  (at EER threshold, from full dataset)")
    print(f"{sep}")
    print(f"  FAR : {far*100:.4f}%  → 1 in {int(1/far) if far>0 else 'inf'} impostors accepted")
    print(f"  FRR : {frr*100:.4f}%  → 1 in {int(1/frr) if frr>0 else 'inf'} genuine users rejected")
    print(f"  EER : {cal['eer']*100:.4f}%")

    all_ok = v1 and not v2 and (s3 is None or not v3)
    print(f"\n{'═'*64}")
    if all_ok:
        print("  🎉  DEMO PASSED — correct accept + correct reject")
    elif v1 and v2:
        print("  ⚠️   FALSE ACCEPT — impostor passed threshold")
    elif not v1:
        print("  ⚠️   FALSE REJECT — genuine pair below threshold")
    else:
        print("  ❌  MULTIPLE FAILURES")
    print(f"{'═'*64}")


def _save_verify_csv(scores, cal, path):
    thr = cal["threshold"]
    with open(path, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["Comparison", "Score", "Threshold", "Verdict", "Type"])
        for key, label, ptype in [
            ("p1_vs_p1b", "P1-base vs P1-other", "Genuine pair"),
            ("p1_vs_p2",  "P1-base vs P2",        "Impostor pair"),
            ("p1b_vs_p2", "P1-other vs P2",        "Impostor pair"),
        ]:
            v = scores.get(key)
            if v is not None:
                if ptype == "Genuine pair":
                    verdict = "MATCH" if v >= thr else "NO MATCH (False Reject)"
                else:
                    verdict = "CORRECTLY REJECTED" if v < thr else "FALSE ACCEPT"
                w.writerow([label, f"{v:.6f}", f"{thr:.6f}", verdict, ptype])
        w.writerow([])
        w.writerow(["Metric", "Value"])
        for k, v in [
            ("EER (%)",              f"{cal['eer']*100:.4f}"),
            ("EER Threshold",        f"{cal['threshold']:.6f}"),
            ("FAR at threshold (%)", f"{cal['far_at_thr']*100:.4f}"),
            ("FRR at threshold (%)", f"{cal['frr_at_thr']*100:.4f}"),
            ("Genuine mean",         f"{cal['genuine_mean']:.4f}"),
            ("Impostor mean",        f"{cal['impostor_mean']:.4f}"),
        ]:
            w.writerow([k, v])
        for ft, tar in cal["tar_results"].items():
            w.writerow([f"TAR @ FAR={ft*100:.1f}% (%)", f"{tar*100:.4f}"])
    print(f"  CSV   → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # No mode given → interactive menu
    if args.mode is None:
        mode, args = interactive_menu()
    else:
        mode = args.mode

    print(f"\n  Device : {DEVICE}")

    if   mode == "train":        mode_train()
    elif mode == "eval":         mode_eval(args.model)
    elif mode == "train_eval":   mode_train(); mode_eval(args.model)
    elif mode == "verify":       mode_verify(args, args.model)
    elif mode == "verify_quick": mode_verify_quick(args, args.model)


if __name__ == "__main__":
    main()