"""
test.py — Intelligent Biometric System — Full Evaluation Suite
==============================================================
Loads best_biometric_model.pth and produces a comprehensive report:

  Metrics
  -------
  1.  Rank-1 / Rank-5 / Rank-10 Identification Accuracy
  2.  EER  (Equal Error Rate)
  3.  TAR @ FAR = 0.1% / 1% / 10%
  4.  AUC-ROC

  Plots  (all saved to OUTPUT_DIR)
  ----------------------------------
  01. cmc_curve.png              – Cumulative Match Characteristic
  02. roc_curve.png              – ROC curve with EER point
  03. det_curve.png              – Detection Error Trade-off (log scale)
  04. score_distribution.png     – Genuine vs Impostor KDE + histogram
  05. similarity_heatmap.png     – Full N×N pairwise cosine matrix
  06. tsne_embeddings.png        – 2-D t-SNE scatter, colour = class
  07. per_class_rank1.png        – Per-subject Rank-1 bar chart
  08. per_class_intra_sim.png    – Per-subject intra-class mean similarity
  09. confusion_matrix.png       – Confusion / top confused class pairs
  10. embedding_stats.png        – L2 norm, mean, std of each embedding
  11. modal_weights.png          – Learned fingerprint vs vein weight
  12. precision_recall.png       – Precision-Recall curve
  13. far_frr_curve.png          – FAR / FRR vs threshold
  14. intra_inter_box.png        – Box plot of intra vs inter-class scores
  15. loss_curves.png            – Replayed from loss_curves.csv (if present)

  CSVs
  ----
  evaluation_summary.csv         – All scalar metrics
  per_class_summary.csv          – Per-subject stats
  cmc_data.csv                   – Raw CMC values

Usage
-----
  python test.py                                        # full evaluation
  python test.py --fp a.bmp --vein b.bmp               # score a single pair
  python test.py --fp a.bmp --vein b.bmp \
                 --fp2 c.bmp --vein2 d.bmp             # compare two identities
  python test.py --fp a.bmp --vein b.bmp --eval        # pair + full eval
"""

import os
import sys
import csv
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED PATHS  ← edit these to match your machine
# ══════════════════════════════════════════════════════════════════════════════
DATASET_ROOT  = r"D:\Intelligent_Biometric_System\NUPT-FPV-main\image"
MODEL_PATH    = r"D:\Intelligent_Biometric_System\best_biometric_model.pth"
OUTPUT_DIR    = r"D:\Intelligent_Biometric_System\test_results"
LOSS_CSV_PATH = r"D:\Intelligent_Biometric_System\loss_curves.csv"   # optional
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Consistent colour palette
C_BLUE   = "#378ADD"
C_RED    = "#E24B4A"
C_PURPLE = "#7F77DD"
C_GREEN  = "#1D9E75"
C_ORANGE = "#F5A623"
C_GREY   = "#888780"


# ──────────────────────────────────────────────────────────────────────────────
#  Project-module import
# ──────────────────────────────────────────────────────────────────────────────
def _import_project_modules():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from dataset import BiometricDataset
    from model   import FusionModel
    return BiometricDataset, FusionModel


# ──────────────────────────────────────────────────────────────────────────────
#  Image loading (single-pair inference)
# ──────────────────────────────────────────────────────────────────────────────
_val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_image_pair(fp_path, vein_path):
    fp_img = cv2.imread(fp_path)
    if fp_img is None:
        raise IOError(f"Cannot read: {fp_path}")
    fp_img = cv2.cvtColor(fp_img, cv2.COLOR_BGR2RGB)

    vn_img  = cv2.imread(vein_path)
    if vn_img is None:
        raise IOError(f"Cannot read: {vein_path}")
    vn_gray = cv2.cvtColor(vn_img, cv2.COLOR_BGR2GRAY)
    vn_img  = cv2.cvtColor(_clahe.apply(vn_gray), cv2.COLOR_GRAY2RGB)

    return (_val_tf(fp_img).unsqueeze(0),
            _val_tf(vn_img).unsqueeze(0))


# ──────────────────────────────────────────────────────────────────────────────
#  Embedding extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_embeddings(model, dataset, device):
    dataset.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    all_emb, all_lbl = [], []
    model.eval()
    with torch.no_grad():
        for fp, vn, labels in loader:
            emb = model(fp.to(device), vn.to(device))
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(labels.numpy())
    return np.vstack(all_emb), np.concatenate(all_lbl)


# ──────────────────────────────────────────────────────────────────────────────
#  Core metric helpers
# ──────────────────────────────────────────────────────────────────────────────
def cosine_matrix(emb):
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    e = emb / np.clip(n, 1e-8, None)
    return e @ e.T


def genuine_impostor(sim, labels):
    N = sim.shape[0]
    g, im = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (g if labels[i] == labels[j] else im).append(sim[i, j])
    return np.array(g), np.array(im)


def rank1_acc(sim, labels):
    s = sim.copy(); np.fill_diagonal(s, -np.inf)
    return np.mean(labels[np.argmax(s, axis=1)] == labels)


def cmc_curve(sim, labels, max_rank=20):
    N = sim.shape[0]; max_rank = min(max_rank, N - 1)
    hits = np.zeros(max_rank)
    for i in range(N):
        s = sim[i].copy(); s[i] = -np.inf
        ranked = np.argsort(-s)
        pos = np.where(labels[ranked] == labels[i])[0]
        if len(pos) and pos[0] < max_rank:
            hits[pos[0]:] += 1
    return hits / N


def compute_eer(genuine, impostor):
    thresholds = np.linspace(-1, 1, 4000)
    best, best_t = 1.0, 0.0
    for t in thresholds:
        far = np.mean(impostor >= t)
        frr = np.mean(genuine  <  t)
        if (far + frr) / 2 < best:
            best, best_t = (far + frr) / 2, t
    return best, best_t


def tar_at_far(genuine, impostor, far_target):
    for t in np.sort(impostor)[::-1]:
        if np.mean(impostor >= t) <= far_target:
            return np.mean(genuine >= t)
    return 0.0


def roc_data(genuine, impostor, n=600):
    ts = np.linspace(impostor.min(), impostor.max(), n)
    far = np.array([np.mean(impostor >= t) for t in ts])
    tar = np.array([np.mean(genuine  >= t) for t in ts])
    return far, tar


def auc_score(far, tar):
    order = np.argsort(far)
    return float(np.trapz(tar[order], far[order]))


def precision_recall_data(genuine, impostor, n=600):
    ts = np.linspace(-1, 1, n)
    prec, rec = [], []
    for t in ts:
        tp = np.sum(genuine  >= t)
        fp = np.sum(impostor >= t)
        fn = np.sum(genuine  <  t)
        prec.append(tp / max(tp + fp, 1))
        rec.append( tp / max(tp + fn, 1))
    return np.array(rec), np.array(prec)


def far_frr_vs_thresh(genuine, impostor, n=600):
    ts  = np.linspace(-1, 1, n)
    far = np.array([np.mean(impostor >= t) for t in ts])
    frr = np.array([np.mean(genuine  <  t) for t in ts])
    return ts, far, frr


# ──────────────────────────────────────────────────────────────────────────────
#  Save helper
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 01 — CMC Curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_01_cmc(cmc, out_dir):
    ranks = np.arange(1, len(cmc) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ranks, cmc * 100, color=C_BLUE, linewidth=2.5,
            marker="o", markersize=5)
    ax.fill_between(ranks, cmc * 100, alpha=0.12, color=C_BLUE)
    for r in [1, 5, 10]:
        if r <= len(cmc):
            ax.annotate(f"R{r}: {cmc[r-1]*100:.1f}%",
                        xy=(r, cmc[r-1] * 100),
                        xytext=(r + 0.3, cmc[r-1] * 100 - 4),
                        fontsize=9, color=C_RED,
                        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Identification Rate (%)", fontsize=12)
    ax.set_title("01 — CMC Curve (Cumulative Match Characteristic)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(ranks); ax.set_ylim(0, 108); ax.grid(alpha=0.3)
    _save(fig, "01_cmc_curve.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 02 — ROC Curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_02_roc(far, tar, eer, auc_val, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(far * 100, tar * 100, color=C_PURPLE, linewidth=2.5,
            label=f"ROC  (AUC = {auc_val:.4f})")
    ax.plot([0, 100], [0, 100], "--", color=C_GREY, linewidth=1, label="Random")
    ax.scatter([eer * 100], [(1 - eer) * 100], color=C_RED,
               zorder=6, s=90, label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("FAR (%)", fontsize=12)
    ax.set_ylabel("TAR (%)", fontsize=12)
    ax.set_title("02 — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, "02_roc_curve.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 03 — DET Curve (log-log)
# ══════════════════════════════════════════════════════════════════════════════
def plot_03_det(far, tar, eer, out_dir):
    frr  = 1 - tar
    mask = (far > 0) & (frr > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(far[mask] * 100, frr[mask] * 100, color=C_GREEN,
            linewidth=2.5, label="DET curve")
    ax.scatter([eer * 100], [eer * 100], color=C_RED, zorder=6, s=90,
               label=f"EER = {eer*100:.2f}%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FAR (%)", fontsize=12)
    ax.set_ylabel("FRR (%)", fontsize=12)
    ax.set_title("03 — DET Curve (Detection Error Trade-off, log scale)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(which="both", alpha=0.25)
    _save(fig, "03_det_curve.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 04 — Score Distribution with KDE
# ══════════════════════════════════════════════════════════════════════════════
def plot_04_score_dist(genuine, impostor, eer_thresh, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(impostor, bins=80, density=True, alpha=0.30,
            color=C_RED,  label="Impostor (diff. subject)")
    ax.hist(genuine,  bins=80, density=True, alpha=0.30,
            color=C_BLUE, label="Genuine (same subject)")

    # KDE overlays (scipy optional)
    try:
        from scipy.stats import gaussian_kde
        for data, color in [(impostor, C_RED), (genuine, C_BLUE)]:
            if len(data) > 1:
                xs  = np.linspace(data.min() - 0.05, data.max() + 0.05, 400)
                kde = gaussian_kde(data, bw_method="scott")
                ax.plot(xs, kde(xs), color=color, linewidth=2.5)
    except ImportError:
        pass

    ax.axvline(eer_thresh, color=C_ORANGE, linewidth=2, linestyle="--",
               label=f"EER threshold = {eer_thresh:.3f}")
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("04 — Genuine vs Impostor Score Distribution (with KDE)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, "04_score_distribution.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 05 — Cosine Similarity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_05_heatmap(sim, labels, out_dir):
    n   = sim.shape[0]
    lbl = labels.tolist()
    fig, ax = plt.subplots(figsize=(max(10, n // 8), max(8, n // 8)))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cosine Similarity")

    unique  = sorted(set(lbl))
    bounds  = [max(i for i, l in enumerate(lbl) if l == c) + 0.5
               for c in unique][:-1]
    for b in bounds:
        ax.axhline(b, color="black", linewidth=0.5, alpha=0.7)
        ax.axvline(b, color="black", linewidth=0.5, alpha=0.7)

    step = max(1, n // 40)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"S{i}\nL{int(lbl[i])}" for i in ticks],
                       fontsize=6, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"S{i}\nL{int(lbl[i])}" for i in ticks], fontsize=6)

    mask = ~np.eye(n, dtype=bool)
    ax.set_title(
        f"05 — Cosine Similarity Matrix  ({n}×{n}, {len(unique)} classes)\n"
        f"self={np.diag(sim).mean():.4f}   cross={sim[mask].mean():.4f}",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_similarity_heatmap.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 06 — t-SNE Embedding Scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_06_tsne(embeddings, labels, out_dir):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [SKIP 06] scikit-learn not installed — skipping t-SNE.")
        return

    n_samples  = embeddings.shape[0]
    perplexity = min(30, max(5, n_samples // 4))
    print(f"  t-SNE: n={n_samples}, perplexity={perplexity} …")

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000, init="pca")
    proj = tsne.fit_transform(embeddings)

    unique = sorted(set(labels.tolist()))
    cmap   = plt.cm.get_cmap("tab20", len(unique))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(unique):
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=cmap(i), s=60, alpha=0.85,
                   label=f"Class {cls}",
                   edgecolors="white", linewidths=0.4)

    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.set_title("06 — t-SNE Embedding Visualisation (colour = identity)",
                 fontsize=13, fontweight="bold")
    if len(unique) <= 30:
        ax.legend(fontsize=7,
                  ncol=max(1, len(unique) // 10),
                  loc="best", framealpha=0.7)
    ax.grid(alpha=0.2)
    _save(fig, "06_tsne_embeddings.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 07 — Per-Class Rank-1 Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_07_per_class_rank1(sim, labels, out_dir):
    s = sim.copy(); np.fill_diagonal(s, -np.inf)
    unique = sorted(set(labels.tolist()))
    accs   = []
    for cls in unique:
        idx     = np.where(labels == cls)[0]
        correct = sum(labels[np.argmax(s[i])] == cls for i in idx)
        accs.append(correct / len(idx))

    fig, ax = plt.subplots(figsize=(max(10, len(unique) // 2), 5))
    colors  = [C_GREEN if a == 1.0 else C_BLUE if a >= 0.5 else C_RED
               for a in accs]
    bars    = ax.bar([str(c) for c in unique],
                    [a * 100 for a in accs],
                    color=colors, edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{acc*100:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    ax.axhline(np.mean(accs) * 100, color=C_ORANGE, linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(accs)*100:.1f}%")
    ax.set_xlabel("Class (Subject ID)", fontsize=11)
    ax.set_ylabel("Rank-1 Accuracy (%)", fontsize=11)
    ax.set_title("07 — Per-Class Rank-1 Identification Accuracy",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    patches = [mpatches.Patch(color=C_GREEN,  label="100%"),
               mpatches.Patch(color=C_BLUE,   label="50–99%"),
               mpatches.Patch(color=C_RED,    label="< 50%"),
               mpatches.Patch(color=C_ORANGE, label=f"Mean = {np.mean(accs)*100:.1f}%")]
    ax.legend(handles=patches, fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "07_per_class_rank1.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 08 — Per-Class Intra-Similarity Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_08_per_class_intra(sim, labels, out_dir):
    unique     = sorted(set(labels.tolist()))
    means, stds = [], []
    for cls in unique:
        idx    = np.where(labels == cls)[0]
        scores = [sim[i, j] for i in idx for j in idx if i != j]
        means.append(np.mean(scores) if scores else 0)
        stds.append(np.std(scores)   if scores else 0)

    fig, ax = plt.subplots(figsize=(max(10, len(unique) // 2), 5))
    xs = np.arange(len(unique))
    ax.bar(xs, means, yerr=stds, color=C_BLUE, alpha=0.75,
           edgecolor="white", capsize=3,
           error_kw={"elinewidth": 1})
    ax.axhline(np.mean(means), color=C_ORANGE, linestyle="--",
               linewidth=1.5, label=f"Overall mean = {np.mean(means):.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in unique], fontsize=7, rotation=45)
    ax.set_xlabel("Class (Subject ID)", fontsize=11)
    ax.set_ylabel("Mean Intra-class Cosine Similarity", fontsize=11)
    ax.set_title("08 — Per-Class Intra-Class Similarity (mean ± std)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "08_per_class_intra_sim.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 09 — Confusion Matrix / Top Confused Pairs
# ══════════════════════════════════════════════════════════════════════════════
def plot_09_confusion(sim, labels, out_dir, top_n=10):
    s      = sim.copy(); np.fill_diagonal(s, -np.inf)
    unique = sorted(set(labels.tolist()))
    n_cls  = len(unique)

    confusion = np.zeros((n_cls, n_cls))
    for i in range(sim.shape[0]):
        pred_idx = np.argmax(s[i])
        tc = unique.index(labels[i])
        pc = unique.index(labels[pred_idx])
        confusion[tc, pc] += 1

    if n_cls <= 30:
        fig, ax = plt.subplots(figsize=(max(8, n_cls // 2),
                                        max(6, n_cls // 2)))
        im = ax.imshow(confusion, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="# Predictions")
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels([str(c) for c in unique], rotation=45, fontsize=7)
        ax.set_yticklabels([str(c) for c in unique], fontsize=7)
        ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")
        ax.set_title("09 — Confusion Matrix (Rank-1 Identification)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig, "09_confusion_matrix.png", out_dir)
    else:
        conf_err = confusion.copy(); np.fill_diagonal(conf_err, 0)
        pairs = sorted(
            [(conf_err[i, j], unique[i], unique[j])
             for i in range(n_cls) for j in range(n_cls)
             if i != j and conf_err[i, j] > 0],
            reverse=True
        )[:top_n]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([f"{a}→{b}" for _, a, b in pairs],
               [v for v, _, _ in pairs],
               color=C_RED, alpha=0.8, edgecolor="white")
        ax.set_xlabel("True → Predicted Class", fontsize=11)
        ax.set_ylabel("# Confusion Instances", fontsize=11)
        ax.set_title(f"09 — Top-{top_n} Most Confused Class Pairs",
                     fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _save(fig, "09_confusion_top.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 10 — Embedding Statistics per Sample
# ══════════════════════════════════════════════════════════════════════════════
def plot_10_embedding_stats(embeddings, labels, out_dir):
    norms = np.linalg.norm(embeddings, axis=1)
    means = embeddings.mean(axis=1)
    stds  = embeddings.std(axis=1)

    unique = sorted(set(labels.tolist()))
    cmap   = plt.cm.get_cmap("tab20", len(unique))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, title in zip(
        axes,
        [norms, means, stds],
        ["L2 Norm (should ≈ 1.0)", "Embedding Mean", "Embedding Std"]
    ):
        for i, cls in enumerate(unique):
            mask = labels == cls
            ax.scatter(np.where(mask)[0], values[mask],
                       color=cmap(i), s=20, alpha=0.7)
        ax.axhline(np.mean(values), color=C_ORANGE, linewidth=1.5,
                   linestyle="--", label=f"Mean = {np.mean(values):.3f}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)

    fig.suptitle("10 — Per-Sample Embedding Statistics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_embedding_stats.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 11 — Learned Modal Weights
# ══════════════════════════════════════════════════════════════════════════════
def plot_11_modal_weights(model, out_dir):
    raw     = model.modal_weight.detach().cpu()
    weights = torch.softmax(raw, dim=0).numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Fingerprint", "Vein"],
                  weights * 100,
                  color=[C_BLUE, C_GREEN],
                  edgecolor="white", width=0.4)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{w*100:.2f}%",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Softmax Weight (%)", fontsize=11)
    ax.set_title("11 — Learned Modality Weights\n"
                 "(how much the model relies on each input)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    _save(fig, "11_modal_weights.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 12 — Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_12_precision_recall(genuine, impostor, out_dir):
    rec, prec = precision_recall_data(genuine, impostor)
    ap = float(np.trapz(prec[np.argsort(rec)], np.sort(rec)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec * 100, prec * 100, color=C_GREEN, linewidth=2.5,
            label=f"PR curve  (AP = {ap:.4f})")
    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("12 — Precision-Recall Curve",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, "12_precision_recall.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 13 — FAR / FRR vs Threshold
# ══════════════════════════════════════════════════════════════════════════════
def plot_13_far_frr(genuine, impostor, eer, eer_thresh, out_dir):
    ts, far, frr = far_frr_vs_thresh(genuine, impostor)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, far * 100, color=C_RED,    linewidth=2.2, label="FAR")
    ax.plot(ts, frr * 100, color=C_BLUE,   linewidth=2.2, label="FRR")
    ax.axvline(eer_thresh, color=C_ORANGE, linestyle="--", linewidth=1.5,
               label=f"EER threshold = {eer_thresh:.3f}  (EER={eer*100:.2f}%)")
    ax.axhline(eer * 100, color=C_ORANGE, linestyle=":", linewidth=1.2)
    ax.set_xlabel("Decision Threshold (cosine similarity)", fontsize=12)
    ax.set_ylabel("Error Rate (%)", fontsize=12)
    ax.set_title("13 — FAR / FRR vs Decision Threshold",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, "13_far_frr_curve.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 14 — Intra vs Inter Box Plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_14_box(genuine, impostor, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(
        [genuine, impostor],
        patch_artist=True,
        notch=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    bp["boxes"][0].set_facecolor(C_BLUE + "99")
    bp["boxes"][1].set_facecolor(C_RED  + "99")
    ax.set_xticklabels(["Genuine\n(same identity)",
                        "Impostor\n(different identity)"], fontsize=11)
    ax.set_ylabel("Cosine Similarity Score", fontsize=11)
    ax.set_title("14 — Intra vs Inter-class Score Distribution (Box + Notch)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (data, label) in enumerate(
        [(genuine, f"Genuine mean={np.mean(genuine):.3f}"),
         (impostor, f"Impostor mean={np.mean(impostor):.3f}")], start=1
    ):
        ax.scatter(i, np.mean(data), color="black", zorder=5,
                   s=60, marker="D")
    _save(fig, "14_intra_inter_box.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 15 — Training Loss Curves (from CSV)
# ══════════════════════════════════════════════════════════════════════════════
def plot_15_loss_curves(out_dir):
    if not os.path.isfile(LOSS_CSV_PATH):
        print(f"  [SKIP 15] loss_curves.csv not found at: {LOSS_CSV_PATH}")
        return

    epochs, train_l, val_l, arc_l, tri_l = [], [], [], [], []
    with open(LOSS_CSV_PATH) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_l.append(float(row["train_loss"]))
            val_l.append(float(row["val_loss"]))
            arc_l.append(float(row["arc_loss"]))
            tri_l.append(float(row["tri_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_l, color=C_BLUE, linewidth=2, label="Train loss")
    axes[0].plot(epochs, val_l,   color=C_RED,  linewidth=2, label="Val loss")
    axes[0].fill_between(epochs, train_l, val_l, alpha=0.08, color=C_GREY)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Validation Loss",
                      fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, arc_l, color=C_PURPLE, linewidth=2, label="ArcFace")
    axes[1].plot(epochs, tri_l, color=C_GREEN,  linewidth=2, label="Triplet")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("ArcFace vs Triplet Loss",
                      fontsize=12, fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("15 — Training Loss Curves (replayed from CSV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "15_loss_curves.png", out_dir)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-class CSV
# ──────────────────────────────────────────────────────────────────────────────
def save_per_class_csv(sim, labels, out_dir):
    s      = sim.copy(); np.fill_diagonal(s, -np.inf)
    unique = sorted(set(labels.tolist()))
    rows   = []
    for cls in unique:
        idx  = np.where(labels == cls)[0]
        oidx = np.where(labels != cls)[0]
        intra  = [sim[i, j] for i in idx for j in idx if i != j]
        inter  = [sim[i, j] for i in idx for j in oidx]
        correct = sum(labels[np.argmax(s[i])] == cls for i in idx)
        rows.append({
            "class":          cls,
            "n_samples":      len(idx),
            "intra_sim_mean": round(float(np.mean(intra)), 4) if intra else "N/A",
            "intra_sim_std":  round(float(np.std(intra)),  4) if intra else "N/A",
            "inter_sim_mean": round(float(np.mean(inter)), 4) if inter else "N/A",
            "inter_sim_std":  round(float(np.std(inter)),  4) if inter else "N/A",
            "rank1_acc":      round(correct / len(idx), 4),
        })
    path = os.path.join(out_dir, "per_class_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Single-pair helpers
# ──────────────────────────────────────────────────────────────────────────────
def score_single_pair(model, fp_path, vein_path, device):
    fp_t, vn_t = load_image_pair(fp_path, vein_path)
    model.eval()
    with torch.no_grad():
        emb = model(fp_t.to(device), vn_t.to(device))
    e = emb.cpu().numpy()
    print(f"  Embedding shape : {e.shape}")
    print(f"  L2 norm         : {np.linalg.norm(e):.6f}  (should ≈ 1.0)")
    return e


def compare_two_pairs(model, fp1, vn1, fp2, vn2, device):
    fp1t, vn1t = load_image_pair(fp1, vn1)
    fp2t, vn2t = load_image_pair(fp2, vn2)
    model.eval()
    with torch.no_grad():
        e1 = model(fp1t.to(device), vn1t.to(device))
        e2 = model(fp2t.to(device), vn2t.to(device))
    sim = F.cosine_similarity(e1, e2).item()
    print(f"\n{'='*52}")
    print(f"  Cosine similarity : {sim:.6f}")
    if   sim > 0.85: verdict = "Very likely the SAME identity ✓"
    elif sim > 0.60: verdict = "Possibly the same identity (borderline)"
    elif sim > 0.30: verdict = "Likely DIFFERENT identities"
    else:            verdict = "Almost certainly DIFFERENT identities ✗"
    print(f"  Verdict           : {verdict}")
    print(f"{'='*52}")


# ══════════════════════════════════════════════════════════════════════════════
#  Full evaluation orchestrator
# ══════════════════════════════════════════════════════════════════════════════
def run_full_evaluation(model, dataset, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sep = "═" * 60
    print(f"\n{sep}")
    print("  Intelligent Biometric System — Evaluation Suite")
    print(sep)
    print(f"  Samples   : {len(dataset)}")
    print(f"  Classes   : {dataset.num_classes}")
    print(f"  Device    : {device}")
    print(f"  Output    : {out_dir}")
    print(sep + "\n")

    print("► [1/6] Extracting embeddings …")
    embeddings, labels = extract_embeddings(model, dataset, device)
    print(f"        Shape: {embeddings.shape}")

    print("► [2/6] Computing cosine similarity matrix …")
    sim = cosine_matrix(embeddings)

    print("► [3/6] Splitting genuine / impostor pairs …")
    genuine, impostor = genuine_impostor(sim, labels)
    print(f"        Genuine: {len(genuine)}   Impostor: {len(impostor)}")

    print("► [4/6] Computing metrics …")
    r1       = rank1_acc(sim, labels)
    cmc      = cmc_curve(sim, labels, max_rank=min(20, len(dataset) - 1))
    eer, et  = compute_eer(genuine, impostor)
    t01      = tar_at_far(genuine, impostor, 0.001)
    t1       = tar_at_far(genuine, impostor, 0.01)
    t10      = tar_at_far(genuine, impostor, 0.10)
    far, tar = roc_data(genuine, impostor)
    auc_val  = auc_score(far, tar)
    mask     = ~np.eye(len(labels), dtype=bool)
    mean_self  = float(np.diag(sim).mean())
    mean_cross = float(sim[mask].mean())

    print(f"\n{'─'*52}")
    print("  RESULTS")
    print(f"{'─'*52}")
    print(f"  Rank-1  Accuracy      : {r1*100:7.3f} %")
    print(f"  Rank-5  Accuracy      : {cmc[min(4,len(cmc)-1)]*100:7.3f} %")
    print(f"  Rank-10 Accuracy      : {cmc[min(9,len(cmc)-1)]*100:7.3f} %")
    print(f"  AUC-ROC               : {auc_val:7.4f}")
    print(f"  EER                   : {eer*100:7.3f} %  (threshold {et:.4f})")
    print(f"  TAR @ FAR = 0.1%      : {t01*100:7.3f} %")
    print(f"  TAR @ FAR = 1.0%      : {t1*100:7.3f} %")
    print(f"  TAR @ FAR = 10.0%     : {t10*100:7.3f} %")
    print(f"  Mean self-similarity  : {mean_self:7.4f}")
    print(f"  Mean cross-similarity : {mean_cross:7.4f}")
    print(f"  Discrimination gap    : {mean_self - mean_cross:7.4f}")
    print(f"{'─'*52}\n")

    print("► [5/6] Saving CSVs …")
    summary_path = os.path.join(out_dir, "evaluation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in [
            ("Rank-1 Accuracy (%)",    f"{r1*100:.4f}"),
            ("Rank-5 Accuracy (%)",    f"{cmc[min(4,len(cmc)-1)]*100:.4f}"),
            ("Rank-10 Accuracy (%)",   f"{cmc[min(9,len(cmc)-1)]*100:.4f}"),
            ("AUC-ROC",                f"{auc_val:.6f}"),
            ("EER (%)",                f"{eer*100:.4f}"),
            ("EER Threshold",          f"{et:.6f}"),
            ("TAR @ FAR=0.1% (%)",     f"{t01*100:.4f}"),
            ("TAR @ FAR=1.0% (%)",     f"{t1*100:.4f}"),
            ("TAR @ FAR=10.0% (%)",    f"{t10*100:.4f}"),
            ("Mean Self-Similarity",   f"{mean_self:.6f}"),
            ("Mean Cross-Similarity",  f"{mean_cross:.6f}"),
            ("Discrimination Gap",     f"{mean_self-mean_cross:.6f}"),
            ("Total Samples",          len(dataset)),
            ("Total Classes",          dataset.num_classes),
            ("Genuine Pairs",          len(genuine)),
            ("Impostor Pairs",         len(impostor)),
        ]:
            w.writerow([k, v])
    print(f"  Saved → {summary_path}")

    save_per_class_csv(sim, labels, out_dir)

    cmc_csv = os.path.join(out_dir, "cmc_data.csv")
    with open(cmc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "identification_rate_%"])
        for r, v in enumerate(cmc, 1):
            w.writerow([r, f"{v*100:.4f}"])
    print(f"  Saved → {cmc_csv}")

    print("► [6/6] Generating 15 plots …")
    plot_01_cmc(cmc, out_dir)
    plot_02_roc(far, tar, eer, auc_val, out_dir)
    plot_03_det(far, tar, eer, out_dir)
    plot_04_score_dist(genuine, impostor, et, out_dir)
    plot_05_heatmap(sim, labels, out_dir)
    plot_06_tsne(embeddings, labels, out_dir)
    plot_07_per_class_rank1(sim, labels, out_dir)
    plot_08_per_class_intra(sim, labels, out_dir)
    plot_09_confusion(sim, labels, out_dir)
    plot_10_embedding_stats(embeddings, labels, out_dir)
    plot_11_modal_weights(model, out_dir)
    plot_12_precision_recall(genuine, impostor, out_dir)
    plot_13_far_frr(genuine, impostor, eer, et, out_dir)
    plot_14_box(genuine, impostor, out_dir)
    plot_15_loss_curves(out_dir)

    print(f"\n✅  All done!  {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fp",    default=None)
    p.add_argument("--vein",  default=None)
    p.add_argument("--fp2",   default=None)
    p.add_argument("--vein2", default=None)
    p.add_argument("--eval",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    BiometricDataset, FusionModel = _import_project_modules()

    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Train first:  python train.py")
        sys.exit(1)

    print(f"\nModel  : {MODEL_PATH}")
    print(f"Device : {DEVICE}")
    model = FusionModel(embed_dim=256, num_heads=8,
                        num_layers=2, output_dim=512).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Params : {model.count_total_params():,} total  "
          f"/ {model.count_trainable_params():,} trainable")

    if args.fp and args.vein:
        print("\nSingle-pair mode")
        score_single_pair(model, args.fp, args.vein, DEVICE)
        if args.fp2 and args.vein2:
            compare_two_pairs(model, args.fp, args.vein,
                               args.fp2, args.vein2, DEVICE)
        if not args.eval:
            return

    print(f"\nDataset: {DATASET_ROOT}")
    dataset = BiometricDataset(root_dir=DATASET_ROOT)
    run_full_evaluation(model, dataset, DEVICE, out_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()