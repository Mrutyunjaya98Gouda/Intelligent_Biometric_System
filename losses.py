"""
losses.py — Loss Functions for the Intelligent Biometric System
===============================================================
Provides five losses used by train.py:

  ArcFaceLoss       — Additive Angular Margin Softmax (m=0.65).
                      Pushes same-identity embeddings into tight angular
                      clusters on the unit hypersphere.

  TripletLoss       — Batch-hard triplet mining with L2 distance.
                      Ensures genuine pairs are closer than impostor pairs
                      by at least `margin`.

  SupConLoss        — Supervised Contrastive Loss (Khosla et al., 2020).
                      For every anchor ALL same-class samples are positives;
                      directly optimises the cosine-similarity distribution
                      that TAR@FAR measures.

  VerificationLoss  — Explicit pairwise cosine-similarity hinge:
                        genuine  pairs → score >= pos_margin (0.85)
                        impostor pairs → score <= neg_margin (0.20)

  IntraClassLoss    — Pulls each embedding toward its class prototype (mean),
                      minimising intra-class variance.

  CombinedLoss      — Weighted sum of all five.  Used by train.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  ArcFace Loss
#  Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
#  (Deng et al., 2019)
# ══════════════════════════════════════════════════════════════════════════════

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss.

    Parameters
    ----------
    in_features  : embedding dimension (must match model output_dim = 512)
    out_features : number of training identities
    s            : feature scale  (default 32.0)
    m            : additive angular margin in radians  (default 0.65)
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        s:   float = 32.0,
        m:   float = 0.65,
    ):
        super().__init__()
        if out_features < 2:
            raise ValueError(
                f"ArcFaceLoss requires out_features >= 2, got {out_features}."
            )
        self.s         = s
        self.m         = m
        self.weight    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m     = math.cos(m)
        self.sin_m     = math.sin(m)
        self.th        = math.cos(math.pi - m)   # -cos(m)
        self.mm        = math.sin(math.pi - m) * m
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight,  dim=1),
        )
        sine   = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        logits  = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return self.criterion(logits, labels)


# ══════════════════════════════════════════════════════════════════════════════
#  Triplet Loss  (batch-hard mining, L2 distance)
# ══════════════════════════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """
    Batch-hard triplet mining using L2 distance.

    For each anchor selects:
      • hardest positive  (most distant same-class sample)
      • hardest negative  (closest different-class sample)

    Parameters
    ----------
    margin : float  — triplet margin (default 0.3)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin       = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def _pairwise_l2(self, emb: torch.Tensor) -> torch.Tensor:
        dot = torch.mm(emb, emb.t())
        sq  = dot.diag().unsqueeze(1) + dot.diag().unsqueeze(0) - 2.0 * dot
        return torch.clamp(sq, min=0.0)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B        = embeddings.size(0)
        dist_sq  = self._pairwise_l2(embeddings)

        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
        neg_mask = ~pos_mask
        eye      = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask = pos_mask & ~eye

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        INF             = dist_sq.max().item() + 1.0
        hardest_pos_idx = (dist_sq * pos_mask.float()).argmax(dim=1)
        hardest_neg_idx = (dist_sq + (~neg_mask).float() * INF).argmin(dim=1)

        vi = valid.nonzero(as_tuple=False).squeeze(1)
        return self.triplet_loss(
            embeddings[vi],
            embeddings[hardest_pos_idx[vi]],
            embeddings[hardest_neg_idx[vi]],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Supervised Contrastive Loss
#  Paper: "Supervised Contrastive Learning" (Khosla et al., 2020)
# ══════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    For every anchor i, ALL other samples with the same label are positives.
    The loss maximises the sum of positive similarities relative to all
    pairwise similarities (log-softmax style).

    Directly optimises the cosine-similarity distribution:
      same-person scores   → high  (improves TAR)
      different-person     → low   (improves FAR)

    Parameters
    ----------
    temperature : float — scaling factor (default 0.07, standard from paper)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B    = embeddings.size(0)
        emb  = F.normalize(embeddings, dim=1)
        sim  = torch.mm(emb, emb.t()) / self.temperature   # (B, B)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        eye       = torch.eye(B, dtype=torch.bool, device=emb.device)
        pos_mask  = labels_eq & ~eye
        has_pos   = pos_mask.any(dim=1)

        if not has_pos.any():
            return torch.tensor(0.0, device=emb.device, requires_grad=True)

        # Numerical stability: subtract row max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        all_mask  = ~eye
        exp_sim   = torch.exp(sim) * all_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        log_prob  = sim - log_denom                              # (B, B)
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
        loss_per  = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count

        return loss_per[has_pos].mean()


# ══════════════════════════════════════════════════════════════════════════════
#  Verification Loss  (explicit pairwise cosine targets)
# ══════════════════════════════════════════════════════════════════════════════

class VerificationLoss(nn.Module):
    """
    Pairwise cosine-similarity hinge loss.

    Genuine  pairs: loss += max(0,  pos_margin - cosine_sim(i, j))
    Impostor pairs: loss += max(0,  cosine_sim(i, j) - neg_margin)

    Gives the model explicit numerical score targets, not just angular
    boundaries.

    Parameters
    ----------
    pos_margin : float — minimum desired genuine similarity (default 0.85)
    neg_margin : float — maximum allowed impostor similarity (default 0.20)
    max_pairs  : int   — cap pairs per batch to limit memory (default 2048)
    """

    def __init__(
        self,
        pos_margin: float = 0.85,
        neg_margin: float = 0.20,
        max_pairs:  int   = 2048,
    ):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.max_pairs  = max_pairs

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        emb      = F.normalize(embeddings, dim=1)
        sim      = torch.mm(emb, emb.t())
        B        = embeddings.size(0)
        eye      = torch.eye(B, dtype=torch.bool, device=emb.device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))

        pos_sim = sim[pos_mask]
        neg_sim = sim[neg_mask]

        if pos_sim.numel() == 0:
            pos_loss = torch.tensor(0.0, device=emb.device, requires_grad=True)
        else:
            if pos_sim.numel() > self.max_pairs:
                idx = torch.randperm(pos_sim.numel(),
                                     device=emb.device)[:self.max_pairs]
                pos_sim = pos_sim[idx]
            pos_loss = F.relu(self.pos_margin - pos_sim).mean()

        if neg_sim.numel() == 0:
            neg_loss = torch.tensor(0.0, device=emb.device, requires_grad=True)
        else:
            if neg_sim.numel() > self.max_pairs:
                idx = torch.randperm(neg_sim.numel(),
                                     device=emb.device)[:self.max_pairs]
                neg_sim = neg_sim[idx]
            neg_loss = F.relu(neg_sim - self.neg_margin).mean()

        return pos_loss + neg_loss, pos_loss.detach(), neg_loss.detach()


# ══════════════════════════════════════════════════════════════════════════════
#  Intra-Class Loss  (prototype pulling)
# ══════════════════════════════════════════════════════════════════════════════

class IntraClassLoss(nn.Module):
    """
    Pulls each embedding toward its class prototype (mean embedding).

    Loss = mean over all samples of  (1 - cosine_sim(embedding, prototype))

    Complements SupCon: SupCon works on pairs while this minimises spread
    around the class centroid.
    """

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb    = F.normalize(embeddings, dim=1)
        unique = labels.unique()
        losses = []
        for cls in unique:
            mask      = labels == cls
            cls_embs  = emb[mask]
            if cls_embs.size(0) < 2:
                continue
            prototype = F.normalize(cls_embs.mean(dim=0, keepdim=True), dim=1)
            sim       = (cls_embs * prototype).sum(dim=1)
            losses.append((1.0 - sim).mean())

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  CombinedLoss  (used by train.py)
# ══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Weighted combination targeting both identification AND verification:

        total = ArcFace(m=0.65)
              + w_tri    * TripletLoss       (batch-hard)
              + w_supcon * SupConLoss        (pair-level attraction/repulsion)
              + w_verif  * VerificationLoss  (explicit cosine targets)
              + w_intra  * IntraClassLoss    (prototype pulling)

    Parameters
    ----------
    num_classes  : int   — number of training identities (REQUIRED)
    s            : float — ArcFace scale           (default 32.0)
    m            : float — ArcFace angular margin  (default 0.65)
    w_tri        : float — Triplet weight           (default 0.20)
    w_supcon     : float — SupCon weight            (default 0.50)
    w_verif      : float — Verification loss weight (default 0.30)
    w_intra      : float — Intra-class loss weight  (default 0.10)
    pos_margin   : float — genuine pair cosine target  (default 0.85)
    neg_margin   : float — impostor pair cosine target (default 0.20)
    temperature  : float — SupCon temperature       (default 0.07)
    """

    def __init__(
        self,
        num_classes: int,
        s:           float = 32.0,
        m:           float = 0.65,
        w_tri:       float = 0.20,
        w_supcon:    float = 0.50,
        w_verif:     float = 0.30,
        w_intra:     float = 0.10,
        pos_margin:  float = 0.85,
        neg_margin:  float = 0.20,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.arcface  = ArcFaceLoss(in_features=512,
                                    out_features=num_classes, s=s, m=m)
        self.triplet  = TripletLoss(margin=0.3)
        self.supcon   = SupConLoss(temperature=temperature)
        self.verif    = VerificationLoss(pos_margin=pos_margin,
                                         neg_margin=neg_margin)
        self.intra    = IntraClassLoss()

        self.w_tri    = w_tri
        self.w_supcon = w_supcon
        self.w_verif  = w_verif
        self.w_intra  = w_intra

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Returns
        -------
        total       : scalar loss
        arc_loss    : scalar (for logging)
        tri_loss    : scalar
        supcon_loss : scalar
        verif_loss  : scalar
        intra_loss  : scalar
        """
        arc_loss              = self.arcface(embeddings, labels)
        tri_loss              = self.triplet(embeddings, labels)
        supcon_loss           = self.supcon(embeddings,  labels)
        verif_loss, _, _      = self.verif(embeddings,   labels)
        intra_loss            = self.intra(embeddings,   labels)

        total = (arc_loss
                 + self.w_tri    * tri_loss
                 + self.w_supcon * supcon_loss
                 + self.w_verif  * verif_loss
                 + self.w_intra  * intra_loss)

        return total, arc_loss, tri_loss, supcon_loss, verif_loss, intra_loss