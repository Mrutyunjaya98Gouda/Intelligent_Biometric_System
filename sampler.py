"""
sampler.py — Session-Aware Balanced Batch Sampler
==================================================
Standard random sampling rarely puts two samples of the SAME person from
DIFFERENT sessions in the same mini-batch, so VerificationLoss and SupConLoss
have few genuine cross-session pairs to learn from.

SessionAwareSampler guarantees:
  • Every batch contains exactly `n_subjects` classes.
  • Each class contributes exactly `n_per_class` samples, drawn from
    DIFFERENT sessions wherever possible.

This means every batch contains cross-session genuine pairs — giving
VerificationLoss and SupConLoss rich supervision to push same-person
similarity above 0.85.

Parameters
----------
dataset       : BiometricDataset instance (or any object with .samples list
                of (fp_path, vein_path, label) tuples).
n_subjects    : number of subjects per batch  (default 8).
n_per_class   : samples per subject per batch (default 4).
                → effective batch size = n_subjects × n_per_class
drop_last     : bool  (default False)
"""

import random
from collections import defaultdict
from torch.utils.data import Sampler


class SessionAwareSampler(Sampler):
    """
    Yields batches of indices such that:
      1. Exactly `n_subjects` distinct classes appear in each batch.
      2. For each class, `n_per_class` samples are selected, preferring
         samples from distinct sessions when available.

    Compatible with DataLoader(batch_sampler=...) — each call to __iter__
    yields one list of indices (one complete batch).
    """

    def __init__(
        self,
        dataset,
        n_subjects:  int  = 8,
        n_per_class: int  = 4,
        drop_last:   bool = False,
    ):
        self.n_subjects  = n_subjects
        self.n_per_class = n_per_class
        self.drop_last   = drop_last

        # Build  label → list of (index, session_key)
        # session_key is extracted from the filepath: "Session1", "Session2", …
        self.label_to_samples: dict[int, list[tuple[int, str]]] = defaultdict(list)

        for idx, (fp_path, vein_path, label) in enumerate(dataset.samples):
            # Extract session from path e.g. ".../Session1/Fingerprint/..."
            session = "unknown"
            for part in fp_path.replace("\\", "/").split("/"):
                if part.lower().startswith("session"):
                    session = part
                    break
            self.label_to_samples[int(label)].append((idx, session))

        self.all_labels = list(self.label_to_samples.keys())

        # Estimate total batches
        max_per_class = max(len(v) for v in self.label_to_samples.values())
        n_labels      = len(self.all_labels)
        self._len     = max(1, (n_labels * max_per_class) //
                              (n_subjects * n_per_class))

    # ──────────────────────────────────────────────────────────────────────
    def _pick_diverse(
        self,
        samples: list[tuple[int, str]],
        n: int,
    ) -> list[int]:
        """
        Pick `n` sample indices from `samples`, maximising session diversity.
        Falls back to random sampling if n > len(samples).
        """
        if len(samples) <= n:
            return [s[0] for s in samples]

        # Group by session
        by_session: dict[str, list[int]] = defaultdict(list)
        for idx, sess in samples:
            by_session[sess].append(idx)

        sessions = list(by_session.keys())
        random.shuffle(sessions)

        picked = []
        while len(picked) < n:
            progress = False
            for sess in sessions:
                pool = by_session[sess]
                if pool:
                    chosen = random.choice(pool)
                    pool.remove(chosen)
                    picked.append(chosen)
                    progress = True
                    if len(picked) == n:
                        break
            if not progress:
                # All sessions exhausted — pad with random repeats
                all_idx = [s[0] for s in samples]
                while len(picked) < n:
                    picked.append(random.choice(all_idx))
                break

        return picked[:n]

    # ──────────────────────────────────────────────────────────────────────
    def __iter__(self):
        labels = self.all_labels.copy()
        random.shuffle(labels)

        for start in range(
            0,
            len(labels) - self.n_subjects + 1,
            self.n_subjects,
        ):
            batch_labels  = labels[start: start + self.n_subjects]
            batch_indices = []
            for lbl in batch_labels:
                samples = self.label_to_samples[lbl].copy()
                random.shuffle(samples)
                batch_indices.extend(
                    self._pick_diverse(samples, self.n_per_class)
                )
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self._len