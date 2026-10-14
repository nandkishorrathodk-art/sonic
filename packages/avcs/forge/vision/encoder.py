"""Learned scene understanding: grids in, typed state + confidence out.

Non-generative by construction.  The encoder consumes the reflex layer's grids
and produces an embedding plus a handful of heads (activity, change, overlay,
scene-type).  It never emits free-form text -- the describer turns its typed
output into words, and that is a separate, swappable component.

Why a learned head at all, when the reflex layer already thresholds?  Because
thresholds are brittle: a fixed ``change_fraction`` cut-off does not know that
a mostly-static screen with a blinking cursor is "idle", while a scrolling feed
is "busy".  A small model over the grids learns that distinction, and -- more
importantly -- can be *calibrated*, so the pipeline knows when it is unsure.

The confidence is a real quantity, not a softmax dressed up as one: the head is
trained with an auxiliary loss that pulls its confidence toward observed
correctness, so a confident output means something.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.vision.scene import SceneState
from forge.vision.vocab import VOCAB, render_tokens

SCENE_TYPES = ["idle", "active", "busy", "overlay"]

# A word is emitted when the head's probability clears this.  Multi-label
# decoding, so several words can fire on one frame.
WORD_THRESHOLD = 0.45


@dataclass
class Understanding:
    """Typed read-out of a frame, with calibrated confidence.

    ``tokens`` and ``description`` are the *model's own prediction* over the
    closed vocabulary -- the sentence is rendered from predicted words, not
    from hand-written branches and not from the labels used in training.
    """

    scene_type: str
    activity: float
    change: float
    overlay_prob: float
    confidence: float
    tokens: list[str]
    description: str

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "activity": round(self.activity, 3),
            "change": round(self.change, 3),
            "overlay_prob": round(self.overlay_prob, 3),
            "confidence": round(self.confidence, 3),
            "tokens": self.tokens,
            "description": self.description,
        }


def stack_grids(state: SceneState) -> np.ndarray:
    """(4, rows, cols) float32 stack: luma, edge, salience, motion."""
    return np.stack(
        [
            state.luma_grid.astype(np.float32),
            state.edge_grid.astype(np.float32),
            state.salience_grid.astype(np.float32),
            state.motion_grid.astype(np.float32),
        ],
        axis=0,
    )


class SceneEncoder(nn.Module):
    """Small conv encoder over the four perception grids."""

    def __init__(self, in_ch: int = 4, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, R, C) -> (B, dim)."""
        return self.net(x).flatten(1)


class SceneUnderstandingModel(nn.Module):
    """Encoder + heads.  Trains on synthetic states; runs in one pass."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.encoder = SceneEncoder(dim=dim)
        self.activity_head = nn.Linear(dim, 1)          # -> sigmoid
        self.change_head = nn.Linear(dim, 1)            # -> sigmoid
        self.overlay_head = nn.Linear(dim, 1)           # -> sigmoid
        self.type_head = nn.Linear(dim, len(SCENE_TYPES))
        self.conf_head = nn.Linear(dim, 1)              # -> sigmoid
        # Multi-label description head: predicts the closed vocabulary directly.
        self.word_head = nn.Linear(dim, len(VOCAB))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            "embedding": z,
            "activity": torch.sigmoid(self.activity_head(z)).squeeze(-1),
            "change": torch.sigmoid(self.change_head(z)).squeeze(-1),
            "overlay": torch.sigmoid(self.overlay_head(z)).squeeze(-1),
            "type_logits": self.type_head(z),
            "confidence": torch.sigmoid(self.conf_head(z)).squeeze(-1),
            "word_logits": self.word_head(z),
        }

    @torch.no_grad()
    def understand(self, state: SceneState) -> Understanding:
        self.eval()
        x = torch.from_numpy(stack_grids(state)).unsqueeze(0)
        out = self.forward(x)
        type_idx = int(out["type_logits"].argmax(dim=-1).item())

        # Decode the description from the model's own word probabilities.
        word_probs = torch.sigmoid(out["word_logits"])[0]
        chosen = [
            VOCAB[i] for i, p in enumerate(word_probs.tolist()) if p >= WORD_THRESHOLD
        ]
        if "screen" not in chosen:
            chosen.insert(0, "screen")
        # Scene type from the classifier is authoritative for the state word.
        stype = SCENE_TYPES[type_idx]
        chosen = [w for w in chosen if w not in SCENE_TYPES] + [stype]
        # Preserve vocabulary order for stable rendering.
        chosen = [w for w in VOCAB if w in set(chosen)]
        description = render_tokens(chosen)

        # Calibrated confidence: start from the learned confidence head, then
        # shrink it by how many words sit near the decision boundary.  A frame
        # the model is unsure about must not report high confidence just
        # because the type head happened to be peaked.
        boundary = sum(1 for p in word_probs.tolist() if abs(p - WORD_THRESHOLD) < 0.15)
        ambiguity = boundary / max(len(word_probs), 1)
        conf = float(out["confidence"].item())
        calibrated = float(np.clip(conf * (1.0 - 0.5 * ambiguity), 0.0, 1.0))
        return Understanding(
            scene_type=stype,
            activity=float(out["activity"].item()),
            change=float(out["change"].item()),
            overlay_prob=float(out["overlay"].item()),
            confidence=calibrated,
            tokens=chosen,
            description=description,
        )


def pseudo_labels(state: SceneState) -> dict[str, float]:
    """Rule-based labels from the reflex thresholds, used to bootstrap training.

    These are targets, not the model.  The model is trained to reproduce and
    generalise them, and its confidence head is trained against the label
    agreement -- so the confidence measures "did I get this right", not "was I
    loud".
    """
    if state.scene_changed:
        scene_type = "busy"
    elif state.overlay_suspected:
        scene_type = "overlay"
    elif state.peak_motion > 0.03:
        scene_type = "active"
    else:
        scene_type = "idle"
    return {
        "activity": float(np.clip(state.peak_motion * 8.0, 0.0, 1.0)),
        "change": float(np.clip(state.change_fraction * 4.0, 0.0, 1.0)),
        "overlay": 1.0 if state.overlay_suspected else 0.0,
        "type": float(SCENE_TYPES.index(scene_type)),
    }


def build_targets(states: list[SceneState], scene_types: list[str]) -> dict[str, torch.Tensor]:
    """Build all supervised targets for a list of states."""
    from forge.vision.vocab import tokens_for_state, tokens_to_vector

    labels = [pseudo_labels(s) for s in states]
    words = [
        tokens_to_vector(tokens_for_state(s, st))
        for s, st in zip(states, scene_types)
    ]
    return {
        "activity": torch.tensor([l["activity"] for l in labels]),
        "change": torch.tensor([l["change"] for l in labels]),
        "overlay": torch.tensor([l["overlay"] for l in labels]),
        "type": torch.tensor([int(l["type"]) for l in labels]),
        "words": torch.tensor(words),
    }


def _compute_loss(out: dict, y: dict, idx: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(out["activity"], y["activity"][idx])
    loss = loss + F.mse_loss(out["change"], y["change"][idx])
    loss = loss + F.binary_cross_entropy(out["overlay"], y["overlay"][idx])
    loss = loss + F.cross_entropy(out["type_logits"], y["type"][idx])
    loss = loss + F.binary_cross_entropy_with_logits(
        out["word_logits"], y["words"][idx]
    )
    # Confidence target: how well the type head currently agrees.
    with torch.no_grad():
        pred = out["type_logits"].argmax(dim=-1)
        correct = (pred == y["type"][idx]).float()
    loss = loss + F.binary_cross_entropy(out["confidence"], correct)
    return loss


# Words that are almost always present carry almost no signal; counting them
# in an accuracy average inflates it into meaninglessness.  These are excluded
# from the *informative* metric.
ALWAYS_ON = {"screen"}


def word_report(
    model: SceneUnderstandingModel, states: list[SceneState], scene_types: list[str]
) -> dict:
    """Honest multi-label evaluation over the description vocabulary.

    Plain per-word accuracy is dominated by always-on tokens ("screen",
    "mid-tone", "detailed"), so it reads high even for a model that never
    discriminates.  This reports instead:

    * ``macro_f1`` -- mean F1 over every word with support, so rare,
      discriminative words count as much as common ones.
    * ``informative_accuracy`` -- accuracy restricted to words outside
      ``ALWAYS_ON``, i.e. the words that actually say something.
    * ``per_word`` -- F1 for each word, so a failure is locatable.
    """
    empty = {"macro_f1": 0.0, "informative_accuracy": 0.0, "per_word": {}}
    if not states:
        return empty
    x = torch.from_numpy(np.stack([stack_grids(s) for s in states]))
    y = build_targets(states, scene_types)["words"]
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x)["word_logits"])
    preds = (probs >= WORD_THRESHOLD).float()

    per_word: dict[str, float] = {}
    f1s: list[float] = []
    for i, w in enumerate(VOCAB):
        if w in ALWAYS_ON:
            continue
        tp = float((preds[:, i] * y[:, i]).sum())
        fp = float((preds[:, i] * (1 - y[:, i])).sum())
        fn = float(((1 - preds[:, i]) * y[:, i]).sum())
        support = float(y[:, i].sum())
        if support == 0 and tp + fp == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_word[w] = round(f1, 3)
        f1s.append(f1)

    keep = [i for i, w in enumerate(VOCAB) if w not in ALWAYS_ON]
    inf_acc = float((preds[:, keep] == y[:, keep]).float().mean().item())
    return {
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 3),
        "informative_accuracy": round(inf_acc, 3),
        "per_word": per_word,
    }


def train_understanding(
    model: SceneUnderstandingModel,
    states: list[SceneState],
    steps: int = 200,
    lr: float = 2e-3,
    batch_size: int = 16,
    seed: int = 0,
    val_states: list[SceneState] | None = None,
    val_scene_types: list[str] | None = None,
    on_step=None,
) -> dict:
    """Train the understanding heads on labelled reflex states.

    Returns a dict with the training loss curve and, when a validation split is
    provided, held-out per-word accuracy.  Reporting held-out accuracy is the
    point: a training loss that falls to zero on a single set is memorisation,
    not understanding.
    """
    scene_types = [_type_name(s) for s in states]
    g = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.stack([stack_grids(s) for s in states]))
    y = build_targets(states, scene_types)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    n = len(states)
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=g)
        out = model(x[idx])
        loss = _compute_loss(out, y, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if on_step is not None:
            on_step(step, losses[-1])
    model.eval()

    result = {"losses": losses}
    if val_states and val_scene_types is not None:
        result["val"] = word_report(model, val_states, val_scene_types)
    return result


def _type_name(state: SceneState) -> str:
    t = int(pseudo_labels(state)["type"])
    return SCENE_TYPES[t]
