"""Turn typed visual state into natural language.

Two backends, one interface:

* :class:`HeuristicDescriber` -- always available, no download.  It phrases the
  reflex state and events.  It is fast and honest about what it knows: it
  describes layout, motion, change and overlay from pixels, and it does not
  pretend to read text it never recognised.
* :class:`VLMDescriber` -- optional.  If a small vision-language model is
  installed locally it captions the raw frame for richer semantics.  It is
  invoked *on change only*, never every tick, because it is far slower than the
  reflex layer.  If it is not installed the pipeline falls back cleanly.

The design point: the non-generative reflex loop carries the real-time load,
and the generative model is a slow, optional consultant.  That is the only
arrangement in which "real-time" and "rich description" coexist on one machine.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from forge.vision.encoder import Understanding
from forge.vision.scene import SceneEvent, SceneState


class Describer(Protocol):
    name: str

    def describe(
        self, state: SceneState, understanding: Optional[Understanding], events: list[SceneEvent]
    ) -> str: ...


def _where(frac_r: float, frac_c: float) -> str:
    if frac_r < 0.34:
        vert = "top"
    elif frac_r < 0.67:
        vert = "middle"
    else:
        vert = "bottom"
    if frac_c < 0.34:
        horiz = "left"
    elif frac_c < 0.67:
        horiz = "centre"
    else:
        horiz = "right"
    if horiz == "centre" and vert == "middle":
        return "centre"
    return f"{vert} {horiz}"


class ModelDescriber:
    """Describes using the trained model's own prediction.

    It renders ``understanding.description`` -- the sentence the model's word
    head produced -- rather than branching on thresholds here.  If no
    understanding is present (model disabled) it falls back to the heuristic
    describer so the pipeline never silently emits nothing.
    """

    name = "model"

    def __init__(self) -> None:
        self._fallback = HeuristicDescriber()

    def describe(
        self, state: SceneState, understanding: Optional[Understanding], events: list[SceneEvent]
    ) -> str:
        if understanding is not None and understanding.description:
            conf = f" (confidence {understanding.confidence:.2f})"
            return understanding.description + conf
        return self._fallback.describe(state, understanding, events)


class HeuristicDescriber:
    """Phrases the typed state.  No model, no download, always available."""

    name = "heuristic"

    def describe(
        self, state: SceneState, understanding: Optional[Understanding], events: list[SceneEvent]
    ) -> str:
        parts: list[str] = []

        tone = "dark" if state.brightness < 0.25 else "bright" if state.brightness > 0.75 else "mid-tone"
        flat = "flat" if state.contrast < 0.08 else "detailed"
        parts.append(f"A {tone}, {flat} screen.")

        if state.regions:
            kinds = [r.kind for r in state.regions[:3]]
            parts.append(f"Structure reads as {', '.join(kinds)}.")

        if state.scene_changed:
            parts.append("The whole view changed at once, like switching apps or pages.")
        elif state.overlay_suspected:
            parts.append("A small, high-contrast overlay sits on top of the scene.")
        elif state.peak_motion > 0.10:
            where = "somewhere"
            if state.motion_centroid is not None:
                r, c = state.motion_centroid
                where = "in the " + _where(r / state.rows, c / state.cols)
            parts.append(f"There is clear movement {where}.")
        elif state.peak_motion > 0.03:
            parts.append("Something is moving gently on screen.")
        else:
            parts.append("The screen is essentially still.")

        if understanding is not None:
            parts.append(
                f"State reads as {understanding.scene_type} "
                f"(confidence {understanding.confidence:.2f})."
            )

        for ev in events:
            parts.append(f"{ev.kind.replace('_', ' ')}: {ev.text}.")

        return " ".join(parts)


class VLMDescriber:
    """Optional local vision-language captioner.

    Loads a small HuggingFace image-text model if ``transformers`` and the
    weights are present.  Construction raises if unavailable, so callers use
    :meth:`try_load` and fall back.  Captions run on change only.
    """

    name = "vlm"

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct", device: str = "cpu") -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: F401

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        self.model.eval()
        self.device = device

    @staticmethod
    def try_load(model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> Optional["VLMDescriber"]:
        try:
            return VLMDescriber(model_name=model_name)
        except Exception:
            return None

    def caption(self, frame: np.ndarray, prompt: str = "Describe what is happening on this computer screen.") -> str:
        from PIL import Image
        import torch

        img = Image.fromarray(frame[:, :, ::-1] if frame.shape[-1] == 3 else frame)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[img], return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def describe(
        self, state: SceneState, understanding: Optional[Understanding], events: list[SceneEvent]
    ) -> str:  # pragma: no cover - needs weights
        # The frame is not carried on the state; the pipeline calls caption()
        # directly when it has the pixels.  This satisfies the protocol shape.
        return HeuristicDescriber().describe(state, understanding, events)
