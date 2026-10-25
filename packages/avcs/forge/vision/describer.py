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
    # Prompt is deliberately descriptive: empirically, a narrow "just name the
    # app" instruction makes SmolVLM-500M answer with a single word ("Xfce."),
    # while a describe-the-screen instruction makes it read real on-screen text
    # ("Example Domain", "Mozilla Firefox").  The trailing clause keeps it from
    # inventing detail.
    DEFAULT_PROMPT = (
        "Describe what is happening on this computer screen. Name any "
        "applications and windows you can see, and read the important visible "
        "text or button labels. Do not guess about things you cannot see."
    )

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 48,
    ) -> None:
        from transformers import AutoProcessor

        # ``AutoModelForVision2Seq`` was removed in transformers 5.x; the same
        # image-text family is now ``AutoModelForImageTextToText``.  Accept both
        # so one code path works on either major version.
        try:
            from transformers import AutoModelForImageTextToText as _ImageTextModel
        except ImportError:  # transformers < 5
            from transformers import AutoModelForVision2Seq as _ImageTextModel

        self.processor = AutoProcessor.from_pretrained(model_name)
        # ``from_pretrained`` temporarily changes the process-global torch
        # default dtype; serialize construction so a concurrently built model
        # (e.g. the AVCS cortex) cannot pick up the wrong dtype.
        from forge.vision.torch_build import build_model

        self.model = build_model(lambda: _ImageTextModel.from_pretrained(model_name))
        self.model.eval()
        self.device = device
        self.max_new_tokens = int(max_new_tokens)

    @staticmethod
    def try_load(model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> Optional["VLMDescriber"]:
        try:
            return VLMDescriber(model_name=model_name)
        except Exception:
            return None

    def caption(self, frame: np.ndarray, prompt: str = "") -> str:
        from PIL import Image
        import torch

        prompt = prompt or self.DEFAULT_PROMPT
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
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        # SmolVLM echoes the chat template; keep only the assistant's reply.
        marker = "Assistant:"
        if marker in decoded:
            decoded = decoded.split(marker, 1)[-1].strip()
        return decoded

    def describe(
        self, state: SceneState, understanding: Optional[Understanding], events: list[SceneEvent]
    ) -> str:  # pragma: no cover - needs weights
        # The frame is not carried on the state; the pipeline calls caption()
        # directly when it has the pixels.  This satisfies the protocol shape.
        return HeuristicDescriber().describe(state, understanding, events)
