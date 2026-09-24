"""
SONIC — Laya System-1 Non-Autoregressive Decision Engine Integration
=====================================================================
Integrates the Laya System-1 decision architecture (https://github.com/NandhaKishorM/laya)
into SONIC's AI-Human Digital Being cognitive core.

Provides sub-35ms single-pass typed judgments (`choice`, `score`, `noul`) across
100+ languages (including English, Hindi, and Hinglish) via:
  1. Remote `laya-serve` neural forward pass (`LAYA_API_URL`, 0 MB host RAM)
  2. Local `laya.Router` neural forward pass (`SONIC_ENABLE_LOCAL_LAYA=1` with RAM guard)
  3. Calibrated System-1 embedding projection over `criteria` descriptions.
"""

from __future__ import annotations

import math
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx

from sonic.logger import get_logger
from sonic.memory.vector import DenseVectorizer

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


DEFAULT_SCALE_ANCHORS: dict[str, str] = {
    "safe": "safe benign read harmless echo ls cat status dir view inspect check query git status pwd",
    "needs_approval": "needs_approval sensitive privilege sudo chown chmod nmap scan modify install network auth curl wget",
    "destructive": "destructive wipe format delete drop purge kill shutdown rm rf mkfs dd corrupt danger terminate attack",
    "routine": "routine normal standard regular maintenance periodic trivial query background low",
    "important": "important high urgent priority pending deadline immediate action necessary",
    "critical": "critical emergency block outage disaster failure fatal severity immediate escalation blocker",
    "incomplete": "incomplete missing pending error failed blocked unstarted partial none broken",
    "partial": "partial ongoing progress WIP halfway progressing incomplete running",
    "complete": "complete finished done resolved success passed verified 200 OK validated exit 0",
    "redundant": "redundant repeated duplicate static unchanged seen known stale identical repetitive",
    "moderate": "moderate intermediate typical interesting some variation partial novelty",
    "high": "high novel unique new discovery breakthrough unseen extraordinary unprecedented",
    "low": "low minor negligible tiny small zero minimal",
    "medium": "medium average moderate typical standard ordinary",
}


PRESET_QUESTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "triage": {
        "plane": {
            "type": "choice",
            "instructions": "Select the execution plane substrate matching this objective",
            "criteria": {
                "gui": (
                    "gui visual visually screen screenshot desktop window browser click button "
                    "scroll drag ui page website open"
                ),
                "headless": (
                    "headless terminal cli command bash shell script directory mkdir author "
                    "source code file curl netstat ss port ports endpoints background git "
                    "pytest test audit server"
                ),
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Assess operational priority and execution urgency",
            "criteria": ["routine", "important", "critical"],
        },
    },
    "guard": {
        "risk_tier": {
            "type": "score",
            "instructions": "Assess operational safety and command risk tier",
            "criteria": ["safe", "needs_approval", "destructive"],
        },
        "is_dangerous": {
            "type": "noul",
            "instructions": "Determine if command or action carries critical risk of destructive data loss or harm",
            "affirmative_anchor": (
                "destructive wipe rm -rf format drop purge kill shutdown reboot malicious "
                "exploit danger critical rm mkfs dd pkill forkbomb drop table truncate"
            ),
            "negative_anchor": (
                "safe read cat ls echo status inspect harmless query benign test check verify "
                "git status dir pwd find ps"
            ),
        },
    },
    "verification": {
        "empirical_proof": {
            "type": "noul",
            "instructions": "Verify empirical execution receipt and verifiable success outcome",
            "affirmative_anchor": (
                "FILE_VERIFIED verified success confirmed passed 200 OK exit 0 created committed running complete"
            ),
            "negative_anchor": (
                "error failed exception traceback blocked Terminal Ready (0 procs) __sonic_obs_ready__ exit 126"
            ),
        },
        "completeness": {
            "type": "score",
            "instructions": "Evaluate degree of task completion based on receipts",
            "criteria": ["incomplete", "partial", "complete"],
        },
    },
    "curiosity": {
        "novelty": {
            "type": "score",
            "instructions": "Evaluate observation novelty and unexpected informational value",
            "criteria": ["redundant", "moderate", "high"],
        },
        "worth_learning": {
            "type": "noul",
            "instructions": "Evaluate observation information content and enduring learning value",
            "affirmative_anchor": (
                "discovered active service port file version configuration capability novel insight interesting"
            ),
            "negative_anchor": "empty idle waiting nothing unchanged error redundant noise repetitive static",
        },
    },
}


@dataclass
class LayaDecisionResult:
    """Typed output returned by a Laya System-1 forward pass."""

    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)

    def get_choice(self, question_key: str, default: str = "") -> str:
        ans = self.answers.get(question_key, {})
        return str(ans.get("choice") or default)

    def get_noul(self, question_key: str, default: float = 0.5) -> float:
        ans = self.answers.get(question_key, {})
        val = ans.get("noul")
        return float(val) if val is not None else default

    def get_score(self, question_key: str, default: float = 0.5) -> float:
        ans = self.answers.get(question_key, {})
        val = ans.get("score")
        return float(val) if val is not None else default

    def get_label(self, question_key: str, default: str = "") -> str:
        ans = self.answers.get(question_key, {})
        return str(ans.get("label") or ans.get("choice") or default)


def _detect_script_and_model(text: str) -> tuple[str, str]:
    """Detect script family via Unicode character properties (`english` vs `multilingual`)."""
    for ch in text:
        if ch.isalpha() and ord(ch) > 127:
            name = unicodedata.name(ch, "")
            if "DEVANAGARI" in name:
                return "hi", "multilingual"
            return "mul", "multilingual"
    return "en", "english"


def _has_sufficient_ram_gb(min_free_gb: float = 1.5) -> bool:
    """Check if the host has enough free physical memory before loading PyTorch weights."""
    try:
        import psutil  # type: ignore[import-untyped]

        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        return bool(free_gb >= min_free_gb)
    except Exception:
        pass

    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            free_gb = stat.ullAvailPhys / (1024 ** 3)
            return bool(free_gb >= min_free_gb)
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return bool((kb / (1024 ** 2)) >= min_free_gb)
    except Exception:
        pass
    return True


class LayaSystem1Engine:
    """
    System-1 Non-Autoregressive Decision Engine for SONIC.

    Evaluates structured questions (`choice`, `score`, `noul`) over any state.
    """

    PRESET_QUESTIONS = PRESET_QUESTIONS
    DEFAULT_SCALE_ANCHORS = DEFAULT_SCALE_ANCHORS

    def __init__(
        self,
        api_url: str | None = None,
        enable_local_model: bool | None = None,
        preload: bool = False,
        dim: int = 256,
    ):
        self.api_url = (api_url or os.environ.get("LAYA_API_URL") or "").rstrip("/")
        if enable_local_model is None:
            enable_local_model = os.environ.get("SONIC_ENABLE_LOCAL_LAYA", "0") == "1"
        self.enable_local_model = enable_local_model
        self.preload = preload or (os.environ.get("LAYA_PRELOAD", "0") == "1")
        self._vectorizer = DenseVectorizer(dim=dim)
        self._local_router: Any = None
        self._lock = threading.Lock()

    def _get_local_router(self) -> Any:
        """Lazily load `laya.Router` only if enabled and RAM >= 1.5 GB."""
        if not self.enable_local_model:
            return None
        if self._local_router is not None:
            return self._local_router
        with self._lock:
            if self._local_router is not None:
                return self._local_router
            if not _has_sufficient_ram_gb(1.5):
                logger.warning("laya_local_skipped_low_ram", min_free_gb=1.5)
                return None
            try:
                from laya import Router  # type: ignore[import-not-found]

                self._local_router = Router(preload=self.preload)
                logger.info("laya_local_router_initialized", preload=self.preload)
                return self._local_router
            except Exception as exc:
                logger.debug("laya_local_unavailable", error=str(exc))
                return None

    def predict(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        model: str | None = None,
        max_len: int = 1024,
    ) -> LayaDecisionResult:
        """
        Evaluate typed questions (`choice`, `score`, `noul`) over `state` in a single pass.
        """
        state_text = state if isinstance(state, str) else str(state)
        lang_code, routed_model = _detect_script_and_model(state_text)
        chosen_model = model or routed_model

        # 1. Remote HTTP `laya-serve` endpoint (0 MB Laptop RAM)
        if self.api_url:
            try:
                with httpx.Client(timeout=2.5) as client:
                    resp = client.post(
                        f"{self.api_url}/predict",
                        json={
                            "state": state,
                            "questions": questions,
                            "model": chosen_model,
                            "max_len": max_len,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return LayaDecisionResult(
                            answers=data.get("answers", {}),
                            routing=data.get("routing", {"model": chosen_model, "backend": "laya_remote"}),
                        )
            except Exception as exc:
                logger.debug("laya_remote_call_failed", error=str(exc))

        # 2. Local `laya` Python package (if installed & RAM guard passes)
        local_router = self._get_local_router()
        if local_router is not None:
            try:
                raw = local_router.predict(state, questions, model=chosen_model, max_len=max_len)
                return LayaDecisionResult(
                    answers=raw.get("answers", {}),
                    routing=raw.get("routing", {"model": chosen_model, "backend": "laya_local"}),
                )
            except Exception as exc:
                logger.debug("laya_local_predict_failed", error=str(exc))

        # 3. Calibrated Vector + Token Alignment Projection
        state_vec = self._vectorizer.encode(state_text)
        state_tokens = [t.lower() for t in _TOKEN_RE.findall(state_text)]
        answers: dict[str, dict[str, Any]] = {}

        for q_key, q_spec in questions.items():
            q_type = str(q_spec.get("type", "choice")).lower()
            instructions = str(q_spec.get("instructions", ""))
            criteria = q_spec.get("criteria")

            if q_type == "choice" and isinstance(criteria, dict) and criteria:
                probs = self._project_choice_vectors(state_text, state_vec, state_tokens, instructions, criteria)
                best_choice = max(probs.items(), key=lambda kv: kv[1])[0]
                answers[q_key] = {
                    "choice": best_choice,
                    "probability": round(probs[best_choice], 4),
                    "probabilities": {k: round(v, 4) for k, v in probs.items()},
                }
            elif q_type == "score":
                if isinstance(criteria, dict):
                    scale_spec: list[str] | dict[str, str] = criteria
                elif isinstance(criteria, (list, tuple)) and criteria:
                    scale_spec = list(criteria)
                else:
                    scale_spec = ["low", "medium", "high"]
                score_val, label = self._project_scale_vectors(state_text, state_vec, state_tokens, instructions, scale_spec)
                answers[q_key] = {
                    "score": round(score_val, 4),
                    "label": label,
                }
            elif q_type == "noul":
                affirmative_anchor = str(
                    q_spec.get("affirmative_anchor")
                    or f"{instructions} FILE_VERIFIED verified success confirmed passed 200 OK exit 0 created committed"
                )
                negative_anchor = str(
                    q_spec.get("negative_anchor")
                    or "error failed exception traceback blocked Terminal Ready (0 procs) __sonic_obs_ready__ exit 126"
                )
                prob_yes = self._project_noul_vectors(state_text, state_vec, affirmative_anchor, negative_anchor)
                answers[q_key] = {
                    "noul": round(prob_yes, 4),
                    "answer": prob_yes >= 0.5,
                }

        return LayaDecisionResult(
            answers=answers,
            routing={
                "model": chosen_model,
                "language": lang_code,
                "backend": "laya_vector_projection",
            },
        )

    def predict_preset(self, state: str, preset: str = "triage") -> LayaDecisionResult:
        """Evaluate a built-in Laya question preset ('triage', 'guard', 'verification', 'curiosity')."""
        if preset not in PRESET_QUESTIONS:
            raise ValueError(f"Unknown preset '{preset}'. Supported presets: {list(PRESET_QUESTIONS.keys())}")
        return self.predict(state=state, questions=PRESET_QUESTIONS[preset])

    def predict_batch(
        self,
        states: list[str],
        questions: dict[str, dict[str, Any]],
    ) -> list[LayaDecisionResult]:
        """Evaluate questions across a batch of states in System-1 single-pass."""
        if not states:
            return []
        return [self.predict(state=st, questions=questions) for st in states]

    def assess_action_safety(self, command_or_target: str) -> dict[str, Any]:
        """
        Assess operational safety and risk tier of a command or target using the 'guard' preset.

        Returns:
            {"risk_tier": str, "score": float, "is_dangerous_prob": float}
        """
        res = self.predict_preset(command_or_target, preset="guard")
        risk_tier = res.get_label("risk_tier")
        score = res.get_score("risk_tier", default=0.0)
        if not risk_tier:
            if score >= 0.66:
                risk_tier = "destructive"
            elif score >= 0.33:
                risk_tier = "needs_approval"
            else:
                risk_tier = "safe"
        return {
            "risk_tier": risk_tier,
            "score": score,
            "is_dangerous_prob": res.get_noul("is_dangerous", default=0.0),
        }

    def detect_loop_stagnation(self, goal: str, trace_summary: str) -> dict[str, Any]:
        """
        Detect whether an execution loop is stagnant, repeating identical errors, or stuck without progress.

        Returns:
            {"should_replan": bool, "stuck_prob": float}
        """
        state = f"Goal: {goal}\nTrace: {trace_summary}"
        res = self.predict(
            state=state,
            questions={
                "stagnation": {
                    "type": "noul",
                    "instructions": "Determine if execution loop is stagnant, repeating errors, or stuck without progress",
                    "affirmative_anchor": (
                        "stagnant stuck loop repeating error failure identical retry unprogressed "
                        "dead-end same output blocked failed crash loop timeout unchanged"
                    ),
                    "negative_anchor": (
                        "progress advancing success completed novel state resolved next step "
                        "productive output passed verified working moving forward"
                    ),
                }
            },
        )
        stuck_prob = res.get_noul("stagnation", default=0.0)
        return {
            "should_replan": stuck_prob >= 0.5,
            "stuck_prob": stuck_prob,
        }

    def status(self) -> dict[str, Any]:
        """Return clean telemetry status of the Laya System-1 Decision Engine."""
        has_ram = bool(_has_sufficient_ram_gb(1.5))
        if self.api_url:
            backend = "laya_remote"
        elif self.enable_local_model and (self._local_router is not None or has_ram):
            backend = "laya_local"
        else:
            backend = "laya_vector_projection"

        return {
            "backend": backend,
            "api_url_configured": bool(self.api_url),
            "local_model_enabled": bool(self.enable_local_model),
            "sufficient_ram": has_ram,
            "supported_presets": list(PRESET_QUESTIONS.keys()),
        }

    def _project_choice_vectors(
        self,
        state_text: str,
        state_vec: list[float],
        state_tokens: list[str],
        instructions: str,
        criteria: dict[str, str],
    ) -> dict[str, float]:
        """Project state onto criterion descriptions using hybrid token + dense cosine similarity."""
        logits: dict[str, float] = {}
        for key, desc in criteria.items():
            crit_text = f"{key} {desc}"
            crit_vec = self._vectorizer.encode(crit_text)
            cos_sim = DenseVectorizer.cosine_similarity(state_vec, crit_vec)
            crit_tokens = {t.lower() for t in _TOKEN_RE.findall(crit_text)}
            token_hits = sum(1.0 for tok in state_tokens if tok in crit_tokens)
            # GUI dual-plane carries higher weight when visual verification tokens appear
            weight = 2.2 if key.lower() == "gui" and token_hits > 0 else 1.2
            logits[key] = cos_sim * 3.0 + token_hits * weight

        max_logit = max(logits.values()) if logits else 0.0
        exp_scores = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_scores.values()) or 1.0
        return {k: v / total for k, v in exp_scores.items()}

    def _project_scale_vectors(
        self,
        state_text: str,
        state_vec: list[float],
        state_tokens: list[str],
        instructions: str,
        labels: list[str] | dict[str, str],
    ) -> tuple[float, str]:
        """Project state onto ordered scale anchors in [0.0, 1.0]."""
        if isinstance(labels, dict):
            criteria_map = labels
            label_keys = list(labels.keys())
        else:
            label_keys = list(labels)
            criteria_map = {
                lbl: f"{lbl} {self.DEFAULT_SCALE_ANCHORS.get(lbl.lower(), '')} {instructions}".strip()
                for lbl in label_keys
            }
        probs = self._project_choice_vectors(state_text, state_vec, state_tokens, instructions, criteria_map)
        n = max(1, len(label_keys) - 1)
        expected_val = sum((i / n) * probs.get(lbl, 0.0) for i, lbl in enumerate(label_keys))
        best_label = max(probs.items(), key=lambda kv: kv[1])[0]
        return max(0.0, min(1.0, expected_val)), best_label

    def _project_noul_vectors(
        self,
        state_text: str,
        state_vec: list[float],
        affirmative_anchor: str,
        negative_anchor: str,
    ) -> float:
        """Compute calibrated binary probability P(Yes) via bipolar anchor projection."""
        s_lower = state_text.lower()
        pos_tokens = {t.lower() for t in _TOKEN_RE.findall(affirmative_anchor) if len(t) >= 2}
        neg_tokens = {t.lower() for t in _TOKEN_RE.findall(negative_anchor) if len(t) >= 2}
        state_tokens = [t.lower() for t in _TOKEN_RE.findall(s_lower)]

        pos_vec = self._vectorizer.encode(affirmative_anchor)
        neg_vec = self._vectorizer.encode(negative_anchor)
        sim_pos = DenseVectorizer.cosine_similarity(state_vec, pos_vec)
        sim_neg = DenseVectorizer.cosine_similarity(state_vec, neg_vec)

        pos_hits = sum(1.0 for t in state_tokens if t in pos_tokens)
        neg_hits = sum(1.8 for t in state_tokens if t in neg_tokens)
        if ("terminal ready (0 procs)" in s_lower or "__sonic_obs_ready__" in s_lower) and any(
            t in neg_tokens for t in ("terminal", "ready", "procs", "__sonic_obs_ready__")
        ):
            neg_hits += 4.0

        delta = (sim_pos - sim_neg) * 4.0 + (pos_hits - neg_hits) * 0.9
        return 1.0 / (1.0 + math.exp(-delta))

    def decide_execution_plane(self, goal: str, criteria: dict[str, str] | None = None) -> tuple[str, float]:
        """Decide execution plane (`gui` vs `headless`) across English, Hindi, and Hinglish."""
        plane_criteria = criteria or PRESET_QUESTIONS["triage"]["plane"]["criteria"]
        res = self.predict(
            state=goal,
            questions={
                "plane": {
                    "type": "choice",
                    "instructions": "Select the execution plane substrate matching this objective",
                    "criteria": plane_criteria,
                }
            },
        )
        ans = res.answers.get("plane", {})
        return str(ans.get("choice", "gui")), float(ans.get("probability", 0.5))

    def verify_execution_receipt(self, goal: str, evidence: str) -> float:
        """Estimate probability [0.0, 1.0] that `evidence` empirically proves `goal` succeeded."""
        res = self.predict(
            state=evidence,
            questions={
                "empirical_proof": {
                    "type": "noul",
                    "instructions": "Verify empirical execution receipt",
                    "affirmative_anchor": f"{goal} FILE_VERIFIED verified success passed 200 OK exit 0 created committed running",
                    "negative_anchor": "error traceback exception failed Terminal Ready (0 procs) __sonic_obs_ready__ exit 126 blocked",
                }
            },
        )
        return res.get_noul("empirical_proof", default=0.5)

    def evaluate_curiosity_affect(self, observation: str, known_facts: list[str]) -> dict[str, float]:
        """Evaluate semantic novelty and affect shift for `BeingMind` and `CuriosityLoop`."""
        obs_vec = self._vectorizer.encode(observation)
        max_sim = 0.0
        for fact in known_facts:
            fact_vec = self._vectorizer.encode(fact)
            sim = DenseVectorizer.cosine_similarity(obs_vec, fact_vec)
            if sim > max_sim:
                max_sim = sim
        novelty_score = round(max(0.0, min(1.0, 1.0 - max_sim)), 4)
        res = self.predict(
            state=observation,
            questions={
                "worth_learning": {
                    "type": "noul",
                    "instructions": "Evaluate observation information content",
                    "affirmative_anchor": "discovered active service port file version configuration capability novel",
                    "negative_anchor": "empty idle waiting nothing unchanged error",
                }
            },
        )
        return {
            "novelty_score": novelty_score,
            "worth_learning_prob": res.get_noul("worth_learning", 0.5),
        }


_LAYA_SINGLETON: LayaSystem1Engine | None = None
_LAYA_LOCK = threading.Lock()


def get_laya_engine() -> LayaSystem1Engine:
    """Return the process-wide Laya System-1 Decision Engine singleton."""
    global _LAYA_SINGLETON
    if _LAYA_SINGLETON is None:
        with _LAYA_LOCK:
            if _LAYA_SINGLETON is None:
                _LAYA_SINGLETON = LayaSystem1Engine()
    return _LAYA_SINGLETON
