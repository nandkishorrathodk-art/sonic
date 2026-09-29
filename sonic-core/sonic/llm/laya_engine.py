"""
SONIC — Laya System-1 Non-Autoregressive Decision Engine Integration
=====================================================================
Integrates the Laya System-1 decision architecture (https://github.com/NandhaKishorM/laya)
into SONIC's AI-Human Digital Being cognitive core.

Provides single-pass typed judgments (`choice`, `score`, `noul`) for:
  1. Dual-Plane Execution Routing (`gui` vs `headless` vs `dual`)
  2. Empirical Goal Verification (`noul` receipt proof probability)
  3. BeingMind Curiosity & Novelty Affect Evaluation (`score` & `noul`)
  4. Semantic Pre-Action Safety Triage (`safe`, `needs_approval`, `destructive`)
The local lexical fallback supports English typed decisions and Devanagari
Hindi execution-plane routing.

Hardware & RAM Safety Guard:
  - Never preloads all 3 PyTorch checkpoints unless `LAYA_PRELOAD=1`.
  - Supports remote `laya-serve` HTTP API (`LAYA_API_URL`) with 0 MB host RAM overhead.
  - Supports local `laya.Router(preload=False)` when `SONIC_ENABLE_LOCAL_LAYA=1` and
    free system RAM >= 1.5 GB.
  - Includes a zero-RAM calibrated multilingual scoring engine fallback so typed
    decisions always return valid `choice` / `score` / `noul` probability distributions.
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

logger = get_logger(__name__)

_DANGEROUS_ACTION_MARKERS = (
    "rm -rf", "drop", "delete", "format", "truncate", "destroy", "wipe",
    "kill", "mkfs", "dd if=", "dd of=",
)


def _marker_is_negated(text: str, marker: str) -> bool:
    marker_pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in marker.split()) + r"(?!\w)"
    return bool(
        re.search(
            rf"\b(?:do\s+not|don't|dont|never|avoid|without|not|0)\b"
            rf"(?:[ \t]+\w+){{0,5}}[ \t]+{marker_pattern}",
            text,
            re.IGNORECASE,
        )
    )


def _contains_unnegated_marker(text: str, marker: str) -> bool:
    marker_pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in marker.split()) + r"(?!\w)"
    return any(
        not _marker_is_negated(text[max(0, match.start() - 48):match.end()], marker)
        for match in re.finditer(marker_pattern, text, re.IGNORECASE)
    )


def _has_unnegated_dangerous_action(text: str) -> bool:
    return any(_contains_unnegated_marker(text, marker) for marker in _DANGEROUS_ACTION_MARKERS)


def _has_explicit_failure(text: str) -> bool:
    if re.search(
        r"\b(?:(?:do|does|did)\s+not|don't|doesn't|didn't|never)\s+"
        r"(?:pass|succeed|complete|verify)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\bunsuccessful(?:ly)?\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bexit(?:_code|\s+code)?\s*[:= ]+\s*(?!0\b)\d+\b", text, re.IGNORECASE):
        return True
    if re.search(r"\b[1-9]\d*\s+(?:tests?\s+)?failed\b", text, re.IGNORECASE):
        return True
    return any(
        _contains_unnegated_marker(text, marker)
        for marker in (
            "traceback", "assertionerror", "command not found",
            "connection refused", "timed out", "blocked",
            "disk verification failed", "failed", "failure",
            "exception", "error", "syntax error", "syntaxerror",
            "indentationerror", "taberror", "invalid syntax",
            "terminal ready (0 procs)", "__sonic_obs_ready__",
            "exit 126", "unsaved",
        )
    )


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


def _detect_script_and_model(text: str) -> tuple[str, str]:
    """Return the detected script language and the default checkpoint."""
    if any("\u0900" <= char <= "\u097f" for char in text):
        return "hi-Deva", "english"
    return "en", "english"


def _tokenize(text: str) -> list[str]:
    """Split Unicode words while keeping combining marks attached to letters."""
    tokens: list[str] = []
    current: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        category = unicodedata.category(char)[0]
        if category in {"L", "N"} or (category == "M" and current):
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _has_sufficient_ram_gb(min_free_gb: float = 1.5) -> bool:
    """Check if the host has enough free physical memory before loading PyTorch weights."""
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
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                raise OSError("GlobalMemoryStatusEx failed")
            free_gb = stat.ullAvailPhys / (1024 ** 3)
            return free_gb >= min_free_gb
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return (kb / (1024 ** 2)) >= min_free_gb
        raise OSError("No supported physical-memory source is available")
    except Exception as exc:
        logger.warning("laya_ram_check_failed_closed", error=str(exc))
        return False


class LayaSystem1Engine:
    """
    System-1 Non-Autoregressive Decision Engine for SONIC.

    Evaluates structured questions (`choice`, `score`, `noul`) over any state
    (operator prompt, terminal observation, desktop window state, or execution trace).
    """

    def __init__(
        self,
        api_url: str | None = None,
        enable_local_model: bool | None = None,
        preload: bool = False,
    ):
        self.api_url = (api_url or os.environ.get("LAYA_API_URL") or "").rstrip("/")
        if enable_local_model is None:
            enable_local_model = os.environ.get("SONIC_ENABLE_LOCAL_LAYA", "0") == "1"
        self.enable_local_model = enable_local_model
        self.preload = preload or (os.environ.get("LAYA_PRELOAD", "0") == "1")
        self._local_router: Any = None
        self._lock = threading.Lock()
        self._active_backend = "laya_vector_projection"

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
        Matches the `laya.Router.predict(state, questions)` signature and return contract.
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
                        result = self._validate_backend_result(
                            data, questions, chosen_model, "laya_remote"
                        )
                        if result is not None:
                            self._active_backend = "laya_remote"
                            return result
                        logger.warning("laya_remote_invalid_response")
                    else:
                        logger.warning(
                            "laya_remote_non_success_status",
                            status_code=resp.status_code,
                        )
            except Exception as exc:
                logger.warning("laya_remote_call_failed", error=str(exc))

        # 2. Local `laya` Python package (if installed & RAM guard passes)
        local_router = self._get_local_router()
        if local_router is not None:
            try:
                raw = local_router.predict(state, questions, model=chosen_model, max_len=max_len)
                result = self._validate_backend_result(
                    raw, questions, chosen_model, "laya_local"
                )
                if result is not None:
                    self._active_backend = "laya_local"
                    return result
                logger.warning("laya_local_invalid_response")
            except Exception as exc:
                logger.warning("laya_local_predict_failed", error=str(exc))

        # 3. Zero-RAM calibrated scoring fallback.
        answers: dict[str, dict[str, Any]] = {}
        for q_key, q_spec in questions.items():
            q_type = str(q_spec.get("type", "choice")).lower()
            instructions = str(q_spec.get("instructions", ""))
            criteria = q_spec.get("criteria")

            if q_type == "choice" and isinstance(criteria, dict) and criteria:
                risk_levels = {key.lower() for key in criteria}
                negated_only = (
                    {"l0", "l1", "l2"}.issubset(risk_levels)
                    and not _has_unnegated_dangerous_action(state_text)
                    and any(
                        _marker_is_negated(state_text, marker)
                        for marker in _DANGEROUS_ACTION_MARKERS
                    )
                )
                if negated_only:
                    best_choice = next(key for key in criteria if key.lower() == "l0")
                    probs = {key: float(key == best_choice) for key in criteria}
                else:
                    probs = self._score_choice_criteria(state_text, instructions, criteria)
                    best_choice = max(probs.items(), key=lambda kv: kv[1])[0]
                answers[q_key] = {
                    "choice": best_choice,
                    "probability": round(probs[best_choice], 4),
                    "probabilities": {k: round(v, 4) for k, v in probs.items()},
                }
            elif q_type == "score":
                labels = list(criteria) if isinstance(criteria, (list, tuple)) and criteria else ["low", "medium", "high"]
                score_val, label = self._score_scale(state_text, instructions, labels)
                answers[q_key] = {
                    "score": round(score_val, 4),
                    "label": label,
                }
            elif q_type == "noul":
                aff = str(q_spec.get("affirmative_anchor", ""))
                neg = str(q_spec.get("negative_anchor", ""))
                prob_yes = self._score_noul(state_text, instructions, affirmative_anchor=aff, negative_anchor=neg)
                answers[q_key] = {
                    "noul": round(prob_yes, 4),
                    "answer": prob_yes >= 0.5,
                }

        self._active_backend = "laya_vector_projection"
        return LayaDecisionResult(
            answers=answers,
            routing={
                "model": chosen_model,
                "language": lang_code,
                "backend": "laya_vector_projection",
            },
        )

    @staticmethod
    def _validate_backend_result(
        raw: Any,
        questions: dict[str, dict[str, Any]],
        model: str,
        backend: str,
    ) -> LayaDecisionResult | None:
        """Reject incomplete or malformed model results so callers get fallback scoring."""
        if isinstance(raw, LayaDecisionResult):
            answers = raw.answers
            routing = raw.routing
        elif isinstance(raw, dict):
            answers = raw.get("answers")
            routing = raw.get("routing", {})
        else:
            return None

        if not isinstance(answers, dict) or not isinstance(routing, dict):
            return None

        normalized: dict[str, dict[str, Any]] = {}
        for key, spec in questions.items():
            answer = answers.get(key)
            if not isinstance(answer, dict):
                return None
            question_type = str(spec.get("type", "choice")).lower()
            if question_type == "choice":
                criteria = spec.get("criteria")
                choice = answer.get("choice")
                probability = answer.get("probability")
                if not isinstance(criteria, dict) or choice not in criteria:
                    return None
                if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                    return None
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    return None
            elif question_type in {"score", "noul"}:
                field_name = "score" if question_type == "score" else "noul"
                value = answer.get(field_name)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return None
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    return None
            else:
                return None
            normalized[key] = answer

        result_routing = dict(routing)
        result_routing.setdefault("model", model)
        result_routing.setdefault("backend", backend)
        return LayaDecisionResult(answers=normalized, routing=result_routing)

    def _score_choice_criteria(
        self,
        state_text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> dict[str, float]:
        """Compute calibrated Softmax distribution over candidate choices."""
        state_tokens = [t.lower() for t in _tokenize(state_text)]
        state_lower = state_text.lower()

        plane_concepts: dict[str, set[str]] = {
            "gui": {
                "gui", "visual", "visually", "screen", "screenshot", "desktop", "window",
                "browser", "click", "button", "scroll", "drag", "ui", "page", "open",
                "website", "login", "signup", "profile", "view", "display", "render",
                "दृश्य", "स्क्रीन", "डेस्कटॉप", "विंडो", "खिड़की", "ब्राउज़र",
                "अनुप्रयोग", "ऐप", "खोलें", "खोलना", "क्लिक", "दबाएं", "बटन",
                "स्क्रॉल", "पृष्ठ", "वेबसाइट", "सेटिंग", "चित्र", "तस्वीर",
                "डिजाइन", "डिज़ाइन", "संपादित", "फॉर्म", "लॉगिन", "मेनू",
                "देखें", "देखना", "टाइप",
            },
            "headless": {
                "headless", "terminal", "cli", "command", "bash", "shell", "script",
                "directory", "mkdir", "file", "code", "source", "git", "commit", "push",
                "pytest", "test", "server", "port", "ports", "curl", "netstat", "ss",
                "endpoints", "background", "audit", "compile", "run", "python",
                "टर्मिनल", "कमांड", "आदेश", "शेल", "स्क्रिप्ट", "फ़ाइल", "फाइल",
                "कोड", "स्रोत", "जिट", "परीक्षण", "टेस्ट", "जाँच", "चलाएं",
                "चलाना", "चलाने", "बनाएं", "बनाना", "लिखें", "लिखना", "सर्वर",
                "पोर्ट", "पाइथन", "पायथन", "निर्देशिका", "फ़ोल्डर", "फोल्डर",
                "प्रोग्राम", "बिल्ड", "कम्पाइल", "प्रोजेक्ट",
            },
        }

        logits: dict[str, float] = {}
        for key, desc in criteria.items():
            key_lower = key.lower()
            desc_tokens = {t.lower() for t in _tokenize(f"{key} {desc}")}
            expanded = desc_tokens | plane_concepts.get(key_lower, set())

            overlap = sum(1.0 for tok in state_tokens if tok in expanded)
            exact_phrase_bonus = sum(
                1.5 for phrase in expanded if len(phrase) > 3 and phrase in state_lower
            )
            if key_lower == "headless" and ("/home/" in state_lower or ".py" in state_lower):
                exact_phrase_bonus += 2.5

            logits[key] = 0.5 + overlap * 1.2 + exact_phrase_bonus

        max_logit = max(logits.values()) if logits else 0.0
        exp_scores = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_scores.values()) or 1.0
        return {k: v / total for k, v in exp_scores.items()}

    def _score_scale(
        self,
        state_text: str,
        instructions: str,
        labels: list[str],
    ) -> tuple[float, str]:
        """Compute a calibrated [0.0, 1.0] ordinal score across ordered criteria labels."""
        criteria_map = {lbl: lbl for lbl in labels}
        probs = self._score_choice_criteria(state_text, instructions, criteria_map)
        n = max(1, len(labels) - 1)
        expected_val = sum((i / n) * probs.get(lbl, 0.0) for i, lbl in enumerate(labels))
        best_label = max(probs.items(), key=lambda kv: kv[1])[0]
        return max(0.0, min(1.0, expected_val)), best_label

    PRESET_QUESTIONS: dict[str, dict[str, Any]] = {
        "triage": {
            "plane": {
                "type": "choice",
                "instructions": "Which execution plane substrate should execute this task?",
                "criteria": {
                    "gui": "visual desktop application browser navigation clicking buttons scrolling screenshot UI window",
                    "headless": "terminal command bash script file creation source code editing git commit pytest ports CLI",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "Task urgency and execution priority",
                "criteria": ["low", "medium", "high"],
            },
        },
        "plane": {
            "plane": {
                "type": "choice",
                "instructions": "Which execution plane substrate should execute this task?",
                "criteria": {
                    "gui": "visual desktop application browser navigation clicking buttons scrolling screenshot UI window",
                    "headless": "terminal command bash script file creation source code editing git commit pytest ports CLI",
                },
            },
        },
        "guard": {
            "is_dangerous": {
                "type": "noul",
                "instructions": "Is this action dangerous, destructive, or violating the safety envelope?",
                "affirmative_anchor": "rm -rf drop delete format truncate destroy wipe kill bypass dangerous exploit attack",
                "negative_anchor": "ls pwd git status echo read inspect cat test compile check verify status ok safe",
            },
            "risk_tier": {
                "type": "choice",
                "instructions": "Classify risk tier",
                "criteria": {
                    "l0": "safe read only inspect check status echo cat ls git status",
                    "l1": "standard write modification build execute test script python",
                    "l2": "destructive rm -rf drop format dangerous delete destroy kill",
                },
            },
        },
        "verification": {
            "empirical_proof": {
                "type": "noul",
                "instructions": "Does this execution output provide concrete empirical proof that the goal succeeded without errors?",
                "affirmative_anchor": "FILE_VERIFIED verified success passed 200 OK exit 0 created committed",
                "negative_anchor": "error traceback exception failed Terminal Ready (0 procs) __sonic_obs_ready__ exit 126 blocked syntaxerror syntax error *untitled unsaved disk verification failed",
            }
        },
        "curiosity": {
            "novelty": {
                "type": "score",
                "instructions": "How novel and informative is this observation compared to known facts?",
                "criteria": ["redundant duplicate", "partially new", "highly novel discovery"],
            },
            "worth_learning": {
                "type": "noul",
                "instructions": "Does this observation contain a concrete, verifiable new fact about the workstation or project?",
                "affirmative_anchor": "discovered active service port file version configuration capability novel",
                "negative_anchor": "empty idle waiting nothing unchanged error",
            },
        },
    }
    PRESET_QUESTIONS["execution_plane"] = PRESET_QUESTIONS["plane"]
    PRESET_QUESTIONS["receipt"] = PRESET_QUESTIONS["verification"]
    PRESET_QUESTIONS["verify_receipt"] = PRESET_QUESTIONS["verification"]
    PRESET_QUESTIONS["empirical_proof"] = PRESET_QUESTIONS["verification"]
    PRESET_QUESTIONS["safety"] = PRESET_QUESTIONS["guard"]
    PRESET_QUESTIONS["risk"] = PRESET_QUESTIONS["guard"]
    PRESET_QUESTIONS["affect"] = PRESET_QUESTIONS["curiosity"]
    PRESET_QUESTIONS["novelty"] = PRESET_QUESTIONS["curiosity"]
    PRESET_QUESTIONS["worth_learning"] = PRESET_QUESTIONS["curiosity"]

    def status(self) -> dict[str, Any]:
        """Return runtime diagnostic telemetry for the Laya System-1 engine."""
        ram_ok = _has_sufficient_ram_gb(1.5)
        if self.api_url:
            configured_backend = "laya_remote"
        elif self.enable_local_model and ram_ok:
            configured_backend = "laya_local"
        else:
            configured_backend = "laya_vector_projection"

        return {
            "status": "ready",
            "backend": self._active_backend,
            "configured_backend": configured_backend,
            "api_url": self.api_url or None,
            "api_url_configured": bool(self.api_url),
            "local_model_enabled": bool(self.enable_local_model),
            "sufficient_ram": ram_ok,
            "has_ram_headroom": ram_ok,
            "ram_check": {"sufficient": ram_ok, "min_free_gb": 1.5},
            "language_support": ["en", "hi-Deva"],
            "languages": ["en", "hi-Deva"],
            "language_capabilities": {
                "en": ["all typed fallback decisions"],
                "hi-Deva": ["execution-plane routing"],
            },
            "presets": ["triage", "guard", "verification", "curiosity"],
            "supported_presets": ["triage", "guard", "verification", "curiosity"],
        }

    def predict_preset(
        self,
        state: str | dict[str, Any],
        preset: str = "triage",
        model: str | None = None,
    ) -> LayaDecisionResult:
        """
        Evaluate built-in presets (`triage`, `guard`, `verification`, `curiosity`).
        """
        p_lower = preset.strip().lower()
        if p_lower not in self.PRESET_QUESTIONS:
            raise ValueError(f"Unknown preset: {preset}")
        questions = self.PRESET_QUESTIONS[p_lower]
        return self.predict(state=state, questions=questions, model=model)

    def _score_noul(
        self,
        state_text: str,
        instructions: str,
        affirmative_anchor: str = "",
        negative_anchor: str = "",
    ) -> float:
        """Compute calibrated binary `noul` probability P(Yes | state, instructions)."""
        s_lower = state_text.lower()
        inst_lower = instructions.lower()

        if any(w in inst_lower for w in ("dangerous", "harmful", "destructive", "risk")):
            if any(_marker_is_negated(s_lower, marker) for marker in _DANGEROUS_ACTION_MARKERS):
                if not _has_unnegated_dangerous_action(s_lower):
                    return 0.05

        if any(w in inst_lower for w in ("verify", "empirical proof", "succeeded", "proof")):
            if _has_explicit_failure(s_lower):
                return 0.08
            proof_markers = (
                "file_verified", "verified", "success", "succeeded", "passed",
                "created", "committed", "exit 0", "200 ok", "completed",
                "process is running", "authenticated landing state observed",
                "positive service state observed",
            )
            proof_hits = sum(
                1 for marker in proof_markers
                if _contains_unnegated_marker(s_lower, marker)
            )
            return min(0.96, 0.70 + 0.08 * (proof_hits - 1)) if proof_hits else 0.10

        # 1. Anchor-based score if affirmative or negative anchor provided
        if affirmative_anchor or negative_anchor:
            state_tokens = set(_tokenize(s_lower))
            aff_tokens = set(_tokenize(affirmative_anchor.lower()))
            neg_tokens = set(_tokenize(negative_anchor.lower()))

            aff_hits = len(state_tokens & aff_tokens)
            neg_hits = len(state_tokens & neg_tokens)

            # Substring checks for multi-token markers
            for phrase in (
                "terminal ready (0 procs)",
                "__sonic_obs_ready__",
                "exit 126",
                "traceback",
                "assertionerror",
                "syntax error",
                "syntaxerror",
                "indentationerror",
                "taberror",
                "invalid syntax",
                "*untitled",
                "unsaved",
                "disk verification failed",
            ):
                if phrase in negative_anchor.lower() and phrase in s_lower:
                    neg_hits += 5
            for phrase in ("rm -rf", "drop table", "mkfs"):
                if phrase in affirmative_anchor.lower() and phrase in s_lower:
                    aff_hits += 5
            for phrase in ("file_verified", "200 ok", "exit 0", "passed"):
                if phrase in affirmative_anchor.lower() and phrase in s_lower:
                    aff_hits += 5

            if aff_hits > 0 or neg_hits > 0:
                diff = aff_hits - neg_hits
                prob = 1.0 / (1.0 + math.exp(-1.4 * diff))
                return max(0.05, min(0.95, prob))

        # 2. Instruction-based fallback heuristics
        if any(w in inst_lower for w in ("dangerous", "harmful", "destructive", "risk")):
            if _has_unnegated_dangerous_action(s_lower):
                return 0.92
            return 0.15

        if any(w in inst_lower for w in ("verify", "empirical proof", "succeeded", "proof")):
            return 0.10

        if any(w in inst_lower for w in ("stuck", "loop", "repeating")):
            err_hits = sum(s_lower.count(w) for w in ("error", "failed", "blocked", "timeout"))
            return min(0.95, 0.25 + 0.22 * err_hits)

        return 0.50

    # ------------------------------------------------------------------
    # High-Level SONIC AI-Human Being Helpers
    # ------------------------------------------------------------------

    def decide_execution_plane(self, goal: str) -> tuple[str, float]:
        """
        Use System-1 `choice` judgment to decide between `"gui"` and `"headless"`
        planes across English, Hindi, and Hinglish goals.
        """
        res = self.predict(
            state=goal,
            questions={
                "plane": {
                    "type": "choice",
                    "instructions": "Which execution plane should execute this workstation task?",
                    "criteria": {
                        "gui": "visual desktop application, browser navigation, clicking buttons, scrolling, screenshot, UI window",
                        "headless": "terminal command, bash script, file creation, source code editing, git commit, pytest, ports, CLI",
                    },
                }
            },
        )
        ans = res.answers.get("plane", {})
        raw_choice = ans.get("choice")
        raw_prob = ans.get("probability")
        if (
            raw_choice not in {"gui", "headless"}
            or not isinstance(raw_prob, (int, float))
            or isinstance(raw_prob, bool)
            or not math.isfinite(raw_prob)
            or not 0.0 <= raw_prob <= 1.0
        ):
            logger.warning("laya_plane_decision_invalid")
            return "gui", 0.5
        choice = str(raw_choice)
        prob = float(raw_prob)
        return choice, prob

    def verify_execution_receipt(self, goal: str, evidence: str) -> float:
        """
        Use System-1 `noul` judgment to estimate probability [0.0, 1.0] that `evidence`
        is genuine empirical proof for `goal` (Zero Success-by-Decree).
        """
        g_lower = goal.lower()
        ev_lower = evidence.lower()

        # 1. Reject if terminal has syntax errors
        syntax_err_markers = ("syntax error", "syntaxerror", "indentationerror", "taberror", "invalid syntax", "parse error")
        if any(sm in ev_lower for sm in syntax_err_markers):
            if not any(neg in ev_lower for neg in ("no syntax error", "no syntax errors", "clean (no syntax", "terminal: clean")):
                return 0.05

        # 2. Reject if editor titles have unsaved asterisks (*untitled) or unsaved changes
        if re.search(r"(?:^|\s)\*|\*untitled|\[modified\]|\bunsaved\b", evidence, re.IGNORECASE):
            if not any(neg in ev_lower for neg in ("no unsaved", "clean (no unsaved", "editor titles: clean")):
                return 0.08

        if _has_explicit_failure(ev_lower):
            return 0.05

        proof_markers = (
            "file_verified", "actual disk verification: verified",
            "verified file(s) on disk", "verified on disk",
            "passed", "success", "succeeded", "exit 0", "200 ok",
            "completed", "process is running",
            "authenticated landing state observed",
            "positive service state observed",
        )
        if not any(_contains_unnegated_marker(ev_lower, marker) for marker in proof_markers):
            return 0.10

        # 3. For goals that require writing/saving files, creating services, or producing artifacts:
        creates_or_writes = any(
            w in g_lower
            for w in (
                "write", "save", "create", "author", "produce", "generate",
                "build", "compose", "draft", "export", "dump", "edit", "touch",
            )
        )
        requires_artifact = (
            creates_or_writes
            and any(
                k in g_lower
                for k in ("file", "service", "artifact", "script", "report", "code", "server")
            )
        ) or bool(re.search(r'[\w\-./\\]+\.(?:txt|md|py|json|sh|csv|html|log|yaml|yml|toml)', goal, re.IGNORECASE))

        if requires_artifact:
            # Check for actual disk verification
            has_disk_verif = any(
                dv in ev_lower
                for dv in (
                    "file_verified", "actual disk verification: verified",
                    "verified file(s) on disk", "verified on disk",
                )
            ) and "disk verification failed" not in ev_lower

            # If evidence only shows running processes without actual disk verification, NEVER treat as proof!
            is_only_running = any(rp in ev_lower for rp in ("running processes:", "running:", "process is running")) and not has_disk_verif
            if is_only_running or not has_disk_verif:
                return 0.10

        res = self.predict(
            state=f"Goal: {goal}\nEvidence: {evidence}",
            questions={
                "empirical_proof": {
                    "type": "noul",
                    "instructions": "Does this execution output provide concrete empirical proof that the goal succeeded without errors?",
                    "affirmative_anchor": "FILE_VERIFIED verified success passed 200 OK exit 0 created committed",
                    "negative_anchor": "error traceback exception failed Terminal Ready (0 procs) __sonic_obs_ready__ exit 126 blocked syntaxerror syntax error *untitled unsaved disk verification failed",
                }
            },
        )
        return res.get_noul("empirical_proof", default=0.5)

    def evaluate_curiosity_affect(self, observation: str, known_facts: list[str]) -> dict[str, float]:
        """
        Evaluate semantic novelty and affect shift for `BeingMind` and `CuriosityLoop`.
        """
        obs_clean = observation.strip().lower()
        known_clean = [f.strip().lower() for f in known_facts]

        # Duplicate match check: if observation is already in known_facts
        if any(obs_clean == k or (len(k) > 10 and k in obs_clean) for k in known_clean):
            return {
                "novelty_score": 0.10,
                "worth_learning_prob": 0.10,
            }

        known_preview = " | ".join(known_facts[-5:]) if known_facts else "none"
        res = self.predict(
            state=f"Observation: {observation}\nKnown: {known_preview}",
            questions={
                "novelty": {
                    "type": "score",
                    "instructions": "How novel and informative is this observation compared to known facts?",
                    "criteria": ["redundant duplicate", "partially new", "highly novel discovery"],
                },
                "worth_learning": {
                    "type": "noul",
                    "instructions": "Does this observation contain a concrete, verifiable new fact about the workstation or project?",
                    "affirmative_anchor": "discovered active service port file version configuration capability novel",
                    "negative_anchor": "empty idle waiting nothing unchanged error",
                },
            },
        )
        return {
            "novelty_score": res.get_score("novelty", 0.5),
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
