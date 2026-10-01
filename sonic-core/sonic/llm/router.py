"""
SONIC-REDA — Model Router
============================
Routes LLM requests to the right provider based on:
    - Task type (planning, reasoning, fast_recon, coding, etc.)
    - Config from models.yaml
    - Fallback chains if primary provider fails

The router manages multiple CustomLLMProvider instances.
Team configures providers in models.yaml or via API.

Usage:
    router = ModelRouter.from_config("configs/models.yaml")

    # Route by task type
    response = await router.complete(request, task_type="reasoning")

    # Use specific provider
    response = await router.complete(request, provider_name="grok")

    # Stream
    async for chunk in router.stream(request, task_type="coding"):
        print(chunk.content, end="")
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from sonic.logger import get_logger

logger = get_logger(__name__)

from sonic.llm.providers.custom import CustomLLMProvider
from sonic.llm.schemas import (
    CostRecord,
    LLMChunk,
    LLMRequest,
    LLMResponse,
    SpeedTier,
)

REFUSAL_PHRASES = (
    "can't help with that",
    "cannot help with that",
    "cannot assist",
    "can't assist",
    "unethical",
    "against safety policies",
    "against safety policy",
    "as an ai",
    "i am unable to fulfill",
)


class LLMBudgetExceededError(RuntimeError):
    """Raised when a conservative request-cost estimate exceeds a configured cap."""

    def __init__(
        self,
        budget: str,
        estimate_usd: float,
        limit_usd: float,
        engagement_id: str | None = None,
    ) -> None:
        self.budget = budget
        self.estimate_usd = estimate_usd
        self.limit_usd = limit_usd
        self.engagement_id = engagement_id
        super().__init__(
            f"LLM {budget} budget would be exceeded: estimated "
            f"${estimate_usd:.6f}, limit ${limit_usd:.6f}"
            + (f" (engagement {engagement_id})" if engagement_id else "")
        )


class LLMBudgetEstimateUnavailableError(RuntimeError):
    """Raised when configured limits cannot be checked with known model prices."""


def is_refusal_response(response: LLMResponse) -> bool:
    """Detect if an LLM returned a safety refusal rather than an actionable answer."""
    content = (response.content or "").lower()
    reasoning = (response.reasoning_content or "").lower()
    combined = f"{content} {reasoning}"
    has_refusal = any(p in combined for p in REFUSAL_PHRASES)
    has_action = bool(re.search(r"\baction\s*:\s*[a-z_]+", combined))
    return has_refusal and not has_action


class ModelRouter:
    """
    Central router that manages multiple LLM providers and routes requests.

    Providers are CustomLLMProvider instances configured with:
        - name, base_url, api_key, default_model

    Routing rules map task types to preferred providers with fallback chains.
    """

    def __init__(self):
        self.providers: dict[str, CustomLLMProvider] = {}
        self.routing_rules: dict[str, dict] = {}
        self.fallback_chain: list[str] = []
        self.default_provider: str = ""
        self.default_model: str = ""
        self.cost_records: list[CostRecord] = []

        # Cost limits
        # No budget block means legacy behavior: limits are disabled until an
        # operator explicitly configures them in models.yaml.
        self.max_per_request_usd: float | None = None
        self.max_per_engagement_usd: float | None = None
        self.warn_at_percent: float | None = None
        self._budget_lock = asyncio.Lock()
        self._engagement_reservations: dict[str, Decimal] = {}
        self._engagement_uncertain_costs: dict[str, Decimal] = {}
        self._warned_engagements: set[str] = set()
        self.budget_warnings: list[dict[str, Any]] = []
        self.budget_overruns: list[dict[str, Any]] = []
        self.incomplete_image_budget_requests: list[dict[str, Any]] = []
        self.unaccounted_response_costs: list[dict[str, Any]] = []

    # ============================================
    # Setup
    # ============================================

    def add_provider(self, provider: CustomLLMProvider) -> None:
        """Register a provider with the router."""
        self.providers[provider.name] = provider
        logger.info(
            "provider_registered",
            name=provider.name,
            base_url=provider.base_url,
            model=provider.default_model,
        )

    def add_provider_from_config(
        self,
        name: str,
        base_url: str,
        api_key: str,
        default_model: str,
        **kwargs: Any,
    ) -> CustomLLMProvider:
        """Create and register a provider from config values."""
        provider = CustomLLMProvider(
            name=name,
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
            **kwargs,
        )
        self.add_provider(provider)
        return provider

    @classmethod
    def from_config(cls, config_path: str | Path | None = None, env_vars: dict[str, str] | None = None) -> ModelRouter:
        """
        Create a fully configured ModelRouter from a YAML config file.

        Args:
            config_path: Path to models.yaml (defaults to configs/models.yaml).
            env_vars: Dict of environment variables (for API keys).
                      If None, reads from os.environ.
        """
        if config_path is None:
            from sonic.config import PROJECT_ROOT
            config_path = PROJECT_ROOT / "configs" / "models.yaml"
        import os
        env = dict(os.environ)
        try:
            from sonic.config import get_settings
            settings = get_settings()
            for k, v in settings.model_dump().items():
                uk = k.upper()
                if uk not in env and v:
                    env[uk] = str(v)
        except Exception:
            pass
        if env_vars:
            env.update(env_vars)

        config_path = Path(config_path)
        if not config_path.exists() and not config_path.is_absolute():
            repo_config = Path(__file__).resolve().parents[3] / config_path
            if repo_config.exists():
                config_path = repo_config
        if not config_path.exists():
            logger.warning("config_not_found", path=str(config_path))
            return cls()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        router = cls()
        router.default_provider = (
            env.get("DEFAULT_PROVIDER")
            or env.get("DEFAULT_LLM_PROVIDER")
            or env.get("LLM_PROVIDER")
            or config.get("default_provider", "")
        )
        router.default_model = (
            env.get("DEFAULT_MODEL")
            or env.get("LLM_MODEL")
            or config.get("default_model", "")
        )
        router.fallback_chain = config.get("global_fallback_chain", [])

        # Load cost limits
        cost_limits = config.get("cost_limits") or {}
        router.max_per_request_usd = cost_limits.get("max_per_request_usd")
        router.max_per_engagement_usd = cost_limits.get("max_per_engagement_usd")
        router.warn_at_percent = cost_limits.get("warn_at_percent")
        for name in ("max_per_request_usd", "max_per_engagement_usd"):
            amount = getattr(router, name)
            if amount is not None and (not math.isfinite(float(amount)) or float(amount) < 0):
                raise ValueError(f"cost_limits.{name} must be a finite non-negative amount")
        if router.warn_at_percent is not None:
            percent = float(router.warn_at_percent)
            if not math.isfinite(percent) or not 0 <= percent <= 100:
                raise ValueError("cost_limits.warn_at_percent must be between 0 and 100")

        # Register providers from config
        providers_config = config.get("providers", {})
        for provider_name, pconfig in providers_config.items():
            # Resolve API key from env var
            api_key_env = pconfig.get("api_key_env", "")
            base_url_env = pconfig.get("base_url_env", "")

            api_key = env.get(api_key_env, "") if api_key_env else ""
            base_url = env.get(base_url_env, pconfig.get("base_url", "")) if base_url_env else pconfig.get("base_url", "")

            if not base_url:
                logger.debug("provider_skipped_no_url", name=provider_name)
                continue

            # Get first model as default, rest as fallback chain
            models = pconfig.get("models", [])
            if not models:
                continue

            first_model = models[0]
            # Remaining models serve as model-level fallback (EOL/not-found recovery)
            fallback_model_ids = [m["id"] for m in models[1:]]
            speed_map = {"fast": SpeedTier.FAST, "medium": SpeedTier.MEDIUM, "slow": SpeedTier.SLOW}

            # Check if default model for this provider is overridden in env or config
            env_override_model = (
                env.get(f"{provider_name.upper()}_MODEL")
                or (env.get("DEFAULT_MODEL") or env.get("LLM_MODEL") if router.default_provider == provider_name else None)
            )
            config_default_for_provider = router.default_model if router.default_provider == provider_name and router.default_model else None
            model_id = env_override_model or config_default_for_provider or first_model["id"]

            kwargs = {}
            kwargs["model_pricing"] = {
                model["id"]: {
                    "cost_per_1k_input": model["cost_per_1k_input"],
                    "cost_per_1k_output": model["cost_per_1k_output"],
                }
                for model in models
                if "cost_per_1k_input" in model and "cost_per_1k_output" in model
            }
            # Groq on-demand accounts enforce a hard output-tokens-per-minute
            # limit (OTPM=1000). Any request declaring more is rejected with 429
            # before generation starts, regardless of actual output length.
            # Clamp everything going to Groq so the primary path actually works.
            if provider_name == "groq" or "api.groq.com" in base_url:
                kwargs["max_tokens_cap"] = pconfig.get("max_tokens_cap", 900)

            router.add_provider_from_config(
                name=provider_name,
                base_url=base_url,
                api_key=api_key,
                default_model=model_id,
                cost_per_1k_input=first_model.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=first_model.get("cost_per_1k_output", 0.0),
                speed_tier=speed_map.get(first_model.get("speed_tier", "medium"), SpeedTier.MEDIUM),
                capabilities=first_model.get("capabilities", []),
                fallback_models=fallback_model_ids,
                **kwargs,
            )

        # Load routing rules
        rules_config = config.get("routing_rules", {})
        for task_type, rule in rules_config.items():
            router.routing_rules[task_type] = {
                "provider": rule.get("provider", router.default_provider),
                "model": rule.get("model"),
                "fallback": rule.get("fallback", []),
                "description": rule.get("description", ""),
            }

        logger.info(
            "router_loaded",
            providers=list(router.providers.keys()),
            rules=list(router.routing_rules.keys()),
        )
        return router

    # ============================================
    # Routing Logic
    # ============================================

    def _is_provider_ready(self, provider_name: str) -> bool:
        """Check if a provider is configured and ready to receive requests."""
        if provider_name not in self.providers:
            return False
        p = self.providers[provider_name]
        if p.name == "local":
            import os
            return bool(os.environ.get("LOCAL_LLM_URL"))
        return bool(p.api_key)

    def _resolve_provider(
        self,
        provider_name: str | None = None,
        task_type: str | None = None,
    ) -> tuple[CustomLLMProvider, str | None]:
        """
        Resolve which provider to use based on task_type or explicit provider name.

        Returns:
            (provider, model_override) tuple
        """
        # Explicit provider name takes priority
        if provider_name and provider_name in self.providers:
            return self.providers[provider_name], None

        # Visual requests must honor the dedicated multimodal route. A text
        # default provider may be configured globally but still lack reliable
        # image support or have a much smaller input-token budget.
        if task_type in ("vision", "computer_use") and task_type in self.routing_rules:
            rule = self.routing_rules[task_type]
            target_provider = rule["provider"]
            target_model = rule.get("model")
            if self._is_provider_ready(target_provider):
                return self.providers[target_provider], target_model
            for fallback in rule.get("fallback", []):
                fb_provider = fallback.split("/")[0] if "/" in fallback else fallback
                if self._is_provider_ready(fb_provider):
                    fb_model = fallback.split("/", 1)[1] if "/" in fallback else None
                    logger.info(
                        "routing_fallback",
                        task_type=task_type,
                        original=target_provider,
                        fallback=fb_provider,
                    )
                    return self.providers[fb_provider], fb_model

        if task_type == "vision":
            raise RuntimeError(
                "No configured vision provider is ready. Configure credentials "
                "for the vision route or start its local vision model."
            )

        # If user explicitly configured a default provider (e.g. DEFAULT_PROVIDER=deepseek or openrouter),
        # prioritize it over the static routing rules if it is ready.
        if (
            self.default_provider
            and self._is_provider_ready(self.default_provider)
        ):
            target_model = None
            if task_type and task_type in self.routing_rules:
                rule = self.routing_rules[task_type]
                # Prefer the rule's primary model when it belongs to the
                # default provider; only then walk the fallback chain.
                primary = rule.get("provider")
                primary_model = rule.get("model")
                if primary == self.default_provider and primary_model:
                    target_model = primary_model
                else:
                    for fb in rule.get("fallback", []):
                        if fb and fb.startswith(f"{self.default_provider}/"):
                            target_model = fb.split("/", 1)[1]
                            break
            return self.providers[self.default_provider], target_model

        # Route by task type
        if task_type and task_type in self.routing_rules:
            rule = self.routing_rules[task_type]
            target_provider = rule["provider"]
            target_model = rule.get("model")

            if self._is_provider_ready(target_provider):
                return self.providers[target_provider], target_model

            # Try fallbacks from the rule
            for fallback in rule.get("fallback", []):
                fb_provider = fallback.split("/")[0] if "/" in fallback else fallback
                if self._is_provider_ready(fb_provider):
                    fb_model = fallback.split("/")[1] if "/" in fallback else None
                    logger.info(
                        "routing_fallback",
                        task_type=task_type,
                        original=target_provider,
                        fallback=fb_provider,
                    )
                    return self.providers[fb_provider], fb_model

        # Use default provider if ready
        if self._is_provider_ready(self.default_provider):
            return self.providers[self.default_provider], None

        # Try global fallback chain
        for fb_provider in self.fallback_chain:
            if self._is_provider_ready(fb_provider):
                return self.providers[fb_provider], None

        # Try any ready provider
        for name, p in self.providers.items():
            if self._is_provider_ready(name):
                return p, None

        # Last resort: use first available provider
        if self.providers:
            first_name = next(iter(self.providers))
            logger.warning("routing_last_resort", provider=first_name)
            return self.providers[first_name], None

        raise RuntimeError("No LLM providers configured. Add at least one provider.")

    def _budget_is_enabled(self) -> bool:
        return self.max_per_request_usd is not None or self.max_per_engagement_usd is not None

    def _estimate_request_cost(self, provider: CustomLLMProvider, request: LLMRequest) -> Decimal:
        """Conservatively estimate the request from text bytes and output-token ceiling.

        Tokenization is provider-specific, so UTF-8 bytes provide a conservative
        upper bound for textual input tokens. Image payload costs cannot be
        estimated here and are accounted from provider response usage instead.
        """
        request_data = request.model_dump(mode="json")
        for message in request_data["messages"]:
            message["images"] = []
        input_bytes = len(
            json.dumps(
                request_data,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        max_tokens = provider._effective_max_tokens(request.max_tokens)
        models = dict.fromkeys([request.model or provider.default_model, *provider.fallback_models])
        estimates = []
        for model in models:
            prices = provider.pricing_for_model(model)
            if prices is None or any(not math.isfinite(rate) or rate <= 0 for rate in prices):
                raise LLMBudgetEstimateUnavailableError(
                    f"Cannot enforce LLM budget: no complete, non-zero token pricing "
                    f"is configured for possible model {model!r} at provider {provider.name!r}"
                )
            input_rate, output_rate = prices
            estimates.append(
                Decimal(input_bytes) * Decimal(str(input_rate)) / Decimal(1000)
                + Decimal(max_tokens) * Decimal(str(output_rate)) / Decimal(1000)
            )
        return max(estimates)

    def _engagement_spend(self, engagement_id: str) -> Decimal:
        recorded = sum(
            (Decimal(str(record.cost_usd)) for record in self.cost_records if record.engagement_id == engagement_id),
            Decimal(0),
        )
        return recorded + self._engagement_uncertain_costs.get(engagement_id, Decimal(0))

    async def _reserve_budget(
        self,
        provider: CustomLLMProvider,
        request: LLMRequest,
        request_spent: list[Decimal],
    ) -> Decimal | None:
        if not self._budget_is_enabled():
            return None

        estimate = self._estimate_request_cost(provider, request)
        if any(getattr(message, "images", None) for message in request.messages):
            warning = {
                "kind": "image_budget_estimate_incomplete",
                "engagement_id": request.engagement_id,
                "estimated_text_and_output_usd": float(estimate),
                "preflight_estimate_includes_images": False,
                "hard_cap_guaranteed": False,
                "reason": (
                    "Image-token usage is not included in the preflight estimate; "
                    "reported response cost or aggregate token usage is accounted after completion."
                ),
            }
            self.incomplete_image_budget_requests.append(warning)
            logger.warning("llm_image_budget_estimate_incomplete", **warning)
        request_limit = (
            Decimal(str(self.max_per_request_usd))
            if self.max_per_request_usd is not None
            else None
        )
        if request_limit is not None and request_spent[0] + estimate > request_limit:
            raise LLMBudgetExceededError(
                "per-request",
                float(request_spent[0] + estimate),
                float(request_limit),
                request.engagement_id,
            )

        if request.engagement_id and self.max_per_engagement_usd is not None:
            engagement_id = request.engagement_id
            engagement_limit = Decimal(str(self.max_per_engagement_usd))
            async with self._budget_lock:
                committed = self._engagement_spend(engagement_id)
                reserved = self._engagement_reservations.get(engagement_id, Decimal(0))
                projected = committed + reserved + estimate
                if projected > engagement_limit:
                    raise LLMBudgetExceededError(
                        "per-engagement",
                        float(projected),
                        float(engagement_limit),
                        engagement_id,
                    )
                self._engagement_reservations[engagement_id] = reserved + estimate
                self._maybe_warn_budget(engagement_id, projected, engagement_limit)
        return estimate

    def _maybe_warn_budget(
        self,
        engagement_id: str,
        projected: Decimal,
        limit: Decimal,
    ) -> None:
        if self.warn_at_percent is None or engagement_id in self._warned_engagements or limit <= 0:
            return
        threshold = limit * Decimal(str(self.warn_at_percent)) / Decimal(100)
        if projected < threshold:
            return
        warning = {
            "engagement_id": engagement_id,
            "spent_or_reserved_usd": float(projected),
            "limit_usd": float(limit),
            "percent": float(projected / limit * 100),
            "warn_at_percent": float(self.warn_at_percent),
        }
        self._warned_engagements.add(engagement_id)
        self.budget_warnings.append(warning)
        logger.warning("llm_engagement_budget_warning", **warning)

    async def _settle_budget(
        self,
        provider: CustomLLMProvider,
        request: LLMRequest,
        estimate: Decimal | None,
        response: LLMResponse | None = None,
    ) -> Decimal:
        if estimate is None:
            if response is not None:
                # Budget controls are disabled: preserve the historical
                # provider-reported cost behavior rather than enforcing usage.
                self._record_response_cost(request, response, response.cost_usd)
                return Decimal(str(response.cost_usd))
            return Decimal(0)

        engagement_id = request.engagement_id
        try:
            actual = (
                self._account_response_cost(provider, request, response)
                if response is not None
                else estimate
            )
        except LLMBudgetEstimateUnavailableError:
            if engagement_id and self.max_per_engagement_usd is not None:
                async with self._budget_lock:
                    current = self._engagement_reservations.get(engagement_id, Decimal(0))
                    remaining = max(Decimal(0), current - estimate)
                    if remaining:
                        self._engagement_reservations[engagement_id] = remaining
                    else:
                        self._engagement_reservations.pop(engagement_id, None)
                    self._engagement_uncertain_costs[engagement_id] = (
                        self._engagement_uncertain_costs.get(engagement_id, Decimal(0)) + estimate
                    )
            unaccounted = {
                "request_id": response.request_id,
                "engagement_id": engagement_id,
                "model": response.model,
                "reserved_estimate_usd": float(estimate),
                "image_request": any(message.images for message in request.messages),
                "reason": "Response contained no usable cost or priced token usage",
            }
            self.unaccounted_response_costs.append(unaccounted)
            logger.warning("llm_response_cost_unaccounted", **unaccounted)
            raise

        async with self._budget_lock:
            if engagement_id and self.max_per_engagement_usd is not None:
                current = self._engagement_reservations.get(engagement_id, Decimal(0))
                remaining = max(Decimal(0), current - estimate)
                if remaining:
                    self._engagement_reservations[engagement_id] = remaining
                else:
                    self._engagement_reservations.pop(engagement_id, None)
                if response is None:
                    # A failed transport may have reached the provider and been
                    # billed. Keep its full estimate in the engagement ledger.
                    self._engagement_uncertain_costs[engagement_id] = (
                        self._engagement_uncertain_costs.get(engagement_id, Decimal(0)) + estimate
                    )
                    logger.warning(
                        "llm_budget_uncertain_attempt",
                        engagement_id=engagement_id,
                        estimated_cost_usd=float(estimate),
                        message="Provider error; estimate is conservatively retained because billing is unknown",
                    )
                else:
                    self._record_response_cost(request, response, float(actual))
                current_spend = self._engagement_spend(engagement_id)
                limit = Decimal(str(self.max_per_engagement_usd))
                self._maybe_warn_budget(engagement_id, current_spend, limit)
                if response is not None and current_spend > limit:
                    self._record_budget_overrun(
                        "per-engagement",
                        request,
                        current_spend,
                        limit,
                        response.request_id,
                        image_request=any(message.images for message in request.messages),
                    )
            elif response is not None:
                self._record_response_cost(request, response, float(actual))
        return actual

    def _account_response_cost(
        self,
        provider: CustomLLMProvider,
        request: LLMRequest,
        response: LLMResponse,
    ) -> Decimal:
        response_cost = Decimal(str(response.cost_usd))
        if response_cost.is_finite() and response_cost > 0:
            return response_cost

        usage = response.usage
        if usage.prompt_tokens > 0 or usage.completion_tokens > 0 or usage.total_tokens > 0:
            model = response.model or request.model or provider.default_model
            prices = provider.pricing_for_model(model)
            if prices is None:
                model = request.model or provider.default_model
                prices = provider.pricing_for_model(model)
            if prices is not None and all(math.isfinite(rate) and rate >= 0 for rate in prices):
                return Decimal(str(provider._calculate_cost(usage, model)))

        raise LLMBudgetEstimateUnavailableError(
            "LLM response completed but could not be accounted: it included neither a "
            "positive response cost nor usable token usage with configured model pricing"
        )

    def _record_response_cost(
        self,
        request: LLMRequest,
        response: LLMResponse,
        cost_usd: float,
    ) -> None:
        self.cost_records.append(CostRecord(
            request_id=response.request_id,
            provider=response.provider,
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=cost_usd,
            agent_id=request.agent_id,
            engagement_id=request.engagement_id,
        ))

    def _record_budget_overrun(
        self,
        budget: str,
        request: LLMRequest,
        amount: Decimal,
        limit: Decimal,
        request_id: str,
        *,
        image_request: bool,
    ) -> None:
        overrun = {
            "kind": "budget_exceeded_after_response",
            "budget": budget,
            "engagement_id": request.engagement_id,
            "request_id": request_id,
            "amount_usd": float(amount),
            "limit_usd": float(limit),
            "image_request": image_request,
            "hard_cap_guaranteed": not image_request,
        }
        self.budget_overruns.append(overrun)
        logger.warning("llm_budget_exceeded_after_response", **overrun)

    async def _budgeted_complete(
        self,
        provider: CustomLLMProvider,
        request: LLMRequest,
        request_spent: list[Decimal],
    ) -> LLMResponse:
        estimate = await self._reserve_budget(provider, request, request_spent)
        try:
            response = await provider.complete_with_timing(request)
        except Exception:
            if estimate is not None:
                request_spent[0] += await self._settle_budget(provider, request, estimate)
            raise
        actual = await self._settle_budget(provider, request, estimate, response)
        request_spent[0] += actual
        if (
            self.max_per_request_usd is not None
            and request_spent[0] > Decimal(str(self.max_per_request_usd))
        ):
            self._record_budget_overrun(
                "per-request",
                request,
                request_spent[0],
                Decimal(str(self.max_per_request_usd)),
                response.request_id,
                image_request=any(message.images for message in request.messages),
            )
        return response

    # ============================================
    # Complete & Stream
    # ============================================

    async def complete(
        self,
        request: LLMRequest,
        task_type: str | None = None,
        provider_name: str | None = None,
    ) -> LLMResponse:
        """
        Route and complete an LLM request.

        Args:
            request: The LLM request to send
            task_type: Optional task type for routing (e.g., "reasoning", "coding")
            provider_name: Optional explicit provider name

        Returns:
            LLMResponse from the selected provider
        """
        # Use request's task_type if not explicitly provided
        effective_task = task_type or request.task_type

        # Requests containing pixels require a vision-capable model even when
        # their caller labels the surrounding task as reasoning or computer use.
        has_images = any(getattr(msg, "has_images", False) for msg in request.messages)
        if has_images and "vision" in self.routing_rules:
            effective_task = "vision"

        provider, model_override = self._resolve_provider(provider_name, effective_task)
        routed_request = request.model_copy(deep=True)

        # Override model if routing rule specifies one
        if model_override and not routed_request.model:
            routed_request.model = model_override

        # If the resolved target model is a high-parameter reasoning/coding model
        # (e.g. nemotron-3-super-120b or nemotron-3-ultra-550b) that does not accept
        # raw image_url blocks, strip image bytes only from the primary attempt.
        # Keep the routed request intact so a vision fallback still receives pixels.
        active_model_name = (
            routed_request.model or getattr(provider, "default_model", "") or ""
        ).lower()
        if (
            has_images
            and effective_task != "vision"
            and not self._model_supports_images(active_model_name)
        ):
            primary_request = routed_request.model_copy(deep=True)
            for msg in primary_request.messages:
                if getattr(msg, "images", None):
                    msg.images = []
        else:
            primary_request = routed_request

        # Set task metadata
        routed_request.task_type = effective_task
        primary_request.task_type = effective_task
        request_spent = [Decimal(0)]

        try:
            response = await self._budgeted_complete(provider, primary_request, request_spent)

            # Check for safety refusal; trigger model fallback if detected
            if is_refusal_response(response):
                logger.warning(
                    "llm_safety_refusal_detected_triggering_fallback",
                    provider=provider.name,
                    model=response.model,
                    task_type=effective_task,
                )
                return await self._try_fallbacks(
                    routed_request,
                    provider.name,
                    effective_task,
                    request_spent=request_spent,
                )

            return response

        except (LLMBudgetExceededError, LLMBudgetEstimateUnavailableError):
            raise
        except Exception as e:
            logger.error(
                "complete_failed",
                provider=provider.name,
                error=str(e),
                task_type=effective_task,
            )
            # Try fallback chain
            return await self._try_fallbacks(
                routed_request,
                provider.name,
                effective_task,
                request_spent=request_spent,
                primary_error=e,
            )

    @staticmethod
    def _model_supports_images(model_name: str) -> bool:
        """Best-effort check for model families that accept image payloads."""
        return any(marker in model_name for marker in ("vision", "vl", "gpt-4o", "claude"))

    def _fallback_candidates(self, task_type: str | None) -> list[tuple[str, str | None]]:
        """Return configured task and global fallbacks in their configured order."""
        candidates: list[tuple[str, str | None]] = []
        if task_type and task_type in self.routing_rules:
            for fallback in self.routing_rules[task_type].get("fallback", []):
                if "/" in fallback:
                    name, model = fallback.split("/", 1)
                    candidates.append((name, model))
                else:
                    candidates.append((fallback, None))
        if task_type != "vision":
            candidates.extend((name, None) for name in self.fallback_chain)
        return candidates

    async def _try_fallbacks(
        self,
        request: LLMRequest,
        failed_provider: str,
        task_type: str | None,
        request_spent: list[Decimal] | None = None,
        primary_error: Exception | None = None,
    ) -> LLMResponse:
        """Try fallback providers when the primary fails."""
        if request_spent is None:
            request_spent = [Decimal(0)]
        candidates = self._fallback_candidates(task_type)
        original_model = request.model
        failures: list[str] = []
        if primary_error is not None:
            failures.append(
                f"{failed_provider}/{original_model or 'default'}: "
                f"{self._provider_error_summary(primary_error)}"
            )
        for fb_name, fb_model in candidates:
            if not self._is_provider_ready(fb_name):
                if task_type == "vision":
                    failures.append(
                        f"{fb_name}/{fb_model or 'default'}: provider not configured"
                    )
                continue
            # Skip if same provider and no different model specified
            if fb_name == failed_provider and (not fb_model or fb_model == original_model):
                continue

            logger.info("trying_fallback", provider=fb_name, model=fb_model, task_type=task_type)
            try:
                fb_provider = self.providers[fb_name]
                fallback_request = request.model_copy(deep=True)
                fallback_request.model = fb_model
                return await self._budgeted_complete(fb_provider, fallback_request, request_spent)
            except (LLMBudgetExceededError, LLMBudgetEstimateUnavailableError):
                raise
            except Exception as e:
                logger.warning("fallback_failed", provider=fb_name, model=fb_model, error=str(e))
                failures.append(
                    f"{fb_name}/{fb_model or 'default'}: "
                    f"{self._provider_error_summary(e)}"
                )
                continue

        if task_type == "vision":
            failure_detail = f" Attempts: {'; '.join(failures)}." if failures else ""
            raise RuntimeError(
                "All configured vision providers failed."
                f"{failure_detail} Configure a working "
                "vision provider or start its local vision model."
            )
        raise RuntimeError(
            f"All providers failed for task_type={task_type}. "
            f"Primary: {failed_provider} (model: {original_model}), Candidates tried: {candidates}"
        )

    @staticmethod
    def _provider_error_summary(error: Exception) -> str:
        """Expose provider failure class/status without leaking response bodies."""
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return f"HTTP {status_code}"
        return type(error).__name__

    async def stream(
        self,
        request: LLMRequest,
        task_type: str | None = None,
        provider_name: str | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Route and stream, falling back only before the first emitted chunk."""
        effective_task = task_type or request.task_type
        has_images = any(getattr(msg, "has_images", False) for msg in request.messages)
        if has_images and "vision" in self.routing_rules:
            effective_task = "vision"
        provider, model_override = self._resolve_provider(provider_name, effective_task)

        routed_request = request.model_copy(deep=True)
        if model_override and not routed_request.model:
            routed_request.model = model_override
        routed_request.task_type = effective_task
        candidates = [(provider.name, None), *self._fallback_candidates(effective_task)]
        original_model = routed_request.model
        request_spent = [Decimal(0)]
        last_error: Exception | None = None

        for candidate_index, (provider_name, fallback_model) in enumerate(candidates):
            if provider_name != provider.name and not self._is_provider_ready(provider_name):
                continue
            if candidate_index > 0 and provider_name == provider.name and (
                not fallback_model or fallback_model == original_model
            ):
                continue
            attempt_provider = self.providers[provider_name]
            attempt_request = routed_request.model_copy(deep=True)
            if fallback_model:
                attempt_request.model = fallback_model
            attempt_model = (attempt_request.model or attempt_provider.default_model).lower()
            if (
                has_images
                and effective_task != "vision"
                and not self._model_supports_images(attempt_model)
            ):
                for message in attempt_request.messages:
                    message.images = []

            estimate = await self._reserve_budget(
                attempt_provider, attempt_request, request_spent
            )
            usage_response: LLMResponse | None = None
            emitted = False
            try:
                async for chunk in attempt_provider.stream(attempt_request):
                    if chunk.is_final and chunk.usage is not None:
                        response_model = attempt_request.model or attempt_provider.default_model
                        usage_response = LLMResponse(
                            request_id=attempt_provider.generate_request_id(),
                            model=response_model,
                            provider=attempt_provider.provider_name,
                            usage=chunk.usage,
                            cost_usd=attempt_provider._calculate_cost(
                                chunk.usage, response_model
                            ),
                        )
                    emitted = True
                    yield chunk
            except (LLMBudgetExceededError, LLMBudgetEstimateUnavailableError):
                raise
            except Exception as error:
                actual = await self._settle_budget(
                    attempt_provider, attempt_request, estimate, usage_response
                )
                request_spent[0] += actual
                if emitted:
                    raise
                last_error = error
                logger.warning(
                    "llm_stream_provider_failed_before_output",
                    provider=provider_name,
                    model=attempt_request.model or attempt_provider.default_model,
                    error=str(error),
                )
                continue
            except BaseException:
                await self._settle_budget(
                    attempt_provider, attempt_request, estimate, usage_response
                )
                raise

            actual = await self._settle_budget(
                attempt_provider, attempt_request, estimate, usage_response
            )
            request_spent[0] += actual
            if (
                self.max_per_request_usd is not None
                and request_spent[0] > Decimal(str(self.max_per_request_usd))
                and usage_response is not None
            ):
                self._record_budget_overrun(
                    "per-request",
                    attempt_request,
                    request_spent[0],
                    Decimal(str(self.max_per_request_usd)),
                    usage_response.request_id,
                    image_request=has_images,
                )
            return

        if last_error is not None:
            raise RuntimeError(
                f"All configured providers failed before streaming output "
                f"for task_type={effective_task}"
            ) from last_error
        raise RuntimeError(
            f"No configured fallback provider was available for streaming "
            f"task_type={effective_task} after primary {provider.name!r}"
        )

    # ============================================
    # Info & Stats
    # ============================================

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered providers."""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.health_check()
        return results

    def get_all_models(self) -> list[dict]:
        """List all available models across all providers."""
        models = []
        for provider in self.providers.values():
            for model in provider.list_models():
                models.append(model.model_dump())
        return models

    def get_total_cost(self, engagement_id: str | None = None) -> float:
        """Get total cost in USD, optionally filtered by engagement."""
        records = self.cost_records
        if engagement_id:
            records = [r for r in records if r.engagement_id == engagement_id]
        return sum(r.cost_usd for r in records)

    def get_stats(self) -> dict:
        """Get router-level statistics.

        Provider stats are derived from cost_records (single source of truth) so
        they never diverge from the router-level totals on partial failures or
        provider fallback.
        """
        provider_stats: dict[str, dict] = {}
        for name, provider in self.providers.items():
            prov_records = [r for r in self.cost_records if r.provider.value == provider.provider_name.value]
            provider_stats[name] = {
                "provider": provider.provider_name,
                "total_requests": len(prov_records),
                "total_tokens": sum(r.input_tokens + r.output_tokens for r in prov_records),
                "total_cost_usd": round(sum(r.cost_usd for r in prov_records), 6),
            }
        return {
            "providers": provider_stats,
            "total_cost_usd": round(self.get_total_cost(), 6),
            "total_requests": len(self.cost_records),
            "routing_rules": list(self.routing_rules.keys()),
            "budget": {
                "max_per_request_usd": self.max_per_request_usd,
                "max_per_engagement_usd": self.max_per_engagement_usd,
                "warn_at_percent": self.warn_at_percent,
                "reserved_by_engagement_usd": {
                    engagement_id: float(amount)
                    for engagement_id, amount in self._engagement_reservations.items()
                },
                "uncertain_by_engagement_usd": {
                    engagement_id: float(amount)
                    for engagement_id, amount in self._engagement_uncertain_costs.items()
                },
                "warnings": list(self.budget_warnings),
                "incomplete_image_estimates": list(self.incomplete_image_budget_requests),
                "image_preflight_limit_guaranteed": False,
                "overruns": list(self.budget_overruns),
                "unaccounted_responses": list(self.unaccounted_response_costs),
            },
        }
