"""
SONIC-REDA — Mission Director Engine (Phase 15)
=================================================
Top-level autonomous Mission Owner orchestrating the complete lifecycle:
Decomposition -> Planning -> Resource Allocation -> Multi-Track Execution ->
Computer Operation -> Verification -> Replanning -> Deliverable Generation.
"""

from __future__ import annotations

import time
from typing import Any

from sonic.brain.hypothesis import HypothesisEngine
from sonic.computer.models import ComputerWorkspaceType
from sonic.computer.provider import UnifiedComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ComputerAutonomyLevel, EngineeringMissionMode
from sonic.logger import get_logger
from sonic.mission_engine.models import (
    MilestoneStatus,
    MissionDeliverable,
    MissionEvent,
    MissionKnowledgeSummary,
    MissionMilestone,
    MissionObjective,
    MissionOutcome,
    MissionPhase,
    MissionPlan,
    MissionState,
    MissionStatus,
    _new_id,
    _now,
)
from sonic.mission_engine.resource_manager import MissionResourceManager
from sonic.mission_engine.store import MissionStateStore

logger = get_logger(__name__)


class MissionDirector:
    """
    Autonomous Mission Owner coordinating long-horizon missions across
    Cognitive Reasoning, Research Tracks, Computer Workspaces, and Verification.
    """

    def __init__(
        self,
        computer_provider: UnifiedComputerProvider,
        resource_manager: MissionResourceManager | None = None,
        autonomy_level: ComputerAutonomyLevel = ComputerAutonomyLevel.L3_AUTONOMOUS,
        state_store: MissionStateStore | None = None,
        model_router: Any | None = None,
    ):
        self.computer = computer_provider
        self.resource_mgr = resource_manager or MissionResourceManager()
        self.autonomy_level = autonomy_level
        if model_router is not None:
            self.model_router = model_router
        else:
            try:
                from sonic.llm.router import ModelRouter
                self.model_router = ModelRouter()
            except Exception:
                self.model_router = None

        # Active missions state cache (in-memory, persisted to store on mutation)
        self.missions: dict[str, MissionState] = {}
        self.events: dict[str, list[MissionEvent]] = {}
        self.deliverables: dict[str, list[MissionDeliverable]] = {}

        # Persistent backing store (Redis with in-memory fallback)
        self.store = state_store or MissionStateStore()

    async def connect_store(self) -> bool:
        """Initialize the state store connection. Call once at startup."""
        return await self.store.connect()

    async def _persist_mission(self, mission_id: str) -> None:
        state = self.missions.get(mission_id)
        if state:
            await self.store.save_state(state)
            events = self.events.get(mission_id, [])
            if not hasattr(self, "_persisted_event_counts"):
                self._persisted_event_counts: dict[str, int] = {}
            last_count = self._persisted_event_counts.get(mission_id, 0)
            new_events = events[last_count:]
            for evt in new_events:
                await self.store.append_event(evt)
            self._persisted_event_counts[mission_id] = len(events)
            if self.deliverables.get(mission_id):
                await self.store.save_deliverables(mission_id, self.deliverables[mission_id])

    async def restore_mission(self, mission_id: str) -> MissionState | None:
        """Load a mission from the persistent store into the in-memory cache."""
        state = await self.store.load_state(mission_id)
        if not state:
            return None
        self.missions[mission_id] = state
        self.events[mission_id] = await self.store.load_events(mission_id)
        if not hasattr(self, "_persisted_event_counts"):
            self._persisted_event_counts = {}
        self._persisted_event_counts[mission_id] = len(self.events[mission_id])
        self.deliverables[mission_id] = await self.store.load_deliverables(mission_id)
        logger.info("mission_restored", mission_id=mission_id, status=state.status)
        return state

    async def restore_all(self) -> list[str]:
        """Restore all persisted missions into the in-memory cache (startup recovery)."""
        ids = await self.store.list_mission_ids()
        for mid in ids:
            await self.restore_mission(mid)
        return ids

    # =============================================================
    # 1. Mission Creation & Objective Understanding
    # =============================================================
    @staticmethod
    def _derive_hypothesis_from_objective(goal: str, target: str) -> tuple[str, str, str, str]:
        """Derive domain-neutral state-convergence hypothesis pair organically from the target objective."""
        g = goal.strip() or "complete objective"
        tgt = target.strip() or "workstation"
        return (
            f"Target '{tgt}' state converges to satisfy objective with verified execution receipts: {g}",
            f"Target '{tgt}' state fails to converge or lacks empirical verification for objective: {g}",
            "objective_fulfillment",
            "state_convergence",
        )

    async def create_mission(
        self,
        tenant_id: str,
        goal: str,
        constraints: list[str] | None = None,
        scope: list[str] | None = None,
        budget_dollars: float = 25.0,
        deadline_seconds: int = 3600,
        success_criteria: list[str] | None = None,
        failure_criteria: list[str] | None = None,
    ) -> MissionState:
        """Create and initialize a new top-level autonomous mission."""
        mission_id = _new_id("msn")

        if success_criteria is None:
            crit = [
                f"Target state converged for objective: {goal[:60]}",
                "ActionContract postconditions and evidence package verified",
            ]
        else:
            crit = success_criteria

        objective = MissionObjective(
            tenant_id=tenant_id,
            mission_id=mission_id,
            goal=goal,
            constraints=constraints or ["Zero host execution", "Strict tenant isolation"],
            scope=scope or ["/home/sonic/workspace", "http://127.0.0.1:8080"],
            success_criteria=crit,
            failure_criteria=failure_criteria or ["Host safety violation", "Budget exhaustion", "Deadlock without recovery"],
            budget_dollars=budget_dollars,
            deadline_seconds=deadline_seconds,
        )

        self.resource_mgr.register_mission(mission_id, budget_dollars=budget_dollars)

        state = MissionState(
            mission_id=mission_id,
            tenant_id=tenant_id,
            objective=objective,
            current_phase=MissionPhase.DISCOVERY,
            status=MissionStatus.PLANNING,
            current_next_action="Decompose mission objective into state-convergence plan",
        )

        self.missions[mission_id] = state
        self.events[mission_id] = []
        self.deliverables[mission_id] = []

        self._record_event(mission_id, "MissionStarted", {"goal": goal, "tenant_id": tenant_id})
        await self._persist_mission(mission_id)
        logger.info("mission_created", mission_id=mission_id, goal=goal)

        # Automatically decompose objective into initial plan
        await self.decompose_mission(mission_id)
        return state

    # =============================================================
    # 2. Objective Decomposition & Adaptive State-Convergence Planning
    # =============================================================
    async def decompose_mission(self, mission_id: str) -> MissionPlan:
        """Decomposes a high-level goal into domain-neutral state-convergence phases, questions, tracks, and milestones."""
        state = self._require_mission(mission_id)
        goal = state.objective.goal

        questions = [
            f"What is the initial observable state and context required to achieve: '{goal}'?",
            "Which state transitions (StateDelta) and execution planes (GUI / terminal) are optimal?",
            "What empirical receipts confirm that ActionContract postconditions are satisfied?",
        ]
        tracks = [
            "Track-A: Repository & Target Surface State Discovery",
            "Track-B: Dual-Plane State-Convergence Execution",
            "Track-C: ActionContract Postcondition Verification & Deliverables",
        ]
        milestones = [
            MissionMilestone(
                mission_id=mission_id,
                name="Milestone 1: Repository Surface Discovery & Vector Inspection",
                objective=f"Observe initial state, repository context, and target surface for {goal[:60]}",
                success_conditions=["Initial state snapshot captured", "Target context and surface identified"],
                status=MilestoneStatus.ACTIVE,
            ),
            MissionMilestone(
                mission_id=mission_id,
                name="Milestone 2: State-Convergence Execution & Verification",
                objective="Execute actions autonomously across GUI and terminal planes to converge state toward objective",
                success_conditions=["StateDelta transitions validated", "Empirical execution receipts verified"],
                dependencies=["Milestone 1"],
                status=MilestoneStatus.PENDING,
            ),
            MissionMilestone(
                mission_id=mission_id,
                name="Milestone 3: Git Commit & Deliverable Finalization",
                objective="Verify postconditions, persist artifacts, and finalize deliverable report",
                success_conditions=["Git commit created", "Evidence custody chain signed"],
                dependencies=["Milestone 2"],
                status=MilestoneStatus.PENDING,
            ),
        ]

        plan = MissionPlan(
            mission_id=mission_id,
            objective=state.objective.goal,
            research_questions=questions,
            investigation_tracks=tracks,
            milestones=milestones,
            stop_conditions=["Goal satisfied with verified receipts", "Budget exhausted"],
            version=1,
        )

        state.current_plan = plan
        state.open_questions = list(questions)
        state.active_tracks = list(tracks)
        state.remaining_unknowns = [
            f"Verification of: {state.objective.goal[:60]}",
            "StateDelta postcondition validation",
        ] if state.objective and state.objective.goal else [
            "Initial state discovery",
            "Postcondition verification",
        ]
        state.status = MissionStatus.ACTIVE

        self._record_event(mission_id, "PlanDecomposed", {"version": 1, "milestones": len(milestones)})
        return plan

    # =============================================================
    # 2b. Milestone Execution with Being, Toolsmith, MethodLab & LessonsLedger
    # =============================================================
    async def execute_milestone(
        self,
        mission_id: str,
        milestone: Any = None,
        workspace_id: str | None = None,
        steps: int = 12,
    ) -> dict[str, Any]:
        """Execute a single mission milestone with Being, ToolsmithLoop, MethodLab,
        LessonsLedger, and HDCE ActionContract state-delta verification wired in."""
        from sonic.being.craft import BeingCraft
        from sonic.being.identity import get_being_store, get_or_create_being
        from sonic.being.lessons import LessonsLedger
        from sonic.being.method_lab import MethodLab
        from sonic.being.toolsmith import ToolsmithLoop
        from sonic.computer_use.models import ActionExecutionStatus
        from sonic.evolution.engine import EvolutionEngine
        from sonic.evolution.strategy import DynamicStrategyEngine
        from sonic.memory.vector import get_vector_memory
        from sonic.mission_engine.hdce import (
            ActionContract,
            ActionContractValidator,
            ActionReceipt,
            StateDelta,
        )
        from sonic.safety.sealed import seal_default
        from sonic.tools.computer_as_compute_provider import ComputerAsComputeProvider
        from sonic.tools.registry import get_default_registry

        state = self._require_mission(mission_id)
        if state.current_plan is None:
            await self.decompose_mission(mission_id)

        target_ms: MissionMilestone | None = None
        if isinstance(milestone, MissionMilestone):
            target_ms = milestone
        elif isinstance(milestone, str) and state.current_plan:
            for m in state.current_plan.milestones:
                if m.milestone_id == milestone or m.name == milestone:
                    target_ms = m
                    break
        if target_ms is None and state.current_plan and state.current_plan.milestones:
            for m in state.current_plan.milestones:
                if m.status in (MilestoneStatus.ACTIVE, MilestoneStatus.PENDING):
                    target_ms = m
                    break
            if target_ms is None:
                target_ms = state.current_plan.milestones[0]

        ws_id = workspace_id
        if not ws_id:
            if state.computer_workspaces:
                ws_id = state.computer_workspaces[0]
            else:
                ws = await self.computer.create(
                    tenant_id=state.tenant_id,
                    engagement_id=state.mission_id,
                    workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
                )
                ws_id = getattr(ws, "id", getattr(ws, "workspace_id", str(ws)))
                state.computer_workspaces.append(ws_id)
                self.resource_mgr.allocate_sandbox(mission_id, ws_id)

        being = get_or_create_being(state.tenant_id)
        being_store = get_being_store()
        being_mind = being_store.get_mind(being.being_id)
        craft = BeingCraft(being_id=being.being_id)
        security_registry = get_default_registry(ComputerAsComputeProvider(self.computer))
        sec_tools = security_registry.as_dict()
        toolsmith = ToolsmithLoop(
            craft=craft,
            llm=self.model_router,
            registry=security_registry,
        )
        method_lab = MethodLab(
            llm=self.model_router,
            vector_memory=get_vector_memory(),
            toolsmith=toolsmith,
        )
        lessons_ledger = LessonsLedger(tenant_id=state.tenant_id, agent_id=being.being_id)
        evolution_engine = EvolutionEngine(
            strategy_engine=DynamicStrategyEngine(),
            method_lab=method_lab,
            toolsmith=toolsmith,
            lessons_ledger=lessons_ledger,
        )

        workspace_root = "/home/sonic/workspace"
        if getattr(state, "objective", None) and getattr(state.objective, "scope", None):
            for s in state.objective.scope:
                if s.startswith("/"):
                    workspace_root = s
                    break
        safety_policy = seal_default(workspace_root=workspace_root)

        before_snapshot = {
            "milestone_status": str(target_ms.status.value if target_ms else "PENDING"),
            "workspace_id": ws_id,
            "verified": False,
        }
        contract = ActionContract(
            action_type="TERMINAL_EXEC",
            preconditions={"workspace_id": ws_id},
            postconditions={"verified": True},
        )
        validator = ActionContractValidator()

        agent = ComputerUseAgent(
            computer_provider=self.computer,
            llm_router=self.model_router,
            security_tools=sec_tools,
            autonomy_level=self.autonomy_level,
            mode=EngineeringMissionMode.ENGINEERING_MODE,
            browser=None,
            gui_only=False,
            observe_desktop=True,
            tenant_id=state.tenant_id,
            agent_id=being.being_id,
            being_mind=being_mind,
            safety=safety_policy,
            toolsmith=toolsmith,
            method_lab=method_lab,
            lessons_ledger=lessons_ledger,
            evolution_engine=evolution_engine,
        )

        ms_goal = (
            f"{target_ms.name}: {target_ms.objective} (Mission Goal: {state.objective.goal})"
            if target_ms is not None
            else state.objective.goal
        )
        traces = await agent.run_mission(
            workspace_id=ws_id,
            goal=ms_goal,
            steps=steps,
        )
        existing_traces = list(getattr(state, "_last_traces", []) or [])
        existing_traces.extend(traces)
        self.stash_traces(mission_id, existing_traces)

        successful = sum(
            1 for t in traces if str(getattr(t, "status", "")).upper() in ("SUCCESS", "SUCCEEDED", "RECOVERED")
        )
        converged = successful > 0
        if target_ms is not None:
            target_ms.status = MilestoneStatus.COMPLETED if converged else MilestoneStatus.FAILED

        after_snapshot = {
            "milestone_status": str(target_ms.status.value if target_ms else ("COMPLETED" if converged else "FAILED")),
            "workspace_id": ws_id,
            "verified": converged,
        }
        delta = StateDelta.from_states(before_snapshot, after_snapshot)
        receipt = ActionReceipt(
            action_type="TERMINAL_EXEC",
            status=ActionExecutionStatus.SUCCEEDED if converged else ActionExecutionStatus.FAILED,
            state_delta=delta,
            before=before_snapshot,
            after=after_snapshot,
            exit_code=0 if converged else 1,
            evidence=f"Executed {len(traces)} steps ({successful} succeeded) for milestone.",
        )
        contract_validation = validator.validate_receipt(contract, receipt)

        self._record_event(
            mission_id,
            "MilestoneExecuted",
            {
                "milestone": target_ms.name if target_ms else "default",
                "status": after_snapshot["milestone_status"],
                "being_id": being.being_id,
                "traces": len(traces),
                "contract_valid": contract_validation.valid,
            },
        )
        await self._persist_mission(mission_id)
        return {
            "mission_id": mission_id,
            "milestone": target_ms,
            "being_id": being.being_id,
            "traces": traces,
            "state_delta": delta,
            "receipt": receipt,
            "contract_valid": contract_validation.valid,
            "status": after_snapshot["milestone_status"],
        }

    # =============================================================
    # 3. Execution Coordination & Computer Operation
    # =============================================================
    async def coordinate(self, mission_id: str) -> MissionState:
        """Executes the complete long-horizon mission lifecycle autonomously."""
        state = self._require_mission(mission_id)
        t_start = time.perf_counter()

        try:
            # 1. Phase: DISCOVERY & Computer Workspace Allocation
            state.current_phase = MissionPhase.DISCOVERY
            ws = await self.computer.create(
                tenant_id=state.tenant_id,
                engagement_id=state.mission_id,
                workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
            )
            state.computer_workspaces.append(ws.id)
            self.resource_mgr.allocate_sandbox(mission_id, ws.id)
            self._record_event(mission_id, "ComputerCreated", {"workspace_id": ws.id})

            # 2. Phase: UNDERSTANDING & RESEARCH
            state.current_phase = MissionPhase.RESEARCH
            hypothesis_engine = HypothesisEngine()
            target_asset = ""
            for c in getattr(state.objective, "constraints", []):
                c_str = str(c).strip()
                if any(k in c_str.lower() for k in ("target:", "asset:", "host:", "http://", "https://", "/", ".py", ".ts", ".js")):
                    target_asset = c_str
                    break
            if not target_asset:
                if getattr(state.objective, "scope", None) and state.objective.scope:
                    target_asset = state.objective.scope[0]
                elif getattr(state.objective, "constraints", None) and state.objective.constraints:
                    target_asset = state.objective.constraints[0]
                else:
                    target_asset = getattr(state.objective, "goal", "target_asset")

            # Organically derive hypothesis pair from target objective rather than hardcoding vulnerability types
            statement, counter_statement, v_class, claim_type = self._derive_hypothesis_from_objective(
                state.objective.goal, target_asset
            )
            h_primary, h_counter = hypothesis_engine.create_hypothesis_pair(
                statement=statement,
                counter_statement=counter_statement,
                target_asset=target_asset,
                vulnerability_class=v_class,
                claim_type=claim_type,
            )
            object.__setattr__(state, "_hypothesis_engine", hypothesis_engine)
            object.__setattr__(state, "_h_primary", h_primary)
            object.__setattr__(state, "_h_counter", h_counter)

            state.active_hypotheses = [
                f"[{h_primary.status.value.upper()}] Primary ({h_primary.hypothesis_id}): {h_primary.statement} (confidence: {h_primary.confidence:.2f})",
                f"[{h_counter.status.value.upper()}] Counter ({h_counter.hypothesis_id}): {h_counter.statement} (confidence: {h_counter.confidence:.2f})",
            ]
            state.evidence.append(f"Mission initialized in workspace {ws.id}")

            # 3. Phase: ENGINEERING (Delegate to ComputerUseAgent with creative autonomy)
            state.current_phase = MissionPhase.ENGINEERING
            # The agent is granted full creative autonomy to explore and approach
            # the target asset without forced tool chains or scripted recipes.
            # It has access to:
            #   - Unified browser automation (BrowserAgent)
            #   - Security tool adapters (in-sandbox, fail-closed)
            #   - Dynamic tool authoring (ToolsmithLoop)
            #   - Autonomous attack/recon method synthesis (MethodLab)
            #   - Continuous learning (LessonsLedger)
            from sonic.tools.computer_as_compute_provider import ComputerAsComputeProvider
            from sonic.tools.registry import get_default_registry

            security_registry = get_default_registry(
                ComputerAsComputeProvider(self.computer)
            )
            sec_tools = security_registry.as_dict()
            provider = self.computer
            mission = state

            # Ensure ws has workspace_id attribute
            if not hasattr(ws, "workspace_id"):
                try:
                    ws.workspace_id = getattr(ws, "id", str(ws))
                except Exception:
                    pass

            from sonic.being.craft import BeingCraft
            from sonic.being.identity import get_being_store, get_or_create_being
            from sonic.being.method_lab import MethodLab
            from sonic.being.toolsmith import ToolsmithLoop
            from sonic.memory.vector import get_vector_memory
            from sonic.safety.sealed import seal_default
            # Dual-Plane Operational Model (Rule 5): both Graphical Desktop (GUI) and
            # Dedicated Headless Terminal/Tool plane remain concurrently active.
            gui_only = False
            being = get_or_create_being(mission.tenant_id)
            being_store = get_being_store()
            being_mind = being_store.get_mind(being.being_id)

            toolsmith = ToolsmithLoop(
                craft=BeingCraft(being_id=being.being_id),
                llm=self.model_router,
                registry=security_registry,
            )
            method_lab = MethodLab(
                llm=self.model_router,
                vector_memory=get_vector_memory(),
                toolsmith=toolsmith,
            )
            from sonic.being.lessons import LessonsLedger
            obj_tenant = getattr(state.objective, "tenant_id", "default") if hasattr(state, "objective") else "default"
            lessons_ledger = LessonsLedger(tenant_id=obj_tenant, agent_id=being.being_id)
            from sonic.evolution.engine import EvolutionEngine
            from sonic.evolution.strategy import DynamicStrategyEngine
            evolution_engine = EvolutionEngine(
                strategy_engine=DynamicStrategyEngine(),
                method_lab=method_lab,
                toolsmith=toolsmith,
                lessons_ledger=lessons_ledger,
            )

            workspace_root = "/home/sonic/workspace"
            if getattr(state, "objective", None) and getattr(state.objective, "scope", None):
                for s in state.objective.scope:
                    if s.startswith("/"):
                        workspace_root = s
                        break
            safety_policy = seal_default(workspace_root=workspace_root)

            # Check if the mission objective is complex or multi-step (or if autonomy_level >= L3_AUTONOMOUS)
            traces = []
            if self._is_complex_or_autonomous(mission.objective, getattr(state, "current_plan", None)):
                try:
                    from sonic.computer_use.boss import BossAgent

                    boss = BossAgent(
                        computer_provider=provider,
                        llm_router=self.model_router,
                        safety=safety_policy,
                        security_tools=sec_tools,
                        tenant_id=mission.tenant_id,
                        being=being,
                        being_mind=being_mind,
                        agent_id=being.being_id,
                        max_phases=3,
                        sub_agent_steps=6,
                        toolsmith=toolsmith,
                        method_lab=method_lab,
                        lessons_ledger=lessons_ledger,
                        evolution_engine=evolution_engine,
                        gui_only=False,
                    )
                    report = await boss.run(
                        workspace_id=ws.workspace_id,
                        objective=mission.objective,
                    )
                    traces = report.all_traces
                except Exception as e:
                    logger.warning("boss_agent_coordination_fallback", mission_id=mission_id, error=str(e))
                    traces = []

            # If BossAgent run produces traces, use them; if it returns empty, fall back gracefully to single ComputerUseAgent.
            if not traces:
                agent = ComputerUseAgent(
                    computer_provider=self.computer,
                    llm_router=self.model_router,
                    security_tools=sec_tools,
                    autonomy_level=self.autonomy_level,
                    mode=EngineeringMissionMode.ENGINEERING_MODE,
                    browser=None,
                    gui_only=False,
                    observe_desktop=True,
                    tenant_id=mission.tenant_id,
                    agent_id=being.being_id,
                    being_mind=being_mind,
                    safety=safety_policy,
                    toolsmith=toolsmith,
                    method_lab=method_lab,
                    lessons_ledger=lessons_ledger,
                    evolution_engine=evolution_engine,
                )
                mission_steps = getattr(self, "max_actions", getattr(agent, "max_actions", 25))
                traces = await agent.run_mission(
                    workspace_id=ws.workspace_id,
                    goal=getattr(mission.objective, "goal", str(mission.objective)),
                    steps=mission_steps,
                )

            # Stash traces so get_knowledge_summary / re-finalize can derive
            # honest artifacts without re-running the agent.
            self.stash_traces(mission_id, traces)

            # Record Resource Consumption derived from actual execution
            num_traces = len(traces)
            est_tokens = sum(len(getattr(t, "actual_observation", "") or "") // 4 + 400 for t in traces) if traces else 250
            est_dollars = round((est_tokens / 1_000_000.0) * 1.50, 4)
            self.resource_mgr.record_spend(
                mission_id=mission_id,
                dollars=est_dollars,
                tokens=est_tokens,
                compute_seconds=round(time.perf_counter() - t_start, 2),
            )

            # Update Milestones — COMPLETED only where traces prove progress,
            # never force-completed by decree.
            from sonic.mission_engine.trace_synthesis import milestone_status_from_traces
            updated_milestones, progress = milestone_status_from_traces(traces, state.current_plan)
            if updated_milestones and state.current_plan:
                state.current_plan.milestones = updated_milestones
            state.progress_pct = progress

            # 4. Phase: SYNTHESIS & VERIFICATION
            state.current_phase = MissionPhase.VERIFICATION
            # Confidence is derived from the agent's actual success ratio.
            total_actions = len(traces)
            successful = sum(1 for t in traces if t.status in ("SUCCESS", "RECOVERED"))
            state.confidence = round(successful / total_actions, 2) if total_actions else 0.0
            state.completed_tracks = list(state.active_tracks)
            state.active_tracks = []
            state.remaining_unknowns = [] if successful else state.remaining_unknowns

            # Evaluate agent traces against the generated hypotheses.
            # Autonomy-first: evaluate any successful actions that generate concrete
            # evidence, findings, or meaningful observations toward the target.
            evidence_traces = []
            for t in traces:
                st = str(getattr(t.status, "value", t.status))
                obs = str(getattr(t, "actual_observation", "") or "").strip()
                meta = getattr(t, "metadata", {}) or {}
                has_findings = bool(meta.get("findings"))
                cmd_str = str(getattr(t, "target_resource", "") or meta.get("command", "")).strip().lower()
                is_orientation = any(cmd_str == o or cmd_str.startswith(f"{o} ") for o in ("pwd", "ls", "whoami", "id", "echo", "cd"))
                has_receipt = bool(meta.get("reproduction_receipt") or meta.get("verified_finding"))
                if st in ("SUCCESS", "RECOVERED"):
                    # Rule 6: Only real security tool findings or verified reproduction receipts confirm a vulnerability hypothesis
                    if has_findings or has_receipt:
                        evidence_traces.append(t)

            if evidence_traces:
                ev_items = []
                for ft in evidence_traces:
                    act_str = getattr(ft.action_type, "value", ft.action_type)
                    obs_str = getattr(ft, "actual_observation", "") or ""
                    meta = getattr(ft, "metadata", {}) or {}
                    findings_list = meta.get("findings")
                    tgt = getattr(ft, "target_resource", "") or target_asset
                    if isinstance(findings_list, list) and len(findings_list) > 0:
                        for item in findings_list[:2]:
                            ev_items.append(f"{act_str} on {tgt}: {str(item)[:100]}")
                    else:
                        ev_items.append(f"{act_str} on {tgt}: {obs_str[:100]}")

                for ev_text in ev_items[:3]:
                    h_primary.update_confidence(delta=0.4, evidence_summary=ev_text, is_supporting=True)
                    h_counter.update_confidence(delta=0.4, evidence_summary=ev_text, is_supporting=False)
            else:
                fail_summary = (
                    "No actions or execution traces recorded."
                    if not traces
                    else f"No concrete evidence discovered across {len(traces)} execution traces."
                )
                h_primary.update_confidence(delta=0.4, evidence_summary=fail_summary, is_supporting=False)
                h_counter.update_confidence(delta=0.4, evidence_summary=fail_summary, is_supporting=True)

            state.active_hypotheses = [
                f"[{h_primary.status.value.upper()}] Primary ({h_primary.hypothesis_id}): {h_primary.statement} (confidence: {h_primary.confidence:.2f})",
                f"[{h_counter.status.value.upper()}] Counter ({h_counter.hypothesis_id}): {h_counter.statement} (confidence: {h_counter.confidence:.2f})",
            ]
            state.evidence.append(
                f"Hypothesis verdict: {h_primary.status.value.upper()} (primary: {h_primary.confidence:.2f}, counter: {h_counter.confidence:.2f})"
            )

            # 5. Finalize Deliverables — derived from the agent's real actions.
            await self.finalize(mission_id, traces=traces)

            # Outcome reflects reality: success only if the agent actually did
            # something that succeeded.
            state.current_phase = MissionPhase.COMPLETED
            success_ratio = (successful / total_actions) if total_actions else 0.0
            deliverables_count = len(self.deliverables.get(mission_id, []))

            if successful > 0 and success_ratio >= 0.60 and deliverables_count > 0:
                state.status = MissionStatus.COMPLETED
                state.outcome = MissionOutcome.SUCCESS
                state.current_next_action = (
                    f"Mission completed: {successful}/{total_actions} actions succeeded; "
                    f"{deliverables_count} deliverables produced."
                )
            elif successful > 0:
                state.status = MissionStatus.COMPLETED
                state.outcome = MissionOutcome.PARTIAL_SUCCESS
                state.current_next_action = (
                    f"Mission completed with partial success: {successful}/{total_actions} actions succeeded; "
                    f"{deliverables_count} deliverables produced."
                )
            else:
                state.status = MissionStatus.COMPLETED
                state.outcome = MissionOutcome.FAILED
                state.current_next_action = (
                    f"Mission ended with no successful actions ({total_actions} attempted)."
                )

            self._record_event(
                mission_id,
                "MissionCompleted",
                {"outcome": state.outcome.value, "traces": len(traces)},
            )
            await self._persist_mission(mission_id)

        except Exception as e:
            logger.error("mission_coordination_failed", mission_id=mission_id, error=str(e))
            if hasattr(state, "_h_primary") and hasattr(state, "_h_counter"):
                err_summary = f"Mission coordination error: {str(e)}"
                state._h_primary.update_confidence(delta=0.4, evidence_summary=err_summary, is_supporting=False)
                state._h_counter.update_confidence(delta=0.4, evidence_summary=err_summary, is_supporting=True)
                state.active_hypotheses = [
                    f"[{state._h_primary.status.value.upper()}] Primary ({state._h_primary.hypothesis_id}): {state._h_primary.statement} (confidence: {state._h_primary.confidence:.2f})",
                    f"[{state._h_counter.status.value.upper()}] Counter ({state._h_counter.hypothesis_id}): {state._h_counter.statement} (confidence: {state._h_counter.confidence:.2f})",
                ]
            state.status = MissionStatus.FAILED
            state.outcome = MissionOutcome.FAILED
            self._record_event(mission_id, "MissionFailed", {"error": str(e)})
            await self._persist_mission(mission_id)

        state.updated_at = _now()
        await self._persist_mission(mission_id)
        return state

    # =============================================================
    # 4. Dynamic Replanning & Self-Correction
    # =============================================================
    async def replan(self, mission_id: str, trigger: str, context: dict[str, Any]) -> MissionPlan:
        """Adapts the mission plan in response to new evidence or unexpected errors."""
        state = self._require_mission(mission_id)
        if not state.current_plan:
            return await self.decompose_mission(mission_id)

        plan = state.current_plan
        plan.version += 1
        plan.research_questions.append(f"Diagnostic Question: What caused trigger '{trigger}'?")
        plan.stop_conditions.append(f"Verify recovery from {trigger}")

        self._record_event(mission_id, "PlanReplanned", {"trigger": trigger, "version": plan.version, "context": context})
        logger.info("mission_replanned", mission_id=mission_id, trigger=trigger, version=plan.version)
        return plan

    # =============================================================
    # 5. Human Intervention Controls (Pause / Resume / Cancel)
    # =============================================================
    async def pause_mission(self, mission_id: str) -> bool:
        """Pause mission execution preserving state."""
        state = self._require_mission(mission_id)
        state.status = MissionStatus.PAUSED
        state.current_phase = MissionPhase.PAUSED
        state.current_next_action = "Mission paused by operator. Awaiting resume instruction."
        self._record_event(mission_id, "MissionPaused", {"actor": "operator"})
        await self._persist_mission(mission_id)
        return True

    async def resume_mission(self, mission_id: str) -> bool:
        """Resume a paused mission."""
        state = self._require_mission(mission_id)
        state.status = MissionStatus.ACTIVE
        state.current_phase = MissionPhase.ENGINEERING
        state.current_next_action = "Mission resumed. Continuing autonomous coordination."
        self._record_event(mission_id, "MissionResumed", {"actor": "operator"})
        await self._persist_mission(mission_id)
        return True

    async def cancel_mission(self, mission_id: str, reason: str = "Operator cancelled") -> bool:
        """Cancel mission execution."""
        state = self._require_mission(mission_id)
        state.status = MissionStatus.CANCELLED
        state.current_phase = MissionPhase.CANCELLED
        state.outcome = MissionOutcome.FAILED
        state.current_next_action = f"Mission cancelled: {reason}"
        self._record_event(mission_id, "MissionCancelled", {"reason": reason})
        await self._persist_mission(mission_id)
        return True

    # =============================================================
    # 6. Deliverables & Knowledge Summary
    # =============================================================
    async def finalize(self, mission_id: str, traces: list | None = None) -> list[MissionDeliverable]:
        """Generates validated final deliverables derived from the agent's traces.

        No traces (or no successful artifact-producing action) → empty list
        (honest: no fabricated deliverables). Re-derive on each call so a
        re-finalize after more actions stays accurate.
        """
        from sonic.computer_use.models import ComputerDecisionTrace
        from sonic.mission_engine.trace_synthesis import synthesize_deliverables

        state = self._require_mission(mission_id)
        # Prefer explicitly-passed traces; fall back to any previously stored.
        trace_list = traces or getattr(state, "_last_traces", []) or []
        deliverables = synthesize_deliverables(
            mission_id, state.objective.goal,
            [t for t in trace_list if isinstance(t, ComputerDecisionTrace)],
        )
        self.deliverables[mission_id] = deliverables
        self._record_event(mission_id, "DeliverablesFinalized", {"count": len(deliverables)})
        return deliverables

    def get_knowledge_summary(self, mission_id: str) -> MissionKnowledgeSummary:
        """Generates a structured knowledge snapshot derived from the agent's
        actual traces — not hardcoded vulnerability strings."""
        from sonic.mission_engine.trace_synthesis import synthesize_knowledge

        state = self._require_mission(mission_id)
        res_summary = self.resource_mgr.get_resource_summary(mission_id)
        traces = getattr(state, "_last_traces", []) or []
        summary = synthesize_knowledge(
            goal=state.objective.goal,
            traces=traces,
            remaining_unknowns=state.remaining_unknowns,
            active_tracks=state.active_tracks,
        )
        summary.current_hypotheses = list(state.active_hypotheses)
        if hasattr(state, "_h_primary") and state._h_primary is not None:
            summary.evidence.append(
                f"Hypothesis verdict: {state._h_primary.status.value.upper()} (primary: {state._h_primary.confidence:.2f}, counter: {state._h_counter.confidence:.2f})"
            )
        summary.resource_state = res_summary
        return summary

    # =============================================================
    # Helpers
    # =============================================================
    def _require_mission(self, mission_id: str) -> MissionState:
        state = self.missions.get(mission_id)
        if not state:
            raise KeyError(f"Mission '{mission_id}' not found.")
        return state

    def _record_event(self, mission_id: str, event_type: str, payload: dict[str, Any]) -> None:
        evt = MissionEvent(mission_id=mission_id, event_type=event_type, payload=payload)
        self.events.setdefault(mission_id, []).append(evt)

    def stash_traces(self, mission_id: str, traces: list[Any]) -> None:
        """Stash traces on the mission state for artifact synthesis and knowledge derivation."""
        state = self._require_mission(mission_id)
        object.__setattr__(state, "_last_traces", list(traces))

    def _is_complex_or_autonomous(self, objective: Any, plan: Any = None) -> bool:
        """Check if mission objective is complex/multi-step or autonomy_level >= L3_AUTONOMOUS."""
        from sonic.computer_use.models import ComputerAutonomyLevel

        autonomy_ranks = {
            ComputerAutonomyLevel.L0_MANUAL: 0,
            ComputerAutonomyLevel.L1_ASSISTED: 1,
            ComputerAutonomyLevel.L2_SUPERVISED_AUTONOMOUS: 2,
            ComputerAutonomyLevel.L3_AUTONOMOUS: 3,
            "L0_MANUAL": 0,
            "L1_ASSISTED": 1,
            "L2_SUPERVISED_AUTONOMOUS": 2,
            "L3_AUTONOMOUS": 3,
        }
        level_rank = autonomy_ranks.get(self.autonomy_level, 3 if "L3" in str(self.autonomy_level) else 0)
        if level_rank >= 3:
            return True

        goal_str = getattr(objective, "goal", str(objective)).lower()
        complex_markers = (
            " and ", " then ", ";", "\n", "step", "phase",
            "remediate", "investigate", "reproduce", "implement",
            "audit", "recon", "verify", "patch", "multi-step",
            "multistep", "end-to-end", "pipeline",
        )
        if any(m in goal_str for m in complex_markers):
            return True
        if len(goal_str.split()) > 6:
            return True
        if plan and getattr(plan, "milestones", None) and len(plan.milestones) > 1:
            return True
        return False
