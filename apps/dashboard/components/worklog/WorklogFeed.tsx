"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Terminal,
  FileCode,
  Send,
  Square,
  Sparkles,
  GitBranch,
  MoreHorizontal,
  Maximize2,
  PanelRightOpen,
  PanelRightClose,
  Bot,
  User,
  Shield,
  Copy,
  Check,
  Zap,
  Brain,
  MousePointer,
  Eye,
  Monitor,
  AppWindow,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Play,
  X,
  Loader2,
  ShieldAlert,
  Users,
  Code2,
} from "lucide-react";
import { WorklogItem, WorkstationWorker } from "../../types/workstation";
import { api } from "../../lib/api";
import { MarkdownText } from "./MarkdownText";
import { CommandBlock } from "./CommandBlock";
import { FileActionBlock } from "./FileActionBlock";

type Mode = "Normal" | "Autonomous" | "Pair-Program";
const MODES: Mode[] = ["Normal", "Autonomous", "Pair-Program"];

interface WorkerStep {
  step_index?: number | null;
  action_type?: string;
  status?: string;
  target?: string;
  thought?: string;
  summary?: string;
  timestamp?: string;
}

interface WorkerActivity {
  key: string;
  agentUid?: string;
  agentId?: string;
  subMissionId?: string;
  number?: number | null;
  name: string;
  role?: string;
  plane?: string;
  goal: string;
  status: string;
  isRunning?: boolean;
  success?: boolean | null;
  latestAction?: string;
  latestActionStatus?: string;
  latestThought?: string;
  heartbeat?: string;
  heartbeatAt?: string;
  findings?: string;
  keyDiscoveries?: string[];
  actionsTaken?: number | null;
  durationSeconds?: number | null;
  judgeVerdict?: string;
  judgeNotes?: string;
  evidenceVerified?: boolean | null;
  fileReferences?: { path: string; operations?: string[] }[];
  testResults?: {
    command?: string;
    passed?: number;
    failed?: number;
    errors?: number;
    exit_code?: number | null;
  }[];
  evidenceIncomplete?: boolean | null;
  step_count?: number;
  steps?: WorkerStep[];
}

function isWorkerDispatch(item: WorklogItem): boolean {
  return (
    item.orchestration_event === "sub_dispatch" ||
    item.title?.startsWith("Dispatching SubAgent") === true ||
    (item.title?.startsWith("Dispatching ") === true &&
      Boolean(item.agent_id || item.sub_mission_id))
  );
}

function isWorkerReport(item: WorklogItem): boolean {
  return (
    item.orchestration_event === "sub_report" ||
    (item.title?.includes("Report") === true &&
      Boolean(item.agent_id || item.sub_agent_id || item.sub_mission_id))
  );
}

function workerKey(item: WorklogItem): string | undefined {
  if (item.agent_uid || item.sub_agent_uid) {
    return `uid:${item.agent_uid || item.sub_agent_uid}`;
  }
  if (item.agent_id || item.sub_agent_id) {
    return `agent:${item.agent_id || item.sub_agent_id}`;
  }
  if (item.sub_mission_id) return `mission:${item.sub_mission_id}`;
  if (item.sub_agent_number !== undefined) return `number:${item.sub_agent_number}`;
  return undefined;
}

function summarizeWorkerAction(item: WorklogItem): string {
  const action = item.action_type || item.type || "Action";
  const target =
    item.file ||
    item.command ||
    item.content?.match(/Target:\s*([^\n]+)/)?.[1]?.trim() ||
    "";
  const label = action.replaceAll("_", " ").toLowerCase();
  return target ? `${label} · ${target}` : label;
}

function outcomeBadgeClass(status: string): string {
  switch (status.toUpperCase()) {
    case "COMPLETE":
      return "border-emerald-500/25 bg-emerald-500/10 text-emerald-400";
    case "PARTIAL":
    case "WAITING_FOR_HUMAN":
      return "border-amber-500/30 bg-amber-500/10 text-amber-300";
    case "BLOCKED":
    case "INCOMPLETE":
    case "FAILED":
    case "ERROR":
      return "border-rose-500/30 bg-rose-500/10 text-rose-400";
    default:
      return "border-ink-700 bg-ink-850 text-muted-bright";
  }
}

function buildWorkerActivities(worklog: WorklogItem[]): WorkerActivity[] {
  const workers = new Map<string, WorkerActivity>();

  for (const item of worklog) {
    if (isWorkerDispatch(item)) {
      const key = workerKey(item);
      if (!key) continue;
      workers.set(key, {
        key,
        agentUid: item.agent_uid || item.sub_agent_uid,
        agentId: item.agent_id || item.sub_agent_id,
        subMissionId: item.sub_mission_id,
        number: item.sub_agent_number,
        name:
          item.agent_name ||
          item.sub_agent_name ||
          (item.sub_agent_number !== undefined
            ? `Worker ${item.sub_agent_number}`
            : "Worker"),
        role: item.worker_role,
        plane: item.execution_plane,
        goal:
          item.goal ||
          item.content?.match(/Goal:\s*([^\n]+)/)?.[1]?.trim() ||
          "Task details unavailable",
        status: item.status || "RUNNING",
      });
      continue;
    }

    if (isWorkerReport(item)) {
      const key = workerKey(item);
      const worker = key ? workers.get(key) : undefined;
      if (!worker) continue;
      worker.status =
        item.status ||
        (item.success === true ? "COMPLETED" : item.success === false ? "FAILED" : "UNVERIFIED");
      worker.success = item.success;
      worker.findings = item.findings_summary || item.content || "";
      worker.keyDiscoveries = item.key_discoveries || [];
      worker.actionsTaken = item.actions_taken;
      worker.durationSeconds = item.duration_seconds;
      worker.judgeVerdict = item.judge_verdict || undefined;
      worker.judgeNotes = item.judge_notes || undefined;
      worker.evidenceVerified = item.evidence_verified;
      worker.fileReferences = item.file_references || [];
      worker.testResults = item.test_results || [];
      worker.evidenceIncomplete = item.evidence_incomplete;
      continue;
    }

    // A step/heartbeat belongs to whichever worker key it carries. The route
    // records `agent_uid`/`agent_id` on some events and `sub_agent_uid`/
    // `sub_agent_id` on others; guarding on only the `sub_agent_*` pair dropped
    // every `agent_uid`-keyed step, so the operator's drill-down showed a worker
    // with no live activity. Reuse the same key resolution as the dispatch path.
    const stepKey = workerKey(item);
    if (!stepKey) continue;
    const worker = workers.get(stepKey);
    if (!worker) continue;
    if (item.orchestration_event === "sub_heartbeat") {
      // Liveness only. The server projection keeps this separate from the real
      // thinking so a "still reasoning" ping cannot hide what the worker thought.
      worker.heartbeat = item.content || worker.heartbeat;
      continue;
    }
    if (item.type === "thought") {
      worker.latestThought = item.content || worker.latestThought;
      continue;
    }
    worker.latestAction = summarizeWorkerAction(item);
    worker.latestActionStatus = item.action_status || (item.type === "error" ? "FAILED" : undefined);
  }

  return Array.from(workers.values());
}

function workersFromServer(
  serverWorkers: WorkstationWorker[] | undefined,
  fallback: WorkerActivity[],
): WorkerActivity[] {
  if (!serverWorkers || serverWorkers.length === 0) return fallback;
  return serverWorkers.map((worker) => ({
    key: worker.agent_uid ? `uid:${worker.agent_uid}` : worker.key,
    agentUid: worker.agent_uid,
    agentId: worker.agent_id,
    subMissionId: worker.sub_mission_id,
    number: worker.sub_agent_number,
    name:
      worker.agent_name ||
      (worker.sub_agent_number != null ? `Worker ${worker.sub_agent_number}` : "Worker"),
    role: worker.role,
    plane: worker.plane,
    goal: worker.goal || "Task details unavailable",
    status: worker.status || "RUNNING",
    isRunning: worker.is_running,
    success: worker.success,
    latestAction: worker.latest_action,
    latestActionStatus: worker.latest_action_status,
    latestThought: worker.latest_thought,
    heartbeat: worker.heartbeat,
    heartbeatAt: worker.heartbeat_at,
    findings: worker.findings_summary,
    keyDiscoveries: worker.key_discoveries || [],
    actionsTaken: worker.actions_taken,
    durationSeconds: worker.duration_seconds,
    judgeVerdict: worker.judge_verdict || undefined,
    judgeNotes: worker.judge_notes || undefined,
    evidenceVerified: worker.evidence_verified,
    fileReferences: worker.file_references || [],
    testResults: worker.test_results || [],
    evidenceIncomplete: worker.evidence_incomplete,
    step_count: worker.step_count,
    steps: worker.steps,
  }));
}

function workerStatusLabel(worker: WorkerActivity, missionRunning: boolean): string {
  const status = worker.status.toUpperCase();
  if (status === "INTERRUPTED") return "INTERRUPTED";
  if (worker.isRunning === true) return "RUNNING";
  // The server explicitly says this worker is not running; never re-derive a
  // stale RUNNING from an old dispatch event.
  if (worker.isRunning === false && (status === "RUNNING" || status === "PENDING")) {
    return "UNCONFIRMED";
  }
  if (missionRunning && ["RUNNING", "PENDING"].includes(status)) return "RUNNING";
  if (status === "WAITING_FOR_HUMAN") return "WAITING FOR HUMAN";
  if (["BLOCKED", "TIMED_OUT", "FAILED", "PAUSED", "PARTIAL", "INCOMPLETE"].includes(status)) {
    return status;
  }
  if (status === "RUNNING" || status === "PENDING") return "UNCONFIRMED";
  if (worker.success === true) {
    return worker.evidenceVerified === true && worker.judgeVerdict?.toUpperCase() === "CONFIRMED"
      ? "COMPLETED"
      : "UNVERIFIED";
  }
  if (worker.success === false) return "FAILED";
  return status;
}

function workerStatusClass(status: string): string {
  if (status === "COMPLETED" || status === "SUCCESS") {
    return "border-emerald-500/25 bg-emerald-500/10 text-emerald-400";
  }
  if (status === "RUNNING") {
    return "border-secondary-500/25 bg-secondary-500/10 text-secondary-300";
  }
  if (["FAILED", "BLOCKED", "TIMED_OUT", "INCOMPLETE"].includes(status)) {
    return "border-rose-500/25 bg-rose-500/10 text-rose-400";
  }
  if (status === "WAITING FOR HUMAN" || status === "UNVERIFIED" || status === "PAUSED") {
    return "border-amber-500/25 bg-amber-500/10 text-amber-300";
  }
  return "border-amber-500/25 bg-amber-500/10 text-amber-300";
}

function WorkerDetail({
  worker,
  sessionId,
}: {
  worker: WorkerActivity;
  sessionId: string;
}) {
  // The card is rendered from worklog events, which can lag or be truncated.
  // Expanding it asks the backend for the authoritative record of this worker
  // so the operator sees what it is doing now, not what it did when the last
  // event arrived. The worklog-derived view stays as the fallback, so a failed
  // fetch degrades to what we already had instead of an empty panel.
  const [live, setLive] = useState<WorkerActivity | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [fetching, setFetching] = useState(false);

  const agentUid = worker.agentUid;

  useEffect(() => {
    if (!agentUid) return;
    let cancelled = false;
    setFetching(true);
    setFetchError(null);
    api
      .getWorkstationAgent(agentUid, sessionId)
      .then((response: { worker?: WorkstationWorker }) => {
        if (cancelled) return;
        const record = response?.worker;
        if (!record) {
          setFetchError("Backend returned no record for this worker.");
          return;
        }
        setLive(workersFromServer([record], [])[0] || null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setFetchError(err instanceof Error ? err.message : "Could not load live worker state.");
      })
      .finally(() => {
        if (!cancelled) setFetching(false);
      });
    return () => {
      cancelled = true;
    };
  }, [agentUid, sessionId]);

  const view = live || worker;
  const steps = view.steps || [];
  return (
    <div className="mt-2 space-y-2 border-t border-ink-750 pt-2">
      <div className="flex items-center gap-1.5 text-[9px] text-muted-dim">
        {fetching ? (
          <>
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Loading live worker state…</span>
          </>
        ) : live ? (
          <span className="rounded border border-emerald-800 bg-emerald-950 px-1.5 py-0.5 text-emerald-300">
            live
          </span>
        ) : null}
        {fetchError && (
          <span className="rounded border border-amber-800 bg-amber-950 px-1.5 py-0.5 text-amber-300">
            {fetchError} · showing last known events
          </span>
        )}
      </div>
      <div>
        <p className="text-[9px] uppercase tracking-wide text-muted-dim">Goal</p>
        <p className="mt-0.5 text-[10.5px] leading-relaxed text-slate-200">{view.goal}</p>
      </div>
      {view.latestAction && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Current activity</p>
          <p className="mt-0.5 text-[10.5px] leading-relaxed text-slate-200">
            {view.latestAction}
            {view.latestActionStatus && (
              <span className="ml-1 text-[9.5px] text-muted-dim">
                {view.latestActionStatus.toLowerCase()}
              </span>
            )}
          </p>
        </div>
      )}
      {view.latestThought && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Latest thinking</p>
          <p className="mt-0.5 line-clamp-4 whitespace-pre-wrap text-[10.5px] leading-relaxed text-slate-300">
            {view.latestThought}
          </p>
        </div>
      )}
      {view.heartbeat && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Liveness</p>
          <p className="mt-0.5 text-[10px] leading-relaxed text-muted-bright">
            {view.heartbeat}
          </p>
        </div>
      )}
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[9.5px] text-muted-dim">
        {view.agentUid && <span>uid: {view.agentUid}</span>}
        {view.agentId && <span>agent: {view.agentId}</span>}
        {view.subMissionId && <span>mission: {view.subMissionId}</span>}
        {view.step_count !== undefined && <span>{view.step_count} steps</span>}
      </div>
      {steps.length > 0 && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Recent activity</p>
          <ol className="mt-1 space-y-1">
            {steps.slice(-8).map((step, index) => (
              <li
                key={`${worker.key}-step-${index}`}
                className="flex items-start gap-1.5 text-[10px] text-slate-300"
              >
                <span className="mt-0.5 shrink-0 rounded border border-ink-700 bg-ink-850 px-1 text-[8.5px] text-muted-dim">
                  {step.step_index ?? index + 1}
                </span>
                <span className="min-w-0">
                  <span className="font-medium text-slate-200">{step.summary || step.action_type}</span>
                  {step.status && (
                    <span className="ml-1 text-[9px] text-muted-dim">
                      {String(step.status).toLowerCase()}
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
      {view.judgeVerdict && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Independent Judge</p>
          <p className="mt-0.5 text-[10.5px] text-amber-300">
            {view.judgeVerdict}
            {view.evidenceVerified === true
              ? " · evidence confirmed"
              : view.judgeNotes
                ? ` · ${view.judgeNotes}`
                : " · evidence not confirmed"}
          </p>
        </div>
      )}
      {view.keyDiscoveries && view.keyDiscoveries.length > 0 && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Discoveries</p>
          <ul className="mt-1 space-y-0.5">
            {view.keyDiscoveries.map((discovery, index) => (
              <li key={`${worker.key}-disc-${index}`} className="text-[10px] text-secondary-300">
                {discovery}
              </li>
            ))}
          </ul>
        </div>
      )}
      {(view.fileReferences?.length || view.testResults?.length) ? (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
            Verifiable evidence
          </p>
          {view.fileReferences && view.fileReferences.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {view.fileReferences.map((ref, index) => (
                <li
                  key={`${worker.key}-file-${index}`}
                  className="flex items-center gap-1.5 text-[10px] text-slate-300"
                >
                  <span className="shrink-0 rounded border border-ink-700 bg-ink-850 px-1 text-[8.5px] uppercase text-muted-dim">
                    {(ref.operations || []).join("/") || "file"}
                  </span>
                  <span className="min-w-0 truncate font-mono text-[9.5px]">{ref.path}</span>
                </li>
              ))}
            </ul>
          )}
          {view.testResults && view.testResults.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {view.testResults.map((test, index) => {
                const green =
                  (test.failed || 0) === 0 &&
                  (test.errors || 0) === 0 &&
                  (test.exit_code ?? 0) === 0;
                return (
                  <li
                    key={`${worker.key}-test-${index}`}
                    className="flex items-center gap-1.5 text-[10px]"
                  >
                    <span
                      className={`shrink-0 rounded border px-1 text-[8.5px] uppercase ${
                        green
                          ? "border-emerald-800 bg-emerald-950 text-emerald-300"
                          : "border-rose-800 bg-rose-950 text-rose-300"
                      }`}
                    >
                      {green ? "pass" : "fail"}
                    </span>
                    <span className="min-w-0 truncate text-slate-300">
                      {test.passed ?? 0} passed
                      {(test.failed || 0) > 0 ? ` · ${test.failed} failed` : ""}
                      {(test.errors || 0) > 0 ? ` · ${test.errors} errors` : ""}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ) : view.evidenceIncomplete ? (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
            Verifiable evidence
          </p>
          <p className="mt-0.5 text-[10.5px] text-rose-300">
            Worker claimed success but produced no file, test, or discovery
            evidence — treat as unproven.
          </p>
        </div>
      ) : null}
      {view.findings && (
        <div>
          <p className="text-[9px] uppercase tracking-wide text-muted-dim">Report</p>
          <p className="mt-0.5 whitespace-pre-wrap text-[10.5px] leading-relaxed text-slate-300">
            {view.findings}
          </p>
        </div>
      )}
    </div>
  );
}

function WorkerBoard({
  workers,
  missionRunning,
  sessionId,
}: {
  workers: WorkerActivity[];
  missionRunning: boolean;
  sessionId: string;
}) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  if (workers.length === 0) return null;
  // Count with the same function the cards render, so the header can never
  // disagree with a card (e.g. a raw RUNNING worker shown as UNCONFIRMED).
  const runningCount = workers.filter(
    (worker) => workerStatusLabel(worker, missionRunning) === "RUNNING",
  ).length;

  return (
    <section
      aria-label="Delegated worker progress"
      className="my-2.5 rounded-xl border border-ink-750 bg-ink-850/70 p-3"
    >
      <div className="mb-2.5 flex items-center gap-2">
        <Users className="h-4 w-4 shrink-0 text-secondary-400" />
        <h2 className="text-[12px] font-semibold text-slate-100">Parallel work</h2>
        <span className="text-[10px] text-muted-dim">
          {runningCount > 0 ? `${runningCount} active · ` : ""}
          {workers.length} {workers.length === 1 ? "worker" : "workers"}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
        {workers.map((worker) => {
          const statusLabel = workerStatusLabel(worker, missionRunning);
          const statusClass = workerStatusClass(statusLabel);
          const PlaneIcon = worker.plane?.toLowerCase() === "headless" ? Code2 : Monitor;
          const expanded = expandedKey === worker.key;

          return (
            <article
              key={worker.key}
              className={`min-w-0 rounded-lg border bg-ink-900/80 p-2.5 ${
                expanded ? "border-secondary-500/40" : "border-ink-750"
              }`}
            >
              <button
                type="button"
                onClick={() => setExpandedKey(expanded ? null : worker.key)}
                aria-expanded={expanded}
                className="flex w-full items-start gap-2 text-left"
              >
                <PlaneIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-bright" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="truncate text-[11.5px] font-semibold text-slate-100">
                      {worker.name}
                    </span>
                    {worker.agentUid && (
                      <span className="truncate rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 font-mono text-[9px] text-muted-dim">
                        {worker.agentUid}
                      </span>
                    )}
                    {worker.role && worker.role.toLowerCase() !== "subagent" && (
                      <span className="rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 text-[9px] text-muted-bright">
                        {worker.role}
                      </span>
                    )}
                    {worker.plane && (
                      <span className="rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 text-[9px] uppercase text-muted-dim">
                        {worker.plane === "headless"
                          ? "Terminal / Code"
                          : worker.plane === "dual"
                            ? "Desktop + Terminal"
                            : "Computer / GUI"}
                      </span>
                    )}
                    <span className={`ml-auto rounded border px-1.5 py-0.5 text-[9px] font-medium ${statusClass}`}>
                      {statusLabel}
                    </span>
                    {expanded ? (
                      <ChevronDown className="h-3 w-3 shrink-0 text-muted-dim" />
                    ) : (
                      <ChevronRight className="h-3 w-3 shrink-0 text-muted-dim" />
                    )}
                  </div>
                  <p className="mt-1 line-clamp-2 text-[10.5px] leading-relaxed text-muted-bright">
                    {worker.goal}
                  </p>
                  {worker.latestAction && (
                    <p className="mt-1 truncate text-[10px] text-muted-dim">
                      Latest: {worker.latestAction}
                      {worker.latestActionStatus &&
                        ` · ${worker.latestActionStatus.toLowerCase()}`}
                    </p>
                  )}
                  {!expanded && worker.findings && (
                    <p className="mt-1 line-clamp-2 text-[10px] text-slate-300">
                      {worker.findings}
                    </p>
                  )}
                  {!expanded && worker.judgeVerdict && (
                    <p className="mt-1 text-[10px] text-amber-300">
                      Judge: {worker.judgeVerdict}
                      {worker.evidenceVerified === true
                        ? " · evidence confirmed"
                        : worker.judgeNotes
                          ? ` · ${worker.judgeNotes}`
                          : " · evidence not confirmed"}
                    </p>
                  )}
                  {!expanded && worker.keyDiscoveries && worker.keyDiscoveries.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {worker.keyDiscoveries.slice(0, 3).map((discovery, index) => (
                        <span
                          key={`${worker.key}-${index}`}
                          className="max-w-full truncate rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 text-[9px] text-secondary-300"
                        >
                          {discovery}
                        </span>
                      ))}
                    </div>
                  )}
                  {worker.actionsTaken !== undefined && worker.actionsTaken !== null && (
                    <p className="mt-1 text-[9px] text-muted-dim">
                      {worker.actionsTaken} recorded {worker.actionsTaken === 1 ? "action" : "actions"}
                    </p>
                  )}
                  {!expanded && worker.evidenceIncomplete && (
                    <p className="mt-1 inline-flex rounded border border-rose-800 bg-rose-950 px-1.5 py-0.5 text-[9px] text-rose-300">
                      success claimed without verifiable evidence
                    </p>
                  )}
                  {worker.durationSeconds !== undefined &&
                    worker.durationSeconds !== null &&
                    worker.durationSeconds > 0 && (
                      <p className="mt-1 text-[9px] text-muted-dim">
                        Finished in {Math.max(1, Math.round(worker.durationSeconds))}s
                      </p>
                    )}
                </div>
              </button>
              {expanded && <WorkerDetail worker={worker} sessionId={sessionId} />}
            </article>
          );
        })}
      </div>
    </section>
  );
}

interface WorklogFeedProps {
  worklog: WorklogItem[];
  currentAction?: string;
  loading: boolean;
  status?: string;
  waitingForHuman?: boolean;
  waitingForHumanReason?: string;
  onSendPrompt: (prompt: string, mode: Mode) => Promise<void>;
  onInterrupt?: () => Promise<boolean | void> | boolean | void;
  onSelectFile?: (filePath: string) => void;
  sessionName?: string;
  gitBranch?: string;
  rightPanelOpen?: boolean;
  onToggleRightPanel?: () => void;
  sessionId?: string;
  onRefreshState?: () => void;
  workers?: WorkstationWorker[];
  missionTaskAlive?: boolean;
}

function ProbeProposalCard({
  item,
  sessionId = "default",
  onActionComplete,
}: {
  item: WorklogItem;
  sessionId?: string;
  onActionComplete?: () => void;
}) {
  const [loadingActionId, setLoadingActionId] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, { status: string; output?: string }>>({});
  const [error, setError] = useState<string | null>(null);

  const actions: any[] =
    item.proposed_actions && item.proposed_actions.length > 0
      ? item.proposed_actions
      : (item as any).action_id
      ? [{ action_id: (item as any).action_id, tool: item.title, input: { tool: "probe", target: item.content } }]
      : [];

  const handleApprove = async (actionId: string) => {
    setLoadingActionId(actionId);
    setError(null);
    try {
      const res = await api.approveMissionProbe(actionId, true, sessionId);
      setResults((prev) => ({
        ...prev,
        [actionId]: { status: res?.status || "SUCCESS", output: res?.output },
      }));
      onActionComplete?.();
    } catch (err: any) {
      setError(err?.message || "Failed to execute approved probe");
    } finally {
      setLoadingActionId(null);
    }
  };

  const handleDismiss = async (actionId: string) => {
    setLoadingActionId(actionId);
    setError(null);
    try {
      const res = await api.approveMissionProbe(actionId, false, sessionId);
      setResults((prev) => ({
        ...prev,
        [actionId]: { status: "dismissed" },
      }));
      onActionComplete?.();
    } catch (err: any) {
      setError(err?.message || "Failed to dismiss probe");
    } finally {
      setLoadingActionId(null);
    }
  };

  return (
    <div className="my-2.5 rounded-xl border border-amber-500/40 bg-amber-500/5 p-3.5 text-xs shadow-md">
      <div className="flex items-center gap-2 mb-2">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
        <span className="font-semibold text-slate-100 text-[12.5px]">
          {item.title || "Operator Approval Required: Active Security Probes"}
        </span>
        <span className="ml-auto px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-400 font-mono font-semibold text-[10px]">
          OPERATOR-GATED
        </span>
      </div>

      {item.content && (
        <p className="text-[11.5px] text-slate-300 leading-relaxed mb-3 whitespace-pre-wrap">
          {item.content}
        </p>
      )}

      {error && (
        <div className="mb-2.5 p-2 rounded bg-rose-500/10 border border-rose-500/30 text-rose-400 text-[11px] flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {actions.length > 0 && (
        <div className="space-y-2">
          {actions.map((act: any, idx: number) => {
            const actId = act.action_id || `probe-${idx}`;
            const toolName = act.input?.tool || act.tool || "probe";
            const targetHost = act.input?.target || act.target || "";
            const options = act.input?.options || {};
            const preview =
              act.command ||
              (options.args ? `${toolName} ${options.args} ${targetHost}` : `${toolName} ${targetHost}`);
            const resultState = results[actId];
            const isProcessing = loadingActionId === actId;

            return (
              <div
                key={actId}
                className="p-2.5 rounded-lg bg-ink-900/90 border border-ink-750 flex flex-col gap-2"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="px-2 py-0.5 rounded bg-secondary-500/20 border border-secondary-500/40 text-secondary-300 font-mono font-semibold text-[11px] uppercase">
                      {toolName}
                    </span>
                    <span className="text-slate-200 font-mono text-[11.5px] truncate" title={targetHost}>
                      Target: <span className="text-cyan-400 font-mono">{targetHost}</span>
                    </span>
                  </div>

                  {resultState ? (
                    <span
                      className={`px-2 py-0.5 rounded font-mono text-[10px] font-medium ${
                        resultState.status === "dismissed"
                          ? "bg-ink-800 text-muted-dim border border-ink-700"
                          : resultState.status === "SUCCESS"
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                          : "bg-rose-500/20 text-rose-400 border border-rose-500/40"
                      }`}
                    >
                      {resultState.status === "dismissed" ? "DISMISSED" : resultState.status}
                    </span>
                  ) : (
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        type="button"
                        disabled={isProcessing}
                        onClick={() => handleApprove(actId)}
                        className="px-2.5 py-1 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-[11px] flex items-center gap-1 transition shadow-sm disabled:opacity-50"
                      >
                        {isProcessing ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Play className="w-3 h-3 fill-current" />
                        )}
                        <span>Approve &amp; Run</span>
                      </button>
                      <button
                        type="button"
                        disabled={isProcessing}
                        onClick={() => handleDismiss(actId)}
                        className="px-2 py-1 rounded-md bg-ink-800 hover:bg-ink-750 text-slate-300 hover:text-white border border-ink-700 text-[11px] flex items-center gap-1 transition disabled:opacity-50"
                      >
                        <X className="w-3 h-3" />
                        <span>Dismiss</span>
                      </button>
                    </div>
                  )}
                </div>

                {preview && (
                  <div className="font-mono text-[10.5px] text-muted-bright bg-ink-950 px-2 py-1 rounded border border-ink-800 truncate">
                    <code>$ {preview}</code>
                  </div>
                )}

                {resultState?.output && (
                  <div className="mt-1 font-mono text-[10.5px] text-slate-300 bg-ink-950 p-2 rounded border border-ink-800 max-h-32 overflow-y-auto whitespace-pre-wrap">
                    {resultState.output}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function OperatorActionApprovalCard({
  item,
  sessionId = "default",
}: {
  item: WorklogItem;
  sessionId?: string;
}) {
  const [resolution, setResolution] = useState<string>(
    item.approval_status && item.approval_status !== "pending"
      ? item.approval_status
      : "",
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resolve = async (approved: boolean) => {
    if (!item.approval_id) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.approveWorkstationAction(
        item.approval_id,
        approved,
        sessionId,
      );
      setResolution(result?.status || (approved ? "approved" : "rejected"));
    } catch (err: any) {
      setError(err?.message || "Could not resolve this action approval");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="my-2.5 rounded-xl border border-amber-500/40 bg-amber-500/5 p-3.5 text-xs shadow-md">
      <div className="flex items-center gap-2 mb-2">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
        <span className="font-semibold text-slate-100 text-[12.5px]">
          {item.title || "Operator approval required"}
        </span>
        <span className="ml-auto px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-400 font-mono font-semibold text-[10px]">
          {resolution ? resolution.toUpperCase() : "PAUSED"}
        </span>
      </div>
      {item.content && (
        <p className="text-[11.5px] text-slate-300 leading-relaxed mb-3 whitespace-pre-wrap">
          {item.content}
        </p>
      )}
      {error && (
        <div role="alert" className="mb-2.5 p-2 rounded bg-rose-500/10 border border-rose-500/30 text-rose-400 text-[11px]">
          {error}
        </div>
      )}
      {!resolution && (
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            disabled={loading}
            onClick={() => resolve(true)}
            className="px-2.5 py-1 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-[11px] flex items-center gap-1 transition shadow-sm disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
            <span>Approve this action</span>
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={() => resolve(false)}
            className="px-2 py-1 rounded-md bg-ink-800 hover:bg-ink-750 text-slate-300 hover:text-white border border-ink-700 text-[11px] flex items-center gap-1 transition disabled:opacity-50"
          >
            <X className="w-3 h-3" />
            <span>Reject</span>
          </button>
        </div>
      )}
    </div>
  );
}

function PhaseCompleteBanner({ item }: { item: WorklogItem }) {
  return (
    <div className="my-3 flex items-center gap-2 rounded-xl bg-ink-850 border border-secondary-500/30 px-3 py-2 text-xs">
      <Check className="w-4 h-4 text-secondary-400 shrink-0" />
      <span className="font-semibold text-slate-100">{item.title}</span>
      <span className="text-muted-dim text-[11px] ml-auto font-mono">{item.content}</span>
    </div>
  );
}

export function WorklogFeed({
  worklog,
  currentAction,
  loading,
  status,
  waitingForHuman,
  waitingForHumanReason,
  onSendPrompt,
  onInterrupt,
  onSelectFile,
  sessionName = "",
  gitBranch = "",
  rightPanelOpen = true,
  onToggleRightPanel,
  sessionId = "default",
  onRefreshState,
  workers,
  missionTaskAlive,
}: WorklogFeedProps) {
  const [promptText, setPromptText] = useState("");
  const [activeMode, setActiveMode] = useState<Mode>("Normal");
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [resumeNote, setResumeNote] = useState("");
  const [resuming, setResuming] = useState(false);
  const worklogEndRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const shouldFollowTailRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const displayItems: WorklogItem[] = worklog || [];
  let latestObjectiveIndex = -1;
  for (let index = displayItems.length - 1; index >= 0; index -= 1) {
    if (displayItems[index].title === "Objective Received") {
      latestObjectiveIndex = index;
      break;
    }
  }
  const currentRunItems =
    latestObjectiveIndex >= 0 ? displayItems.slice(latestObjectiveIndex) : displayItems;
  const workerActivities = workersFromServer(
    workers,
    buildWorkerActivities(currentRunItems),
  );
  const latestMissionOutcome = [...currentRunItems]
    .reverse()
    .find((item) => Boolean(item.outcome_status));
  const missionOutcomeStatus = latestMissionOutcome?.outcome_status?.toUpperCase();

  const normalizedStatus = (status || "").toUpperCase();
  const isWaitingForHuman = Boolean(waitingForHuman || normalizedStatus === "WAITING_FOR_HUMAN");
  const isPaused = normalizedStatus === "PAUSED";
  const isFinished =
    normalizedStatus === "IDLE" ||
    normalizedStatus === "PAUSED" ||
    normalizedStatus === "WAITING_FOR_HUMAN" ||
    normalizedStatus === "BLOCKED" ||
    normalizedStatus === "COMPLETED" ||
    normalizedStatus === "ERROR" ||
    (!loading && normalizedStatus !== "RUNNING");

  const handleResumeWorkstation = async () => {
    setResuming(true);
    try {
      await api.resumeWorkstation(resumeNote, sessionId);
      setResumeNote("");
      onRefreshState?.();
    } catch (err) {
      console.error("Failed to resume workstation:", err);
    } finally {
      setResuming(false);
    }
  };

  // Only backend lifecycle state can end the active work indicator. A response
  // event is content, not proof that execution has completed.
  const shouldShowAction = !isFinished;

  const cleanActionText = (() => {
    if (!shouldShowAction) return undefined;
    const text = (currentAction || "").trim();
    if (!text) {
      return loading || normalizedStatus === "RUNNING"
        ? "Thinking..."
        : undefined;
    }
    const lower = text.toLowerCase();
    if (
      lower.includes("queued") ||
      lower.includes("reasoning queued") ||
      lower.includes("autonomous reasoning") ||
      lower.startsWith("idle") ||
      lower.startsWith("ready") ||
      lower === "thinking" ||
      lower.startsWith("thinking:")
    ) {
      return normalizedStatus === "RUNNING" || loading
        ? "Thinking..."
        : undefined;
    }
    return text;
  })();

  useEffect(() => {
    const timeline = timelineRef.current;
    if (!timeline || !shouldFollowTailRef.current) return;
    // Keep live output readable without overriding an operator who is
    // reviewing an earlier event in the same conversation.
    timeline.scrollTop = timeline.scrollHeight;
  }, [worklog, cleanActionText]);

  useEffect(() => {
    // A newly selected conversation should start at its latest event.
    shouldFollowTailRef.current = true;
  }, [sessionId]);

  const handleTimelineScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const timeline = event.currentTarget;
    const distanceFromBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight;
    // Once the operator scrolls away from the tail, pause auto-follow. It is
    // re-enabled only when they deliberately return to the bottom.
    shouldFollowTailRef.current = distanceFromBottom <= 24;
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!promptText.trim() || (loading && normalizedStatus === "RUNNING")) return;
    const text = promptText;
    setPromptText("");
    await onSendPrompt(text, activeMode);
  };

  const handleCopyResponse = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // ignore
    }
  };


  return (
    <div className="flex flex-col h-full bg-ink-900 text-slate-200 overflow-hidden font-sans">
      {/* Session Header */}
      <div className="h-10 border-b border-ink-800 bg-ink-900 px-4 flex items-center justify-between text-xs shrink-0 select-none">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-semibold text-white tracking-wide truncate text-[13px]">
            {sessionName || "New conversation"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {gitBranch && (
            <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-success/10 text-success border border-success/40 text-[11px] font-mono font-medium">
              <GitBranch className="w-3 h-3 text-success" />
              <span>{gitBranch}</span>
            </div>
          )}
          <button
            type="button"
            className="p-1 rounded text-muted hover:text-white hover:bg-ink-800 transition"
            title="Session options"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
          <button
            type="button"
            className="p-1 rounded text-muted hover:text-white hover:bg-ink-800 transition"
            title="Expand stream"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          {onToggleRightPanel && (
            <button
              type="button"
              onClick={onToggleRightPanel}
              className="p-1 rounded text-muted hover:text-white hover:bg-ink-800 transition"
              title={rightPanelOpen ? "Hide Computer panel" : "Show Computer panel"}
              aria-label={rightPanelOpen ? "Hide Computer panel" : "Show Computer panel"}
            >
              {rightPanelOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      {/* Main Process Timeline */}
      <div
        ref={timelineRef}
        onScroll={handleTimelineScroll}
        className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-4 py-3 space-y-2 text-xs"
      >
        {(normalizedStatus === "ERROR" || normalizedStatus === "BLOCKED") && (
          <div role="alert" className="mb-3 rounded-lg border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger">
            <strong>{normalizedStatus === "BLOCKED" ? "Workstation blocked" : "SONIC could not complete this task"}</strong>
            <p className="mt-1 text-xs">{currentAction || "Review the latest error and worklog before retrying."}</p>
          </div>
        )}
        {latestMissionOutcome &&
          missionOutcomeStatus !== "COMPLETE" &&
          missionOutcomeStatus !== "WAITING_FOR_HUMAN" && (
          <div
            role="status"
            className={`mb-3 rounded-lg border px-3 py-2 text-sm ${outcomeBadgeClass(missionOutcomeStatus || "INCOMPLETE")}`}
          >
            <strong>Mission outcome: {missionOutcomeStatus || "INCOMPLETE"}</strong>
            <p className="mt-1 whitespace-pre-wrap text-xs">
              {latestMissionOutcome.findings_summary ||
                latestMissionOutcome.content ||
                "The objective was not fully verified."}
            </p>
          </div>
        )}
        {displayItems.length === 0 ? (
          <div className="h-full min-h-[320px] flex flex-col items-center justify-center text-center p-6 text-muted-dim space-y-2 select-none">
            <div className="w-10 h-10 rounded-full bg-ink-850 border border-ink-800 flex items-center justify-center text-secondary-400 font-mono text-sm shadow-inner">
              &gt;_
            </div>
            <p className="text-xs text-slate-300 font-mono font-medium">Ready</p>
            <p className="text-[11.5px] text-muted-dim max-w-xs leading-relaxed">
              Type an objective or instruction below to operate the graphical workstation.
            </p>
          </div>
        ) : (
          <>
            <WorkerBoard
              workers={workerActivities}
              missionRunning={missionTaskAlive ?? !isFinished}
              sessionId={sessionId}
            />
            {displayItems.map((item, idx) => {
            const itemId = item.id || `item-${idx}`;

            if (isWorkerDispatch(item) || isWorkerReport(item)) return null;

            if (item.approval_id) {
              return (
                <OperatorActionApprovalCard
                  key={itemId}
                  item={item}
                  sessionId={sessionId}
                />
              );
            }

            // 0. Operator Gate / Active Probe Approval Proposals
            const isApprovalProposal =
              item.type === "approval" ||
              (item.proposed_actions && item.proposed_actions.length > 0) ||
              item.title?.toLowerCase().includes("probe proposal") ||
              item.title?.toLowerCase().includes("awaiting operator approval");

            if (isApprovalProposal) {
              return (
                <ProbeProposalCard
                  key={itemId}
                  item={item}
                  sessionId={sessionId}
                  onActionComplete={onRefreshState}
                />
              );
            }

            // Keep internal reasoning out of the activity feed.
            if (item.type === "thought" || item.title === "Thinking") {
              return null;
            }

            // 2. Terminal Shell Command (matching Devin's code pill)
            const isTerminalExec =
              item.type === "command" ||
              Boolean(item.command) ||
              item.title?.includes("TERMINAL_EXEC") ||
              item.title?.toLowerCase().includes("terminal_exec");

            if (isTerminalExec) {
              let cmdText = item.command || "";
              let outText = item.output || item.content || "";
              if (!cmdText) {
                const targetMatch = item.content?.match(/Target:\s*(\{[^}]+\}|[^\n]+)/);
                if (targetMatch) {
                  const rawTarget = targetMatch[1].trim();
                  try {
                    const parsed = JSON.parse(rawTarget);
                    cmdText = parsed.command || rawTarget;
                  } catch {
                    cmdText = rawTarget.replace(/^\{['"]command['"]:\s*['"](.*)['"]\}$/, "$1");
                  }
                } else {
                  cmdText = item.title?.replace(/^Step \d+:\s*/, "") || "command";
                }
              }
              return (
                <CommandBlock
                  key={itemId}
                  command={cmdText}
                  output={outText}
                  durationSeconds={item.duration_seconds}
                  exitCode={item.exit_code}
                  actionStatus={item.action_status}
                  agentName={item.sub_agent_name}
                  agentId={item.sub_agent_id}
                />
              );
            }

            // 3. File Operations (matching Devin's Read <file>:<lines> badge)
            const isFileOp =
              item.type === "read" ||
              item.type === "write" ||
              item.title?.startsWith("Read ") ||
              item.title?.startsWith("Write ") ||
              item.title?.includes("FILE_READ") ||
              item.title?.includes("FILE_WRITE");

            if (isFileOp) {
              const opType =
                item.type === "write" ||
                item.title?.startsWith("Write ") ||
                item.title?.includes("FILE_WRITE")
                  ? "write"
                  : "read";
              let targetFile = item.file || "";
              if (!targetFile) {
                const targetMatch = item.content?.match(/Target:\s*([^\n]+)/);
                targetFile = targetMatch ? targetMatch[1].trim() : item.title?.replace(/^(Read|Write)\s+/, "") || "file";
              }
              return (
                <FileActionBlock
                  key={itemId}
                  type={opType}
                  file={targetFile}
                  lines={item.lines}
                  onSelectFile={onSelectFile}
                />
              );
            }

            // 4. GUI & Desktop Interactions (GUI_CLICK, APP_LAUNCH, APP_FOCUS, GUI_TYPE, GUI_WAIT)
            const isGuiOp =
              item.title?.includes("GUI_") ||
              item.title?.includes("APP_") ||
              item.title?.includes("SECURITY_TOOL");

            if (isGuiOp) {
              const actionTitle = item.title?.replace(/^Step \d+:\s*/, "") || "GUI Action";
              let targetText = "";
              const targetMatch = item.content?.match(/Target:\s*([^\n]+)/);
              if (targetMatch) {
                targetText = targetMatch[1].trim();
              }
              const isClick = actionTitle.includes("CLICK");
              const isType = actionTitle.includes("TYPE");
              const isLaunch = actionTitle.includes("LAUNCH") || actionTitle.includes("FOCUS");
              const isSecurity = actionTitle.includes("SECURITY_TOOL");

              const icon = isClick ? (
                <MousePointer className="w-3.5 h-3.5 text-accent-cyan shrink-0" />
              ) : isLaunch ? (
                <AppWindow className="w-3.5 h-3.5 text-secondary-400 shrink-0" />
              ) : isSecurity ? (
                <Shield className="w-3.5 h-3.5 text-accent-purple shrink-0" />
              ) : isType ? (
                <Terminal className="w-3.5 h-3.5 text-accent-amber shrink-0" />
              ) : (
                <Eye className="w-3.5 h-3.5 text-muted-bright shrink-0" />
              );

              return (
                <div key={itemId} className="my-1 flex items-center gap-2 font-mono text-[11px] py-1 px-2.5 rounded-lg bg-ink-850/60 border border-ink-800/80 hover:border-ink-750 transition">
                  {icon}
                  <span className="font-semibold text-slate-200">{actionTitle}</span>
                  {targetText && (
                    <span className="text-secondary-300 bg-ink-900 px-1.5 py-0.5 rounded border border-ink-800 text-[10.5px] truncate max-w-xs font-mono">
                      {targetText}
                    </span>
                  )}
                  {item.duration_seconds && item.duration_seconds > 0 ? (
                    <span className="ml-auto text-[9.5px] text-muted-dim">{item.duration_seconds}s</span>
                  ) : null}
                </div>
              );
            }

            // 5. Desktop Observation Brief Banner
            if (item.title === "Agent Desktop Observation" || item.title === "Agent Visual Computer Use") {
              const isObservation = item.title === "Agent Desktop Observation";
              return (
                <div key={itemId} className="my-1.5 flex items-center gap-2 text-[10.5px] font-mono text-muted-bright py-1 px-2.5 rounded-lg bg-ink-900 border border-ink-800">
                  <Monitor className="w-3.5 h-3.5 text-secondary-400 shrink-0" />
                  <span className="text-slate-300 font-medium">
                    {isObservation ? "🖥️ Desktop Observation Synchronized" : "🎯 Autonomous Visual Objective"}
                  </span>
                  <span className="text-muted-dim text-[9.5px] ml-auto">1920x1080</span>
                </div>
              );
            }

            // 4. User Chat Message
            if (item.title === "Objective Received" || item.role === "user") {
              const userText = item.prompt || item.content || item.title;
              return (
                <div key={itemId} className="flex justify-end pl-8 my-2 animate-fade-in-up">
                  <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-secondary-900/40 border border-secondary-600/40 px-3.5 py-2.5 text-slate-100 shadow-md">
                    <div className="flex items-center justify-between gap-3 text-[10.5px] text-secondary-400 font-mono font-medium mb-1">
                      <div className="flex items-center gap-1.5">
                        <User className="w-3 h-3 text-secondary-400" />
                        <span>You</span>
                      </div>
                      {item.timestamp && (
                        <span className="text-[9.5px] text-muted-dim font-normal">
                          {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      )}
                    </div>
                    <p className="text-[12.5px] leading-relaxed whitespace-pre-wrap">{userText}</p>
                  </div>
                </div>
              );
            }

            // 5. SONIC Assistant Response
            if (item.type === "response" || item.title === "SONIC Response") {
              const responseText = item.content || "";
              const isCopied = copiedId === itemId;
              return (
                <div key={itemId} className="space-y-1.5 my-2.5 animate-fade-in-up">
                  <div className="flex justify-start pr-4">
                    <div className="max-w-[92%] rounded-2xl rounded-bl-sm bg-ink-850 border border-ink-750 p-3.5 text-slate-200 shadow-lg">
                      <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-ink-800 text-[10.5px]">
                        <div className="flex items-center gap-1.5 text-secondary-400 font-semibold font-mono">
                          <Bot className="w-3.5 h-3.5 text-secondary-400" />
                          <span>SONIC</span>
                          <span className="text-[9px] text-muted-dim uppercase px-1 rounded bg-ink-900 border border-ink-750">
                            A-SEA
                          </span>
                          {item.outcome_status && (
                            <span className={`rounded border px-1.5 py-0.5 text-[9px] font-semibold ${outcomeBadgeClass(item.outcome_status)}`}>
                              {item.outcome_status}
                            </span>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={() => handleCopyResponse(itemId, responseText)}
                          className="flex items-center gap-1 text-muted hover:text-white transition px-1.5 py-0.5 rounded hover:bg-ink-800 text-[10px]"
                          title="Copy response"
                        >
                          {isCopied ? (
                            <>
                              <Check className="w-3 h-3 text-success" />
                              <span className="text-success">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span>Copy</span>
                            </>
                          )}
                        </button>
                      </div>

                      <MarkdownText content={responseText} />
                    </div>
                  </div>

                </div>
              );
            }

            // 6. Parallel Specialist Step (e.g. [NetworkSpecialist] Scanning ports...)
            const specialistMatch =
              item.title?.match(/^\[([a-zA-Z0-9_-]+)\]\s*(.*)/) ||
              item.content?.match(/^\[([a-zA-Z0-9_-]+)\]\s*(.*)/);
            if (specialistMatch) {
              const specialistName = specialistMatch[1];
              const specialistMsg = specialistMatch[2];
              return (
                <div key={itemId} className="my-1 flex items-center gap-2 font-mono text-[11px]">
                  <Shield className="w-3.5 h-3.5 text-secondary-400 shrink-0" />
                  <span className="px-1.5 py-0.2 rounded bg-secondary-950/60 border border-secondary-800 text-secondary-300 font-semibold text-[10px]">
                    {specialistName}
                  </span>
                  <span className="text-muted-bright">{specialistMsg || item.content}</span>
                </div>
              );
            }

            // 7. Phase Completion
            if (item.title?.startsWith("Phase ") && item.title?.includes("Complete")) {
              return <PhaseCompleteBanner key={itemId} item={item} />;
            }

            // 10. Generic Action Item
            return (
              <div key={itemId} className="flex items-center gap-2 text-muted text-[11px] py-0.5">
                <Zap className="w-3.5 h-3.5 text-secondary-400 shrink-0" />
                <span className="text-slate-300 font-medium">{item.title}</span>
                {item.content && <span className="text-muted-dim truncate max-w-sm">({item.content})</span>}
              </div>
            );
            })}
          </>
        )}

        {/* Real-time active thinking / action indicator */}
        {cleanActionText && (
          <div className="pt-2 pb-1 flex items-center gap-2 text-xs text-secondary-400 pl-1 animate-fade-in-up">
            <Brain className="w-3.5 h-3.5 text-secondary-400 animate-pulse shrink-0" />
            <div className="w-1.5 h-1.5 rounded-full bg-secondary-400 animate-ping shrink-0" />
            <span className="text-secondary-300 font-mono font-medium text-[11.5px]">
              {cleanActionText}
            </span>
          </div>
        )}


        <div ref={worklogEndRef} />
      </div>

      {/* Pause and human-assistance states both need an explicit continuation path. */}
      {(isWaitingForHuman || isPaused) && (
        <div className="mx-3 mt-2 rounded-xl border border-amber-500/50 bg-amber-500/10 p-3 text-xs space-y-2 shadow-lg">
          <div className="flex items-center gap-2 text-amber-400 font-mono font-semibold">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>
              {isWaitingForHuman
                ? "WAITING_FOR_HUMAN — Operator Verification / Assistance Needed"
                : "PAUSED — Human Takeover"}
            </span>
          </div>
          <p className="text-[11.5px] text-slate-200 leading-relaxed">
            {isWaitingForHuman
              ? waitingForHumanReason || "Complete the requested verification on the live desktop, then continue."
              : "The worker stopped. Continuing starts a new reasoning run from the original objective and current desktop; review prior actions because there is no restored worker checkpoint."}
          </p>
          <div className="flex items-center gap-2">
            {isWaitingForHuman && (
              <input
                type="text"
                value={resumeNote}
                onChange={(e) => setResumeNote(e.target.value)}
                placeholder="Enter verification note (optional)..."
                className="flex-1 bg-ink-950 border border-amber-500/40 rounded-lg px-2.5 py-1.5 text-xs font-mono text-white outline-none focus:border-amber-400"
              />
            )}
            <button
              type="button"
              disabled={resuming}
              onClick={handleResumeWorkstation}
              className="px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-ink-950 font-mono font-bold text-xs flex items-center gap-1.5 transition disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>
                {resuming
                  ? "Starting..."
                  : isPaused
                    ? "Re-evaluate & Continue"
                    : "Continue"}
              </span>
            </button>
          </div>
        </div>
      )}

      {/* Prompt Area */}
      <div className="p-3 border-t border-ink-800 bg-ink-900 space-y-2">
        {/* Input Bar */}
        <form
          onSubmit={handleSubmit}
          className="rounded-2xl border border-ink-750 bg-ink-850 p-2.5 space-y-2 shadow-lg focus-within:border-secondary-500/70 transition relative"
        >
          <textarea
            ref={textareaRef}
            aria-label="Message SONIC"
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            rows={3}
            placeholder="Instruct SONIC or ask a question about a supplied target, application, or workspace..."
            className="w-full min-h-16 resize-y bg-transparent text-sm text-slate-100 placeholder:text-muted-dim outline-none font-sans px-1 leading-relaxed"
          />

          <div className="flex items-center justify-between pt-0.5">
            <div className="flex items-center gap-1.5 relative">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setModeMenuOpen(!modeMenuOpen)}
                  className="px-2 py-0.5 rounded-md bg-ink-800 hover:bg-ink-750 border border-ink-700 text-muted-bright hover:text-white flex items-center gap-1 text-[11px] font-medium transition"
                >
                  <Sparkles className="w-3 h-3 text-secondary-400" />
                  <span>{activeMode}</span>
                </button>

                {modeMenuOpen && (
                  <div role="menu" aria-label="Execution mode" className="absolute bottom-8 left-0 w-36 rounded-lg bg-ink-850 border border-ink-700 shadow-xl py-1 z-30 text-xs text-muted-bright">
                    {MODES.map((m) => (
                      <button
                        type="button"
                        role="menuitemradio"
                        aria-checked={activeMode === m}
                        key={m}
                        onClick={() => {
                          setActiveMode(m);
                          setModeMenuOpen(false);
                        }}
                        className={`w-full px-3 py-1.5 hover:bg-ink-800 flex items-center justify-between text-left text-[11px] ${
                          activeMode === m ? "text-secondary-400 font-semibold" : ""
                        }`}
                      >
                        <span>{m}</span>
                        {activeMode === m && <Check className="w-3 h-3" />}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              {(loading || normalizedStatus === "RUNNING") && onInterrupt && (
                <button
                  type="button"
                  onClick={onInterrupt}
                  className="px-2.5 py-1 rounded-lg bg-danger/20 hover:bg-danger/30 border border-danger/40 text-danger text-xs font-semibold flex items-center gap-1 transition"
                  title="Interrupt current execution"
                >
                  <Square className="w-3 h-3 fill-current" />
                  <span>Stop</span>
                </button>
              )}

              <button
                type="submit"
                disabled={!promptText.trim() || normalizedStatus === "RUNNING"}
                aria-label="Send instruction"
                className="w-10 h-10 rounded-lg bg-secondary-600 hover:bg-secondary-500 disabled:opacity-40 disabled:hover:bg-secondary-600 text-white flex items-center justify-center transition shadow-glow cursor-pointer"
                title="Send instruction"
              >
                {loading && normalizedStatus === "RUNNING" ? (
                  <div className="w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                ) : (
                  <Send className="w-3.5 h-3.5" />
                )}
              </button>
            </div>

          </div>
        </form>
      </div>
    </div>
  );
}
