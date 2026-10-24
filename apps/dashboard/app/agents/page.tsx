"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Cpu,
  RefreshCw,
  AlertCircle,
  Activity,
  Bot,
  Network,
  ChevronDown,
  ChevronRight,
  Users,
  Loader2,
} from "lucide-react";
import { api } from "../../lib/api";
import { PageShell, PageHeader, StateBlock } from "../../components/common/PageShell";

interface AgentEntry {
  name: string;
  type: string;
  status: string;
  task: string;
  model: string;
  source?: string;
}

interface LiveWorkerStep {
  step_index?: number | null;
  action_type?: string;
  status?: string;
  target?: string;
  summary?: string;
  timestamp?: string;
}

interface LiveWorker {
  key: string;
  agent_uid?: string;
  agent_id?: string;
  sub_mission_id?: string;
  agent_name?: string;
  role?: string;
  plane?: string;
  goal?: string;
  status?: string;
  is_running?: boolean;
  latest_action?: string;
  latest_action_status?: string;
  latest_thought?: string;
  step_count?: number;
  steps?: LiveWorkerStep[];
  findings_summary?: string;
  judge_verdict?: string;
  evidence_verified?: boolean | null;
  key_discoveries?: string[];
}

const STATUS_STYLE: Record<string, string> = {
  running: "text-warning border-warning/40 bg-warning/10",
  active: "text-warning border-warning/40 bg-warning/10",
  idle: "text-muted border-ink-600 bg-ink-800",
  completed: "text-success border-success/40 bg-success/10",
  blocked: "text-danger border-danger/40 bg-danger/10",
  error: "text-danger border-danger/40 bg-danger/10",
};

function StatusDot({ status }: { status: string }) {
  const s = (status || "idle").toLowerCase();
  const color =
    s === "running" || s === "active"
      ? "bg-warning animate-pulse"
      : s === "completed" || s === "success"
        ? "bg-success"
        : s === "blocked" || s === "error"
          ? "bg-danger"
          : "bg-muted-dim";
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentEntry[]>([]);
  const [source, setSource] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Live Boss workers are a different thing from the SwarmRunner registry
  // above: these are the workers a running mission actually launched. They are
  // read from the authoritative per-worker endpoint, and expanding one loads
  // its current activity.
  const [workers, setWorkers] = useState<LiveWorker[]>([]);
  const [workersError, setWorkersError] = useState<string | null>(null);
  const [expandedWorker, setExpandedWorker] = useState<string | null>(null);
  const [workerDetail, setWorkerDetail] = useState<Record<string, LiveWorker>>({});
  const [detailLoading, setDetailLoading] = useState<string | null>(null);

  const fetchAgents = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getAgents();
      setAgents(data.agents || []);
      setSource(data.source || "");
    } catch (err: any) {
      setError(err.message || "Failed to load agents.");
      setAgents([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchWorkers = useCallback(async () => {
    setWorkersError(null);
    try {
      const data = await api.getWorkstationAgents();
      setWorkers(data.workers || []);
    } catch (err: any) {
      setWorkersError(err.message || "Could not load live workers.");
      setWorkers([]);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
    fetchWorkers();
  }, [fetchAgents, fetchWorkers]);

  const toggleWorker = useCallback(
    async (worker: LiveWorker) => {
      const uid = worker.agent_uid || worker.key;
      if (expandedWorker === uid) {
        setExpandedWorker(null);
        return;
      }
      setExpandedWorker(uid);
      if (!worker.agent_uid) return;
      setDetailLoading(uid);
      try {
        const response = await api.getWorkstationAgent(worker.agent_uid);
        if (response?.worker) {
          setWorkerDetail((prev) => ({ ...prev, [uid]: response.worker }));
        }
      } catch (err: any) {
        setWorkersError(err.message || "Could not load worker detail.");
      } finally {
        setDetailLoading(null);
      }
    },
    [expandedWorker],
  );

  const activeCount = agents.filter((a) =>
    ["running", "active"].includes((a.status || "").toLowerCase()),
  ).length;
  const runningWorkers = workers.filter((w) => w.is_running).length;

  return (
    <PageShell>
      <PageHeader
        accent="secondary"
        icon={<Cpu className="w-6 h-6 text-secondary-400" />}
        title="AGENT SWARM REGISTRY"
        badge={`${agents.length} agents · ${activeCount} active${
          workers.length > 0 ? ` · ${runningWorkers}/${workers.length} workers` : ""
        }`}
        subtitle="SwarmRunner registry plus the live workers the Boss launched for the current mission"
        actions={
          <button
            onClick={() => {
              fetchAgents();
              fetchWorkers();
            }}
            disabled={loading}
            className="btn-ghost inline-flex items-center gap-2 text-xs font-mono"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            REFRESH
          </button>
        }
      />

      {error && (
        <div className="p-4 rounded-xl bg-danger/10 border border-danger/30 text-danger text-xs font-mono flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {workersError && (
        <div className="mb-4 p-3 rounded-xl bg-warning/10 border border-warning/30 text-warning text-[10px] font-mono">
          {workersError}
        </div>
      )}

      {workers.length > 0 && (
        <section
          aria-label="Live Boss workers"
          className="mb-4 rounded-2xl border border-ink-750 bg-ink-850/60 p-4"
        >
          <div className="mb-3 flex items-center gap-2">
            <Users className="w-4 h-4 shrink-0 text-secondary-400" />
            <h2 className="text-sm font-semibold text-slate-100">Live Boss workers</h2>
            <span className="text-[10px] font-mono text-muted-dim">
              {runningWorkers > 0 ? `${runningWorkers} active · ` : ""}
              {workers.length} {workers.length === 1 ? "worker" : "workers"}
            </span>
          </div>

          <div className="space-y-2">
            {workers.map((worker) => {
              const uid = worker.agent_uid || worker.key;
              const expanded = expandedWorker === uid;
              const detail = workerDetail[uid] || worker;
              const status = (worker.status || "UNKNOWN").toUpperCase();
              const plane =
                (worker.plane || "").toLowerCase() === "headless"
                  ? "Terminal / Code"
                  : "Computer / GUI";
              return (
                <article
                  key={worker.key}
                  className={`rounded-xl border bg-ink-900/80 p-3 ${
                    expanded ? "border-secondary-500/40" : "border-ink-750"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => toggleWorker(worker)}
                    aria-expanded={expanded}
                    className="flex w-full items-start gap-2 text-left"
                  >
                    <Bot className="mt-0.5 h-4 w-4 shrink-0 text-secondary-400" />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="truncate text-xs font-semibold text-slate-100">
                          {worker.agent_name || worker.agent_id || "Worker"}
                        </span>
                        {worker.agent_uid && (
                          <span className="truncate rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 font-mono text-[9px] text-muted-dim">
                            {worker.agent_uid}
                          </span>
                        )}
                        <span className="rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 text-[9px] text-muted-bright">
                          {plane}
                        </span>
                        <span
                          className={`ml-auto rounded border px-1.5 py-0.5 text-[9px] font-mono uppercase ${
                            STATUS_STYLE[status.toLowerCase()] || STATUS_STYLE.idle
                          }`}
                        >
                          {status}
                        </span>
                        {expanded ? (
                          <ChevronDown className="h-3 w-3 shrink-0 text-muted-dim" />
                        ) : (
                          <ChevronRight className="h-3 w-3 shrink-0 text-muted-dim" />
                        )}
                      </div>
                      <p className="mt-1 line-clamp-2 text-[10.5px] leading-relaxed text-muted-bright">
                        {worker.goal || "No goal recorded for this worker."}
                      </p>
                      {worker.latest_action && (
                        <p className="mt-1 truncate text-[10px] text-muted-dim">
                          Latest: {worker.latest_action}
                          {worker.latest_action_status
                            ? ` · ${worker.latest_action_status.toLowerCase()}`
                            : ""}
                        </p>
                      )}
                    </div>
                  </button>

                  {expanded && (
                    <div className="mt-2 space-y-2 border-t border-ink-750 pt-2">
                      {detailLoading === uid ? (
                        <p className="flex items-center gap-1.5 text-[10px] text-muted-dim">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Loading live worker state…
                        </p>
                      ) : workerDetail[uid] ? (
                        <p className="text-[9px] font-mono text-emerald-400">live</p>
                      ) : (
                        <p className="text-[9px] font-mono text-muted-dim">
                          showing last known events
                        </p>
                      )}
                      {detail.latest_action && (
                        <div>
                          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
                            Current activity
                          </p>
                          <p className="mt-0.5 text-[10.5px] text-slate-200">
                            {detail.latest_action}
                          </p>
                        </div>
                      )}
                      {detail.latest_thought && (
                        <div>
                          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
                            Latest thinking
                          </p>
                          <p className="mt-0.5 line-clamp-4 whitespace-pre-wrap text-[10.5px] leading-relaxed text-slate-300">
                            {detail.latest_thought}
                          </p>
                        </div>
                      )}
                      {(detail.steps?.length || 0) > 0 && (
                        <div>
                          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
                            Recent activity
                          </p>
                          <ol className="mt-1 space-y-1">
                            {(detail.steps || []).slice(-8).map((step, index) => (
                              <li
                                key={`${worker.key}-step-${index}`}
                                className="flex items-start gap-1.5 text-[10px] text-slate-300"
                              >
                                <span className="mt-0.5 shrink-0 rounded border border-ink-700 bg-ink-850 px-1 text-[8.5px] text-muted-dim">
                                  {step.step_index ?? index + 1}
                                </span>
                                <span className="min-w-0">
                                  {step.summary || step.action_type || step.target}
                                </span>
                              </li>
                            ))}
                          </ol>
                        </div>
                      )}
                      {detail.judge_verdict && (
                        <div>
                          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
                            Independent Judge
                          </p>
                          <p className="mt-0.5 text-[10.5px] text-amber-300">
                            {detail.judge_verdict}
                            {detail.evidence_verified === true
                              ? " · evidence confirmed"
                              : " · evidence not confirmed"}
                          </p>
                        </div>
                      )}
                      {detail.findings_summary && (
                        <div>
                          <p className="text-[9px] uppercase tracking-wide text-muted-dim">
                            Report
                          </p>
                          <p className="mt-0.5 whitespace-pre-wrap text-[10.5px] leading-relaxed text-slate-300">
                            {detail.findings_summary}
                          </p>
                        </div>
                      )}
                      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[9.5px] text-muted-dim">
                        {worker.agent_id && <span>agent: {worker.agent_id}</span>}
                        {worker.sub_mission_id && (
                          <span>mission: {worker.sub_mission_id}</span>
                        )}
                        {detail.step_count !== undefined && (
                          <span>{detail.step_count} steps</span>
                        )}
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        </section>
      )}

      <StateBlock
        loading={loading}
        loadingText="Loading agent registry…"
        empty={!loading && agents.length === 0}
        emptyIcon={<Bot className="w-6 h-6" />}
        emptyTitle="No agents registered"
        emptyText="The SwarmRunner has not initialized any agents yet, or the swarm backend is offline."
      >
        <>
          {source && (
            <div className="flex items-center gap-2 text-[10px] font-mono text-muted-dim">
              <Network className="w-3 h-3" />
              SOURCE: {source}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {agents.map((a, i) => {
              const s = (a.status || "idle").toLowerCase();
              return (
                <div
                  key={`${a.name}-${i}`}
                  className="glass-card rounded-2xl p-4 space-y-3"
                >
                  <div className="flex items-start gap-3">
                    <div className="grid place-items-center w-10 h-10 rounded-xl bg-ink-850 border border-ink-700 shrink-0">
                      <Bot className="w-5 h-5 text-secondary-400" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-bold text-white font-mono truncate">
                        {a.name}
                      </div>
                      <div className="text-xs text-muted-dim font-mono">
                        {a.type}
                      </div>
                    </div>
                    <span
                      className={`text-[10px] font-mono px-2 py-1 rounded border uppercase ${
                        STATUS_STYLE[s] || STATUS_STYLE.idle
                      }`}
                    >
                      {a.status}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-muted font-mono">
                    <StatusDot status={a.status} />
                    <span className="truncate">{a.task || "Ready"}</span>
                  </div>

                  <div className="flex items-center gap-2 pt-2 border-t border-ink-800/60 text-[10px] font-mono text-muted-dim">
                    <Activity className="w-3 h-3" />
                    model: {a.model || "Configured LLM"}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      </StateBlock>
    </PageShell>
  );
}
