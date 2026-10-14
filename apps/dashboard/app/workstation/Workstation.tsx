"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  FileText,
  Monitor,
  PanelRightClose,
  Share2,
  FileCheck2,
  Dna,
  Compass,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { getAuthToken } from "../../lib/auth";
import { api } from "../../lib/api";
import {
  WorkstationState,
  SystemStatus,
  WorkstationTab,
  CommandResult,
  SessionItem,
  normalizeSessionList,
  sanitizeCurrentAction,
} from "../../types/workstation";

import { OfflineBanner } from "../../components/common/OfflineBanner";
import { WorkstationHeader as BaseWorkstationHeader } from "../../components/workstation/WorkstationHeader";
import { WorkstationSidebar } from "../../components/workstation/WorkstationSidebar";

const WorkstationHeader = BaseWorkstationHeader as React.ComponentType<
  React.ComponentProps<typeof BaseWorkstationHeader> & { loading?: boolean }
>;
import { WorklogFeed } from "../../components/worklog/WorklogFeed";
import { ComputerSurface } from "../../components/computer/ComputerSurface";
import { CodeViewer } from "../../components/code/CodeViewer";
import { ResearchView } from "../../components/research/ResearchView";
import { EvidenceView } from "../../components/evidence/EvidenceView";
import { EvolutionView } from "../../components/evolution/EvolutionView";
import { MissionView } from "../../components/mission/MissionView";

const TABS: { id: WorkstationTab; label: string; icon: React.ReactNode }[] = [
  { id: "desktop", label: "Computer", icon: <Monitor className="w-3.5 h-3.5 text-success" /> },
  { id: "code", label: "Code", icon: <FileText className="w-3.5 h-3.5 text-secondary-400" /> },
  { id: "changes", label: "Changes", icon: <FileText className="w-3.5 h-3.5 text-warning" /> },
  { id: "research", label: "Research", icon: <Share2 className="w-3.5 h-3.5 text-accent-400" /> },
  { id: "evidence", label: "Evidence", icon: <FileCheck2 className="w-3.5 h-3.5 text-success" /> },
  { id: "evolution", label: "Evolution", icon: <Dna className="w-3.5 h-3.5 text-primary-400" /> },
  { id: "mission", label: "Mission", icon: <Compass className="w-3.5 h-3.5 text-primary-400" /> },
];

function SonicDevinWorkstation() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<WorkstationTab>("desktop");
  const [connectionStatus, setConnectionStatus] = useState<SystemStatus>("CONNECTING");
  const [sessionId, setSessionId] = useState("default");
  const [workstationState, setWorkstationState] = useState<WorkstationState | null>(null);
  const [activeFile, setActiveFile] = useState("");
  const [fileContent, setFileContent] = useState<string[]>([]);
  const [gitDiff, setGitDiff] = useState("");
  const [commandLogs, setCommandLogs] = useState<string[]>([]);
  const combinedCommandLogs = useMemo(() => {
    const list: string[] = [];
    if (workstationState?.worklog && Array.isArray(workstationState.worklog)) {
      for (const entry of workstationState.worklog) {
        if (entry.type === "command" && entry.command) {
          list.push(`sonic@workstation:~$ ${entry.command}`);
          if (entry.output) {
            list.push(entry.output.trim());
          }

        }
      }
    }
    for (const log of commandLogs) {
      list.push(log);
    }
    return list;
  }, [workstationState?.worklog, commandLogs]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [missionErrorMessage, setMissionErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const tabParam = params.get("tab");
      if (tabParam) {
        const validTabs: Record<string, WorkstationTab> = {
          desktop: "desktop",
          computer: "desktop",
          code: "code",
          changes: "changes",
          research: "research",
          evidence: "evidence",
          evolution: "evolution",
          mission: "mission",
          missions: "mission",
        };
        const mapped = validTabs[tabParam.toLowerCase()];
        if (mapped) {
          setActiveTab(mapped);
        }
      }
    }
  }, []);

  const [sessionList, setSessionList] = useState<SessionItem[]>([]);
  const [sessionListError, setSessionListError] = useState<string | null>(null);
  const abortPollRef = useRef<boolean>(false);
  const activePromptSessionRef = useRef<string | null>(null);
  const fetchGenerationRef = useRef(0);

  const fetchWorkstationData = async (targetSession = sessionId) => {
    const generation = ++fetchGenerationRef.current;
    try {
      const state = await api.getWorkstationState(targetSession);
      if (generation !== fetchGenerationRef.current) return;
      setWorkstationState(state);
      setConnectionStatus("LIVE");
      setErrorMessage(null);

      // If backend reports state is not RUNNING, stop loading spinner
      if (state && state.status !== "RUNNING") {
        setLoading(false);
      }

      try {
        const rawSessions = await api.listSessions();
        const sessions = normalizeSessionList(rawSessions);
        setSessionList(sessions);
        setSessionListError(null);
        if (
          targetSession === "fresh" &&
          !state?.desktop?.workspace_id
        ) {
          const attachedSession = sessions.find((session: any) => session.workspace_id);
          if (attachedSession?.session_id) {
            setSessionId(attachedSession.session_id);
            return;
          }
        }
      } catch (err: any) {
        setSessionListError(err.message || "Could not load sessions.");
      }
    } catch (err: any) {
      setConnectionStatus("OFFLINE");
      setErrorMessage(err.message || "Failed to reach backend control plane.");
    }
  };

  const handleNewSession = async () => {
    abortPollRef.current = true;
    setLoading(false);
    const newSessionId = `mission-${
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).substring(2, 10)
    }`;
    setSessionId(newSessionId);
    setActiveFile("");
    setFileContent([]);
    setGitDiff("");
    setCommandLogs([]);
    await fetchWorkstationData(newSessionId);
  };


  const handleDeleteSession = async (delSessionId: string) => {
    try {
      // Optimistically remove from session list immediately
      setSessionList((prev) => prev.filter((s) => s.session_id !== delSessionId));

      await api.deleteSession(delSessionId);

      if (sessionId === delSessionId) {
        // Switch cleanly to the next available session or default
        const remaining = sessionList.filter((s) => s.session_id !== delSessionId);
        const nextSessionId = remaining.length > 0 ? remaining[0].session_id : "default";
        setSessionId(nextSessionId);
        setActiveFile("");
        setFileContent([]);
        setGitDiff("");
        setCommandLogs([]);
        await fetchWorkstationData(nextSessionId);
      } else {
        await fetchWorkstationData(sessionId);
      }
    } catch {
      // Refresh to restore truthful server state on error
      await fetchWorkstationData(sessionId);
    }
  };

  const fetchFile = async (path: string) => {
    if (!path) {
      setActiveFile("");
      setFileContent([]);
      return;
    }
    setActiveFile(path);
    try {
      const data = await api.getFileContent(path, sessionId);
      if (data?.lines) setFileContent(data.lines);
      else if (data?.content) setFileContent(data.content.split("\n"));
    } catch {
      setFileContent(["# Error reading file from filesystem."]);
    }
  };

  const fetchDiff = async () => {
    try {
      const data = await api.getGitDiff(sessionId);
      if (data?.diff) setGitDiff(data.diff);
    } catch {
      setGitDiff("");
    }
  };

  useEffect(() => {
    fetchWorkstationData(sessionId);
    fetchDiff();
    const interval = setInterval(() => fetchWorkstationData(sessionId), 5000);
    return () => clearInterval(interval);
  }, [sessionId]);

  const handleSendPrompt = async (prompt: string, mode: "Normal" | "Autonomous" | "Pair-Program") => {
    const currentTargetSession = sessionId;
    activePromptSessionRef.current = currentTargetSession;
    abortPollRef.current = false;
    setMissionErrorMessage(null);
    setLoading(true);
    let keepLoading = false;

    try {
      const res = await api.sendPrompt(prompt, currentTargetSession, mode.toLowerCase());
      if (mode === "Autonomous") setActiveTab("mission");

      // If user interrupted or navigated away during prompt send, abort immediately
      if (abortPollRef.current || activePromptSessionRef.current !== currentTargetSession) {
        setLoading(false);
        return;
      }

      if (res?.state) {
        setWorkstationState(res.state);
        // If state is already non-running (IDLE, BLOCKED, PAUSED, COMPLETED), stop loading
        if (res.state.status && res.state.status !== "RUNNING") {
          setLoading(false);
          await fetchDiff();
          return;
        }
      }

      // Keep the active task visible while workers are still within their backend deadline.
      const MAX_ATTEMPTS = 1800;
      let finished = false;

      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
        if (abortPollRef.current || activePromptSessionRef.current !== currentTargetSession) {
          break;
        }

        await new Promise((resolve) => setTimeout(resolve, 1000));

        if (abortPollRef.current || activePromptSessionRef.current !== currentTargetSession) {
          break;
        }

        try {
          const nextState = await api.getWorkstationState(currentTargetSession);
          if (abortPollRef.current || activePromptSessionRef.current !== currentTargetSession) {
            break;
          }

          setWorkstationState(nextState);

          // 1. If backend status transitioned to non-RUNNING (IDLE, PAUSED, BLOCKED, COMPLETED)
          if (nextState?.status && nextState.status !== "RUNNING") {
            finished = true;
            break;
          }

        } catch {
          // Retry on transient network blip
        }
      }

      // Reconcile the last server state before reporting a poll timeout.
      if (!finished && !abortPollRef.current) {
        try {
          const finalState = await api.getWorkstationState(currentTargetSession);
          if (activePromptSessionRef.current === currentTargetSession) {
            setWorkstationState(finalState);
            if (finalState?.status === "RUNNING") {
              keepLoading = true;
              setMissionErrorMessage(
                "SONIC is still reporting this task as running. Do not submit a duplicate; use Stop if you want to interrupt it."
              );
            }
          }
        } catch (err: any) {
          setMissionErrorMessage(`Could not confirm the final task state: ${err.message || String(err)}`);
        }
      }

      await fetchDiff();
      try {
        const rawSessions = await api.listSessions();
        setSessionList(normalizeSessionList(rawSessions));
        setSessionListError(null);
      } catch (err: any) {
        setSessionListError(err.message || "Could not load sessions.");
      }
    } catch (err: any) {
      setMissionErrorMessage(`Could not submit or monitor the task: ${err.message || String(err)}`);
    } finally {
      if (activePromptSessionRef.current === currentTargetSession) {
        setLoading(keepLoading);
      }
    }
  };

  const handleRunCommand = async (cmd: string): Promise<CommandResult | null> => {
    setCommandLogs((prev) => [...prev, `sonic@workstation:~$ ${cmd}`]);
    try {
      const res = await api.executeCommand(cmd, sessionId);
      const receipt = [
        `Exit code: ${res?.exit_code ?? "unknown"}`,
        res?.output?.trim(),
        res?.stderr?.trim() ? `stderr:\n${res.stderr.trim()}` : "",
      ].filter(Boolean).join("\n");
      setCommandLogs((prev) => [...prev, receipt]);
      await fetchWorkstationData(sessionId);
      return res;
    } catch (err: any) {
      setCommandLogs((prev) => [...prev, `[FAIL-CLOSED REJECTED]: ${err.message}`]);
      return null;
    }
  };

  const handleSaveFile = async (content: string) => {
    try {
      await api.saveFileContent(activeFile, content, sessionId);
      await fetchFile(activeFile);
      await fetchDiff();
    } catch (err: any) {
      alert(`Failed to save file: ${err.message}`);
    }
  };

  const handleInterrupt = async () => {
    try {
      await api.interruptSession(sessionId);
      abortPollRef.current = true;
      setLoading(false);
      await fetchWorkstationData(sessionId);
      return true;
    } catch (err: any) {
      setMissionErrorMessage(`Could not confirm the task was stopped: ${err.message || String(err)}`);
      return false;
    }
  };

  return (
    <div className="sonic-workstation flex flex-col h-screen w-screen bg-ink-950 text-slate-200 overflow-hidden font-sans">
      {connectionStatus === "OFFLINE" && (
        <OfflineBanner onRetry={() => fetchWorkstationData(sessionId)} message={errorMessage || undefined} />
      )}
      {missionErrorMessage && (
        <div role="alert" className="border-b border-amber-500/50 bg-amber-500/10 px-4 py-2 text-sm text-amber-200">
          {missionErrorMessage}
        </div>
      )}

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <WorkstationSidebar
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          sessions={sessionList}
          sessionsError={sessionListError}
          currentSessionId={sessionId}
          onSelectSession={(id) => {
            abortPollRef.current = true;
            setLoading(false);
            setSessionId(id);
            setActiveFile("");
            setFileContent([]);
            setGitDiff("");
            setCommandLogs([]);
            fetchWorkstationData(id);
          }}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
        />

        <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden bg-ink-950">
          <WorkstationHeader
            sidebarOpen={sidebarOpen}
            setSidebarOpen={setSidebarOpen}
            sessionName={workstationState?.session_name || workstationState?.mission_name || "New conversation"}
            gitBranch={workstationState?.git_branch || ""}
            latestCommit={workstationState?.latest_commit}
            connectionStatus={connectionStatus}
            loading={loading}
          />

          <div className="sonic-workstation-content flex-1 min-h-0 grid grid-cols-12 overflow-hidden">
            <div className={`${rightPanelOpen ? "col-span-5" : "col-span-12"} sonic-workstation-conversation flex flex-col h-full bg-ink-900 overflow-hidden`}>
              <WorklogFeed
                worklog={workstationState?.worklog || []}
                currentAction={sanitizeCurrentAction(
                  workstationState?.current_action,
                  workstationState?.status,
                  loading
                )}
                status={workstationState?.status}
                waitingForHuman={Boolean(workstationState?.waiting_for_human || workstationState?.status === "WAITING_FOR_HUMAN")}
                waitingForHumanReason={workstationState?.waiting_for_human_reason}
                loading={loading}
                onSendPrompt={handleSendPrompt}
                onInterrupt={handleInterrupt}
                onSelectFile={fetchFile}
                sessionName={workstationState?.session_name || workstationState?.mission_name || ""}
                gitBranch={workstationState?.git_branch || ""}
                rightPanelOpen={rightPanelOpen}
                onToggleRightPanel={() => setRightPanelOpen((open) => !open)}
                sessionId={sessionId}
                onRefreshState={() => fetchWorkstationData(sessionId)}
              />
            </div>


            {rightPanelOpen && (
              <div className="col-span-7 sonic-workstation-workspace flex flex-col h-full bg-ink-950 overflow-hidden">
                <div className="h-10 border-b border-ink-800 bg-ink-900 px-3 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1 min-w-0 overflow-x-auto">
                    {TABS.map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => {
                          setActiveTab(tab.id);
                          if (tab.id === "changes") fetchDiff();
                        }}
                        className={`px-3 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition whitespace-nowrap ${
                          activeTab === tab.id
                            ? "bg-ink-800 text-white font-semibold"
                            : "text-muted hover:text-white"
                        }`}
                      >
                        {tab.icon}
                        <span>{tab.label}</span>
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    onClick={() => setRightPanelOpen(false)}
                    className="p-1 rounded text-muted hover:text-white hover:bg-ink-800 transition shrink-0"
                    title="Hide Computer panel"
                    aria-label="Hide Computer panel"
                  >
                    <PanelRightClose className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex-1 overflow-hidden p-3 flex flex-col">
                  {activeTab === "desktop" && (
                    <ComputerSurface
                      desktopState={workstationState?.desktop}
                      status={workstationState?.status}
                      waitingForHuman={Boolean(workstationState?.waiting_for_human || workstationState?.status === "WAITING_FOR_HUMAN")}
                      waitingForHumanReason={workstationState?.waiting_for_human_reason}
                      onRunCommand={handleRunCommand}
                      commandLogs={combinedCommandLogs}
                      onProvision={async () => { await api.provisionWorkstation(sessionId); await fetchWorkstationData(sessionId); }}
                      onInterrupt={handleInterrupt}
                      onRefreshState={() => fetchWorkstationData(sessionId)}
                      sessionId={sessionId}
                    />
                  )}
                  {activeTab === "code" && (
                    <CodeViewer activeFile={activeFile} fileContent={fileContent} gitDiff={gitDiff} viewMode="code" onSaveFile={handleSaveFile} />
                  )}
                  {activeTab === "changes" && (
                    <CodeViewer activeFile={activeFile} fileContent={fileContent} gitDiff={gitDiff} viewMode="changes" onRefreshDiff={fetchDiff} />
                  )}
                  {activeTab === "research" && <ResearchView />}
                  {activeTab === "evidence" && <EvidenceView sessionId={sessionId} />}
                  {activeTab === "evolution" && <EvolutionView sessionId={sessionId} />}
                  {activeTab === "mission" && <MissionView workstationState={workstationState} sessionId={sessionId} />}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function WorkstationPage() {
  const router = useRouter();
  const [authorized, setAuthorized] = useState(false);

  useEffect(() => {
    if (getAuthToken()) {
      setAuthorized(true);
      return;
    }
    router.replace("/login?next=%2Fworkstation");
  }, [router]);

  if (!authorized) {
    return (
      <div className="min-h-screen bg-ink-950 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return <SonicDevinWorkstation />;
}
