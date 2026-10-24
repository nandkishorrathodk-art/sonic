// SONIC — Unified API Client

import { getAuthToken, ensureAuthToken, clearAuth } from "./auth";
import { SessionItem, normalizeSessionList } from "../types/workstation";

function resolveApiBase(): string {
  // Explicit env override wins: prod docker-compose passes http://sonic-core:8000;
  // nginx/proxy deployments pass their API origin here too.
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;

  // Documented local topology (AI_AGENT_GUIDE.md): backend serves on 12000,
  // frontend on 12001; CORS allowlists both origins. Same-origin only works
  // behind an API-routing proxy, which is exactly what env override is for.
  return "http://127.0.0.1:12000";
}

export const API_BASE = resolveApiBase();

interface RequestOptions extends RequestInit {
  timeout?: number;
}

export async function apiClient<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { timeout = 15000, headers = {}, ...rest } = options;

  // Ensure an authenticated token exists
  let token = getAuthToken();
  if (!token && typeof window !== "undefined" && !endpoint.includes("/auth/")) {
    token = await ensureAuthToken(API_BASE);
  }

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const requestHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(headers as Record<string, string>),
  };

  const url = endpoint.startsWith("http") ? endpoint : `${API_BASE}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;

  try {
    const res = await fetch(url, {
      ...rest,
      headers: requestHeaders,
      signal: controller.signal,
    });

    clearTimeout(id);

    if (res.status === 401) {
      // Re-provision token if expired and retry once
      clearAuth();
      const newToken = await ensureAuthToken(API_BASE, true);
      if (newToken) {
        requestHeaders.Authorization = `Bearer ${newToken}`;
        const retryRes = await fetch(url, { ...rest, headers: requestHeaders });
        if (retryRes.ok) return (await retryRes.json()) as T;
      }
      throw new Error("Authentication required (401 Unauthorized)");
    }

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errorData.detail || `API error ${res.status}: ${res.statusText}`);
    }

    const rawText = await res.text();
    if (!rawText.trim()) return {} as T;
    try {
      return JSON.parse(rawText) as T;
    } catch {
      const events = rawText
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => {
          try {
            return JSON.parse(l);
          } catch {
            return { raw: l };
          }
        });
      return { events } as unknown as T;
    }
  } catch (err: any) {
    clearTimeout(id);
    if (err.name === "AbortError") {
      throw new Error(`Request timeout (${timeout}ms) on ${endpoint}`);
    }
    throw err;
  }
}

// Workstation Specific API Endpoints
export const api = {
  getWorkstationState: (sessionId = "default") =>
    apiClient<any>(`/workstation/state?session_id=${encodeURIComponent(sessionId)}`, { timeout: 30000 }),

  getWorkstationAgents: (sessionId = "default") =>
    apiClient<any>(`/workstation/agents?session_id=${encodeURIComponent(sessionId)}`, { timeout: 30000 }),

  getWorkstationAgent: (agentUid: string, sessionId = "default") =>
    apiClient<any>(
      `/workstation/agents/${encodeURIComponent(agentUid)}?session_id=${encodeURIComponent(sessionId)}`,
      { timeout: 30000 },
    ),

  messageWorkstationAgent: (agentUid: string, message: string, sessionId = "default") =>
    apiClient<any>(
      `/workstation/agents/${encodeURIComponent(agentUid)}/message?session_id=${encodeURIComponent(sessionId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        timeout: 30000,
      },
    ),

  listSessions: async (): Promise<SessionItem[]> => {
    const data = await apiClient<any>("/workstation/sessions");
    return normalizeSessionList(data);
  },

  createSession: (sessionId?: string) =>
    apiClient<{ session_id: string; display_name?: string; status?: string }>(
      "/workstation/session",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sessionId ? { session_id: sessionId } : {}),
      },
    ),

  deleteSession: (sessionId: string) =>
    apiClient<any>(`/workstation/session?session_id=${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),

  interruptSession: (sessionId = "default") =>
    apiClient<any>(`/workstation/session/interrupt?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    }),

  getDesktopStatus: (sessionId = "default") =>
    apiClient<any>(`/workstation/desktop/status?session_id=${encodeURIComponent(sessionId)}`),

  provisionDesktop: (sessionId = "default") =>
    apiClient<any>(`/workstation/desktop/provision?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      timeout: 120000,
    }),

  provisionWorkstation: (sessionId = "default") =>
    apiClient<any>(`/workstation/desktop/provision?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      timeout: 120000,
    }),

  resumeWorkstation: (note = "", sessionId = "default") =>
    apiClient<any>(
      `/workstation/resume?session_id=${encodeURIComponent(sessionId)}&note=${encodeURIComponent(note)}`,
      { method: "POST" }
    ),

  getBeingState: (sessionId = "default") =>
    apiClient<any>(`/workstation/being?session_id=${encodeURIComponent(sessionId)}`),

  addVaultCredential: (
    payload: {
      service: string;
      username: string;
      secret?: string;
      token?: string;
      profile_url?: string;
    },
    sessionId = "default"
  ) =>
    apiClient<any>(`/workstation/vault?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ ...payload, session_id: sessionId }),
    }),

  provisionResearchLab: (sessionId = "default") =>
    apiClient<any>(`/workstation/research-lab/provision?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      timeout: 120000,
    }),

  getResearchLabStatus: (sessionId = "default") =>
    apiClient<any>(`/workstation/research-lab/status?session_id=${encodeURIComponent(sessionId)}`),

  destroyResearchLab: (sessionId = "default") =>
    apiClient<any>(`/workstation/research-lab?session_id=${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),

  provisionTargetSandbox: (target: string, scopeConfig: Record<string, unknown>, sessionId = "default") =>
    apiClient<any>(`/workstation/target-sandbox/provision?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ target, scope_config: scopeConfig }),
      timeout: 120000,
    }),

  getTargetSandboxStatus: (sessionId = "default") =>
    apiClient<any>(`/workstation/target-sandbox/status?session_id=${encodeURIComponent(sessionId)}`),

  destroyTargetSandbox: (sessionId = "default") =>
    apiClient<any>(`/workstation/target-sandbox?session_id=${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),

  executeTargetSandboxCommand: (command: string, sessionId = "default", approved = false) =>
    apiClient<any>(`/workstation/target-sandbox/command?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ command, approved }),
    }),

  startMission: (objective: string, sessionId = "default") =>
    apiClient<any>(`/workstation/mission/start?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ objective }),
    }),

  getMissionEvents: async (sessionId = "default", after = 0) => {
    const data = await apiClient<any>(
      `/workstation/mission/events?session_id=${encodeURIComponent(sessionId)}&after=${after}`
    );
    if (typeof data === "string") {
      const events = data
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return { raw: line };
          }
        });
      return { events };
    }
    if (Array.isArray(data)) {
      return { events: data };
    }
    return data;
  },

  getMissionEvidence: (sessionId = "default") =>
    apiClient<any>(`/workstation/mission/evidence?session_id=${encodeURIComponent(sessionId)}`),

  approveMissionProbe: (actionId: string, approved = true, sessionId = "default") =>
    apiClient<any>(`/workstation/mission/approve-probe?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ action_id: actionId, approved }),
    }),

  approveWorkstationAction: (approvalId: string, approved: boolean, sessionId = "default") =>
    apiClient<any>(`/workstation/action/approve?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ approval_id: approvalId, approved }),
    }),

  openMissionBrowser: (url: string, sessionId = "default", approved = false) =>
    apiClient<any>(`/workstation/mission/browser-open?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ url, approved }),
    }),

  getDesktopScreenshot: (sessionId = "default") =>
    apiClient<any>(`/workstation/desktop/screenshot?session_id=${encodeURIComponent(sessionId)}`),

  getDesktopStream: (sessionId = "default") =>
    apiClient<{ status: string; vnc_url: string | null; tenant_id: string }>(
      `/workstation/desktop/stream?session_id=${encodeURIComponent(sessionId)}`
    ),

  postGUIAction: (
    sessionId: string = "default",
    actionData: {
      action: string;
      x?: number;
      y?: number;
      text?: string;
      key?: string;
      app_name?: string;
      scroll_delta?: number;
    }
  ) =>
    apiClient<any>(`/workstation/desktop/gui-action?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify(actionData),
    }),

  executeDesktopAction: (actionData: {
    action: string;
    target?: string;
    coordinates?: [number, number];
    x?: number;
    y?: number;
    end_x?: number;
    end_y?: number;
    button?: number;
    direction?: string;
    amount?: number;
    text?: string;
    key?: string;
    delay_ms?: number;
    sessionId?: string;
  }) =>
    apiClient<any>("/workstation/desktop/action", {
      method: "POST",
      body: JSON.stringify({
        action: actionData.action,
        target: actionData.target,
        coordinates: actionData.coordinates,
        x: actionData.x,
        y: actionData.y,
        end_x: actionData.end_x,
        end_y: actionData.end_y,
        button: actionData.button,
        direction: actionData.direction,
        amount: actionData.amount,
        text: actionData.text,
        key: actionData.key,
        delay_ms: actionData.delay_ms,
        session_id: actionData.sessionId || "default",
      }),
    }),

  getFileTree: (sessionId = "default") =>
    apiClient<{ files: string[] }>(`/workstation/tree?session_id=${encodeURIComponent(sessionId)}`),

  getFileContent: (path: string, sessionId = "default") =>
    apiClient<any>(`/workstation/file?path=${encodeURIComponent(path)}&session_id=${encodeURIComponent(sessionId)}`),

  saveFileContent: (path: string, content: string, sessionId = "default") =>
    apiClient<any>(`/workstation/file?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: JSON.stringify({ path, content }),
    }),

  getGitDiff: (sessionId = "default") =>
    apiClient<{ diff: string; success: boolean }>(`/workstation/git-diff?session_id=${encodeURIComponent(sessionId)}`),

  sendPrompt: (prompt: string, sessionId = "default", mode = "normal") =>
    apiClient<any>("/workstation/prompt", {
      method: "POST",
      // Prompt submission may cold-start the provider/router and must not
      // be confused with the background mission duration.
      timeout: 30000,
      body: JSON.stringify({ prompt, session_id: sessionId, mode }),
    }),

  executeCommand: (command: string, sessionId = "default") =>
    apiClient<any>("/workstation/command", {
      method: "POST",
      body: JSON.stringify({ command, session_id: sessionId }),
    }),

  // Graph Memory
  getGraph: () =>
    apiClient<any>("/live/graph"),

  // Evidence
  getEvidence: () =>
    apiClient<any>("/live/evidence"),

  // Experiments / Self-Evolution
  getExperiments: () =>
    apiClient<any>("/live/experiments"),

  runBenchmark: () =>
    apiClient<any>("/live/experiments/benchmark", { method: "POST" }),

  // Settings
  getSettings: () =>
    apiClient<any>("/live/settings"),

  updateSettings: (config: any) =>
    apiClient<any>("/live/settings", {
      method: "POST",
      body: JSON.stringify(config),
    }),

  // Self-Security Lab
  runSecurityAudit: () =>
    apiClient<any>("/security/audit", { method: "POST", timeout: 60000 }),

  getSecurityTests: () =>
    apiClient<any>("/security/tests"),

  getSecurityFindings: () =>
    apiClient<any>("/security/findings"),

  getReleaseGate: () =>
    apiClient<any>("/security/release-gate"),

  getAttackSurface: () =>
    apiClient<any>("/security/attack-surface"),

  reproduceSecurityTest: (testId: string) =>
    apiClient<any>(`/security/reproduce/${encodeURIComponent(testId)}`, { method: "POST", timeout: 60000 }),

  // Experiment lifecycle
  approveExperiment: (id: string) =>
    apiClient<any>(`/experiments/${encodeURIComponent(id)}/approve`, { method: "POST" }),

  rejectExperiment: (id: string, reason: string) =>
    apiClient<any>(`/experiments/${encodeURIComponent(id)}/reject?reason=${encodeURIComponent(reason)}`, { method: "POST" }),

  promoteExperiment: (id: string) =>
    apiClient<any>(`/experiments/${encodeURIComponent(id)}/promote`, { method: "POST" }),

  getExperimentWeaknesses: () =>
    apiClient<any>("/experiments/weaknesses/summary"),

  getExperimentHistory: () =>
    apiClient<any>("/experiments/history/timeline"),

  // Organization / AI Employees
  getOrganization: () => apiClient<any>("/organization/"),

  createEmployee: (payload: any) =>
    apiClient<any>("/organization/employees", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  createWorkItem: (payload: any) =>
    apiClient<any>("/organization/work-items", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateWorkItemStatus: (workItemId: string, status: string) =>
    apiClient<any>(`/organization/work-items/${encodeURIComponent(workItemId)}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),

  // Agents
  getAgents: () =>
    apiClient<any>("/agents/"),

  getAgent: (agentId: string) =>
    apiClient<any>(`/agents/${encodeURIComponent(agentId)}`),

  // Persistent AI employee organization and work ownership
  createOrganizationEmployee: (payload: Record<string, unknown>) =>
    apiClient<any>("/organization/employees", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  createOrganizationProject: (payload: Record<string, unknown>) =>
    apiClient<any>("/organization/projects", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  createOrganizationWorkItem: (payload: Record<string, unknown>) =>
    apiClient<any>("/organization/work-items", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateOrganizationWorkItem: (workItemId: string, payload: Record<string, unknown>) =>
    apiClient<any>(`/organization/work-items/${encodeURIComponent(workItemId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  // Laya System-1 Decision Engine
  getLayaStatus: () =>
    apiClient<any>("/llm/laya/status"),

  predictLaya: (payload: {
    state: string | Record<string, unknown>;
    questions?: Record<string, Record<string, unknown>> | null;
    preset?: string | null;
    model?: string | null;
  }) =>
    apiClient<any>("/llm/laya/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
