"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { RefreshCw, ArrowLeft } from "lucide-react";
import { api } from "../../lib/api";
import { ComputerSurface } from "../../components/computer/ComputerSurface";
import { CommandResult } from "../../types/workstation";

export default function ComputerWorkspacePage() {
  const [desktopState, setDesktopState] = useState<any>(null);
  const [commandLogs, setCommandLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const stateData = await api.getWorkstationState().catch(() => null);
      const data = await api.getDesktopStatus();
      setDesktopState(data);
      if (stateData?.worklog && Array.isArray(stateData.worklog)) {
        const logs: string[] = [];
        for (const entry of stateData.worklog) {
          if (entry.type === "command" && entry.command) {
            logs.push(`sonic@workstation:~$ ${entry.command}`);
            if (entry.output) logs.push(entry.output.trim());
          }
        }
        setCommandLogs((prevManual) => {
          const combined = [...logs];
          for (const m of prevManual) {
            if (!combined.includes(m)) combined.push(m);
          }
          return combined;
        });
      }
    } catch (err: any) {
      setError(err.message || "Failed to query computer status.");
    } finally {
      setLoading(false);
    }
  };

  const provisionDesktop = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.provisionDesktop();
      await fetchStatus();
    } catch (err: any) {
      setError(err.message || "Failed to provision Cyber Workstation.");
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleRunCommand = async (cmd: string): Promise<CommandResult | null> => {
    setCommandLogs((prev) => [...prev, `sonic@workstation:~$ ${cmd}`]);
    try {
      const res = await api.executeCommand(cmd);
      if (res?.output) setCommandLogs((prev) => [...prev, res.output.trim()]);
      await fetchStatus();
      return res;
    } catch (err: any) {
      setCommandLogs((prev) => [...prev, `[FAIL-CLOSED REJECTED]: ${err.message}`]);
      return null;
    }
  };

  return (
    <div className="min-h-screen bg-ink-950 bg-grid-glow max-w-7xl mx-auto p-6 space-y-6 flex flex-col h-[calc(100vh-0px)]">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card rounded-2xl p-4 flex-shrink-0">
        <div className="flex items-center gap-4">
          <div className="grid place-items-center w-10 h-10 rounded-xl bg-gradient-to-tr from-primary-600 to-secondary-600 p-0.5 shadow-glow">
            <div className="bg-ink-950 w-full h-full rounded-[10px] grid place-items-center text-secondary-400 font-mono font-black text-sm">S</div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-white tracking-wide">CYBER WORKSTATION</h2>
              {desktopState?.vnc_url ? (
                <span className="chip border border-success/40 bg-success/10 text-success">
                  <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" /> LIVE DESKTOP
                </span>
              ) : (
                <span className="chip border border-ink-700 bg-ink-850 text-muted">
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-dim" /> OFFLINE
                </span>
              )}
            </div>
            <p className="text-xs text-muted mt-0.5 font-mono">
              Real Docker Sandbox Execution: X11 GUI Desktop, PTY Shell, Filesystem, and Native Computer Use.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/?tab=desktop"
            className="btn-ghost !px-3 !py-1.5 text-xs font-mono text-muted hover:text-white flex items-center gap-1.5 border border-ink-700/60 rounded-lg hover:bg-ink-800 transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Dual-Plane Workstation</span>
          </Link>
          <button onClick={fetchStatus} className="btn-secondary !px-3 !py-1.5 text-xs font-mono">
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh Desktop</span>
          </button>
        </div>
      </div>

      {error ? (
        <div className="p-4 rounded-xl bg-danger/10 border border-danger/30 text-danger text-xs font-mono flex-shrink-0">
          <strong>Backend Error:</strong> {error}
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 overflow-hidden">
          <ComputerSurface desktopState={desktopState} onRunCommand={handleRunCommand} commandLogs={commandLogs} onProvision={provisionDesktop} />
        </div>
      )}
    </div>
  );
}
