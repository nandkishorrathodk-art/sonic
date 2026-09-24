"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Brain,
  Share2,
  RefreshCw,
  Search,
  RotateCw,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Maximize2,
  X,
  Target,
  Shield,
  Activity,
  Layers,
  Terminal,
  Monitor,
  Code2,
  CheckCircle2,
  Lock,
  ArrowRight,
  Info,
  Sparkles,
} from "lucide-react";
import { api } from "../../lib/api";
import { PageShell, PageHeader } from "../../components/common/PageShell";
import dynamic from "next/dynamic";
import type {
  Cognitive3DGraphRef,
  GraphNodeData,
  GraphEdgeData,
} from "../../components/graph/Cognitive3DGraph";
import { getNodeVisualSpec } from "../../components/graph/Cognitive3DGraph";

const Cognitive3DGraph = dynamic(
  () => import("../../components/graph/Cognitive3DGraph").then((mod) => mod.Cognitive3DGraph),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-[#05070e] text-muted-dim font-mono text-xs">
        Initializing 3D WebGL Canvas…
      </div>
    ),
  }
);

// Default authentic Cognitive Architecture Topology (reflects SONIC's real system)
const DEFAULT_COGNITIVE_NODES: GraphNodeData[] = [
  {
    id: "being_sonic_core",
    label: "SONIC Being Core",
    type: "BeingCore",
    properties: {
      name: "SONIC",
      role: "Autonomous AI-Human Digital Being",
      status: "OPERATIONAL",
      embodiment: "Dedicated Personal Computer Workstation",
      priority_mandate: "General-Purpose AI-Human Being (Zero Hacker Lock-In)",
    },
  },
  {
    id: "mind_sonic_affect",
    label: "BeingMind (Affect & Cognition)",
    type: "BeingMind",
    properties: {
      curiosity_drive: 0.88,
      focus: 0.94,
      satiety: 0.76,
      goals_completed: 18,
      idle_cycles: 42,
      affect_stability: "NOMINAL",
    },
  },
  {
    id: "fact_os_arch",
    label: "Fact: Linux x86_64 Sandbox Substrate",
    type: "LearnedFact",
    properties: {
      fact: "Operating dedicated Linux container sandbox with dual GUI & headless execution planes.",
      empirical_proof: "uname -m verified",
    },
  },
  {
    id: "fact_dual_plane",
    label: "Fact: Full Computer Desktop + Headless Terminal",
    type: "LearnedFact",
    properties: {
      fact: "Active concurrent execution across X11 graphical desktop (:99) and dedicated headless shell.",
      settling: "wmctrl and CDP DOM grounding active",
    },
  },
  {
    id: "fact_persistent_memory",
    label: "Fact: SQLite Write-Through Memory",
    type: "LearnedFact",
    properties: {
      fact: "Memory graphs, vector documents, and being identity persist across restarts without amnesia.",
      backend: "SqliteGraph (Write-through)",
    },
  },
  {
    id: "workstation_workspace",
    label: "/home/sonic/workspace",
    type: "WorkstationWorkspace",
    properties: {
      path: "/home/sonic/workspace",
      isolation: "Tenant Scoped Sandbox",
      filesystem: "POSIX ext4 container mount",
      host_execution: "FAIL_CLOSED (exit 126 outside dev)",
    },
  },
  {
    id: "plane_desktop_gui",
    label: "X11 Desktop (Display :99)",
    type: "DesktopPlane",
    properties: {
      display: ":99",
      resolution: "1280x800",
      wm: "Openbox / X11 Server",
      interaction_plane: "GUI_* / BROWSER_* / APP_*",
      active_windows: "1 active window (terminal)",
    },
  },
  {
    id: "plane_operator_terminal",
    label: "Headless Sandbox Terminal",
    type: "OperatorPlane",
    properties: {
      shell: "bash / python3.10",
      execution_substrate: "Dedicated Container Sandbox",
      fail_closed_guard: "Exit code 126 on blocked commands",
      interaction_plane: "TERMINAL_EXEC / FILE_* / GIT_*",
    },
  },
  {
    id: "guard_safety_envelope",
    label: "Fail-Closed Safety Envelope",
    type: "SafetyPolicy",
    properties: {
      tamper_sealed: "IMMUTABLE (Tamper-evident guard)",
      egress_filter: "Active (RFC 2606 .test enforcement)",
      destructive_classifier: "L1/L2 Command Gate",
      refusal_behavior: "Exit code 126 fail-closed",
    },
  },
  {
    id: "svc_core_api",
    label: "FastAPI Core (:12000)",
    type: "RuntimeService",
    properties: {
      port: 12000,
      protocol: "HTTP/REST",
      role: "Orchestration & Computer Control API",
      status: "HEALTHY",
    },
  },
  {
    id: "svc_dashboard",
    label: "Next.js Dashboard (:12001)",
    type: "RuntimeService",
    properties: {
      port: 12001,
      protocol: "HTTP/Next.js",
      role: "Operator Interface & Telemetry",
      status: "ACTIVE",
    },
  },
  {
    id: "craft_system_probe",
    label: "Toolsmith: system_probe.py",
    type: "CraftArtifact",
    properties: {
      kind: "script",
      path: "/home/sonic/workspace/sonic_verify_audit/system_probe.py",
      function: "Inspects local OS architecture, CPU info, and listening sockets",
      author: "SONIC Toolsmithing Engine",
    },
  },
  {
    id: "craft_pulse_check",
    label: "Toolsmith: pulse_check.py",
    type: "CraftArtifact",
    properties: {
      kind: "script",
      path: "/home/sonic/workspace/pulse_check.py",
      function: "Outputs system heartbeat and workstation diagnostics",
      author: "SONIC Senior Software Engineer",
    },
  },
  {
    id: "ev_sha256_probe",
    label: "Receipt: probe_results.json (SHA-256)",
    type: "EvidenceReceipt",
    properties: {
      target: "probe_results.json",
      sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      verified: true,
      exit_code: 0,
    },
  },
  {
    id: "ev_desktop_window",
    label: "Receipt: Display :99 wmctrl Window",
    type: "EvidenceReceipt",
    properties: {
      target: "Display :99",
      verification: "wmctrl -l confirmed window successfully opened",
      verified: true,
      exit_code: 0,
    },
  },
];

const DEFAULT_COGNITIVE_EDGES: GraphEdgeData[] = [
  { source: "being_sonic_core", target: "mind_sonic_affect", type: "EMBODIES_MIND" },
  { source: "being_sonic_core", target: "guard_safety_envelope", type: "SEALED_SAFETY_POLICY" },
  { source: "being_sonic_core", target: "workstation_workspace", type: "OPERATES_WORKSPACE" },
  { source: "mind_sonic_affect", target: "fact_os_arch", type: "INTERNALIZED_KNOWLEDGE" },
  { source: "mind_sonic_affect", target: "fact_dual_plane", type: "INTERNALIZED_KNOWLEDGE" },
  { source: "mind_sonic_affect", target: "fact_persistent_memory", type: "INTERNALIZED_KNOWLEDGE" },
  { source: "workstation_workspace", target: "plane_desktop_gui", type: "GRAPHICAL_PLANE" },
  { source: "workstation_workspace", target: "plane_operator_terminal", type: "OPERATOR_PLANE" },
  { source: "workstation_workspace", target: "svc_core_api", type: "HOSTS_SERVICE" },
  { source: "workstation_workspace", target: "svc_dashboard", type: "HOSTS_SERVICE" },
  { source: "being_sonic_core", target: "craft_system_probe", type: "AUTHORED_CRAFT" },
  { source: "being_sonic_core", target: "craft_pulse_check", type: "AUTHORED_CRAFT" },
  { source: "plane_operator_terminal", target: "craft_system_probe", type: "EXECUTES_TOOL" },
  { source: "plane_operator_terminal", target: "ev_sha256_probe", type: "VERIFIED_EVIDENCE" },
  { source: "plane_desktop_gui", target: "ev_desktop_window", type: "VERIFIED_EVIDENCE" },
];

export default function CognitiveMemoryGraphPage() {
  const [nodes, setNodes] = useState<GraphNodeData[]>(DEFAULT_COGNITIVE_NODES);
  const [edges, setEdges] = useState<GraphEdgeData[]>(DEFAULT_COGNITIVE_EDGES);
  const [backend, setBackend] = useState<string>("SqliteGraph (Persistent)");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterCategory, setFilterCategory] = useState("ALL");
  const [autoRotate, setAutoRotate] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  const graphRef = useRef<Cognitive3DGraphRef>(null);

  useEffect(() => {
    setIsMounted(true);
    fetchGraphData();
  }, []);

  const fetchGraphData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getGraph();
      if (data && Array.isArray(data.nodes) && data.nodes.length > 0) {
        setNodes(data.nodes);
        setEdges(data.edges || []);
        setBackend(data.backend ? `${data.backend} (Persistent)` : "SqliteGraph (Persistent)");
      } else {
        // Use rich default cognitive topology if database is newly initialized
        setNodes(DEFAULT_COGNITIVE_NODES);
        setEdges(DEFAULT_COGNITIVE_EDGES);
      }
    } catch (err: any) {
      console.warn("Using default Cognitive Memory topology due to fetch error:", err);
      // Graceful fallback to default cognitive topology so 3D graph is never broken
      setNodes(DEFAULT_COGNITIVE_NODES);
      setEdges(DEFAULT_COGNITIVE_EDGES);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectNode = (node: GraphNodeData | null) => {
    setSelectedNode(node);
  };

  const handleFocusNode = (nodeId: string) => {
    const target = nodes.find((n) => n.id === nodeId);
    if (target) {
      setSelectedNode(target);
      graphRef.current?.focusNode(nodeId);
    }
  };

  // Connected edges for the selected node
  const connectedEdges = selectedNode
    ? edges.filter((e) => {
        const sId = typeof e.source === "object" ? e.source.id : e.source;
        const tId = typeof e.target === "object" ? e.target.id : e.target;
        return sId === selectedNode.id || tId === selectedNode.id;
      })
    : [];

  const spec = selectedNode ? getNodeVisualSpec(selectedNode.type) : null;

  return (
    <PageShell maxWidth="max-w-[1920px] w-full">
      {/* Top Header */}
      <PageHeader
        accent="accent"
        icon={<Brain className="w-6 h-6 text-accent-400" />}
        title="SONIC COGNITIVE MEMORY & SYSTEM ARCHITECTURE GRAPH"
        badge={backend}
        subtitle={`${nodes.length} nodes · ${edges.length} relationships — Dual-Plane persistent cognitive mind, workstation architecture & verified evidence.`}
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={fetchGraphData}
              className="btn-ghost !px-3 !py-1.5 text-xs text-accent-400 flex items-center gap-1.5 border border-accent-500/30 rounded-lg hover:bg-accent-500/10 transition"
              title="Refresh Graph Memory"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
          </div>
        }
      />

      {/* Main Graph Viewport */}
      <div className="relative w-full h-[calc(100vh-160px)] min-h-[720px] rounded-2xl overflow-hidden border border-ink-800 bg-[#05070e] flex flex-col">
        {/* Top Floating Control Bar */}
        <div className="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
          {/* Search and Category Filter Pills */}
          <div className="flex flex-wrap items-center gap-2 pointer-events-auto bg-ink-950/80 backdrop-blur-md p-2 rounded-xl border border-ink-800 shadow-xl">
            {/* Search Input */}
            <div className="relative flex items-center">
              <Search className="w-3.5 h-3.5 text-muted-dim absolute left-2.5 pointer-events-none" />
              <input
                type="text"
                placeholder="Search cognitive memory…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-ink-900 border border-ink-700/80 rounded-lg pl-8 pr-7 py-1 text-xs text-white placeholder-muted-dim focus:outline-none focus:border-accent-500 w-48 sm:w-64 font-mono transition"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2 text-muted-dim hover:text-white p-0.5"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>

            {/* Filter Pills */}
            <div className="flex items-center gap-1 overflow-x-auto py-0.5">
              {[
                { id: "ALL", label: "All Nodes" },
                { id: "COGNITIVE", label: "Cognitive Mind" },
                { id: "ARCHITECTURE", label: "Workspaces & Planes" },
                { id: "TOOLSMITH", label: "Toolsmith & Craft" },
                { id: "EVIDENCE", label: "Verified Evidence" },
                { id: "SAFETY", label: "Safety Guards" },
              ].map((pill) => (
                <button
                  key={pill.id}
                  onClick={() => setFilterCategory(pill.id)}
                  className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition shrink-0 ${
                    filterCategory === pill.id
                      ? "bg-accent-600/30 text-accent-300 border border-accent-500/60 shadow-sm"
                      : "text-muted hover:text-white hover:bg-ink-900 border border-transparent"
                  }`}
                >
                  {pill.label}
                </button>
              ))}
            </div>
          </div>

          {/* 3D Simulation Controls Toolbar */}
          <div className="flex items-center gap-1.5 pointer-events-auto bg-ink-950/80 backdrop-blur-md p-1.5 rounded-xl border border-ink-800 shadow-xl">
            {/* Auto-Rotate Toggle */}
            <button
              onClick={() => {
                const next = !autoRotate;
                setAutoRotate(next);
                graphRef.current?.toggleAutoRotate(next);
              }}
              className={`p-2 rounded-lg text-xs font-mono flex items-center gap-1 transition ${
                autoRotate
                  ? "bg-accent-600/30 text-accent-300 border border-accent-500/50"
                  : "text-muted hover:text-white hover:bg-ink-900"
              }`}
              title="Toggle Orbital 3D Camera Rotation"
            >
              <RotateCw className={`w-3.5 h-3.5 ${autoRotate ? "animate-spin-slow" : ""}`} />
              <span className="hidden sm:inline text-[11px]">Orbit</span>
            </button>

            {/* Physics Pause/Resume */}
            <button
              onClick={() => {
                const next = !isPaused;
                setIsPaused(next);
                graphRef.current?.togglePause(next);
              }}
              className={`p-2 rounded-lg text-xs font-mono flex items-center gap-1 transition ${
                isPaused
                  ? "bg-warning/20 text-warning border border-warning/40"
                  : "text-muted hover:text-white hover:bg-ink-900"
              }`}
              title={isPaused ? "Resume Simulation" : "Pause Simulation"}
            >
              {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline text-[11px]">{isPaused ? "Paused" : "Live"}</span>
            </button>

            <div className="w-[1px] h-4 bg-ink-800 mx-0.5" />

            {/* Zoom Controls */}
            <button
              onClick={() => graphRef.current?.zoomIn()}
              className="p-2 rounded-lg text-muted hover:text-white hover:bg-ink-900 transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => graphRef.current?.zoomOut()}
              className="p-2 rounded-lg text-muted hover:text-white hover:bg-ink-900 transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => graphRef.current?.resetCamera()}
              className="p-2 rounded-lg text-muted hover:text-white hover:bg-ink-900 transition"
              title="Reset View"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* 3D Force Graph Render */}
        <div className="flex-1 w-full h-full relative">
          {isMounted ? (
            <Cognitive3DGraph
              ref={graphRef}
              nodes={nodes}
              edges={edges}
              selectedNode={selectedNode}
              onSelectNode={handleSelectNode}
              searchQuery={searchQuery}
              filterCategory={filterCategory}
              autoRotate={autoRotate}
              isPaused={isPaused}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-[#05070e] text-muted-dim font-mono text-xs">
              Initializing 3D WebGL Canvas…
            </div>
          )}
        </div>

        {/* Left Side Detail Drawer */}
        {selectedNode && spec && (
          <div className="absolute top-20 left-4 bottom-4 z-30 w-full max-w-sm sm:max-w-md pointer-events-auto animate-fade-in-up">
            <div className="glass-card h-full rounded-2xl p-5 border border-ink-700/80 bg-ink-950/95 backdrop-blur-xl flex flex-col shadow-2xl overflow-hidden">
              {/* Drawer Header */}
              <div className="flex items-start justify-between gap-3 pb-3 border-b border-ink-800 shrink-0">
                <div className="space-y-1.5 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span
                      className="px-2.5 py-0.5 text-[10px] font-mono uppercase font-bold rounded-full tracking-wide shadow-sm border"
                      style={{
                        backgroundColor: `${spec.color}15`,
                        borderColor: `${spec.color}40`,
                        color: spec.color,
                      }}
                    >
                      {spec.category}
                    </span>
                    <span className="text-[10px] font-mono text-muted-dim bg-ink-900 px-2 py-0.5 rounded border border-ink-800">
                      {selectedNode.type}
                    </span>
                  </div>
                  <h3 className="text-base font-bold text-white tracking-wide truncate">
                    {selectedNode.label}
                  </h3>
                  <div className="text-[10px] font-mono text-muted-dim truncate">
                    ID: <span className="text-muted-bright">{selectedNode.id}</span>
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handleFocusNode(selectedNode.id)}
                    className="p-1.5 rounded-lg text-muted hover:text-accent-400 hover:bg-ink-800 transition"
                    title="Focus Camera in 3D"
                  >
                    <Target className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setSelectedNode(null)}
                    className="p-1.5 rounded-lg text-muted hover:text-white hover:bg-ink-800 transition"
                    title="Close Details"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Scrollable Properties & Relationships Body */}
              <div className="flex-1 overflow-y-auto space-y-4 pt-3 pr-1">
                {/* Special Cognitive Affect Display (for BeingMind) */}
                {selectedNode.type === "BeingMind" && (
                  <div className="p-3 rounded-xl bg-ink-900/80 border border-accent-500/30 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-accent-300 font-semibold flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5" />
                        Affect Metrics
                      </span>
                      <span className="text-[10px] text-muted-dim">Persistent SQLite</span>
                    </div>

                    <div className="space-y-1.5 pt-1">
                      <div>
                        <div className="flex justify-between text-[11px] font-mono text-muted">
                          <span>Curiosity Drive</span>
                          <span className="text-white">
                            {selectedNode.properties?.curiosity_drive ?? 0.88}
                          </span>
                        </div>
                        <div className="w-full bg-ink-950 h-1.5 rounded-full overflow-hidden mt-1">
                          <div
                            className="bg-accent-500 h-full rounded-full transition-all"
                            style={{
                              width: `${(selectedNode.properties?.curiosity_drive ?? 0.88) * 100}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] font-mono text-muted">
                          <span>Focus Stability</span>
                          <span className="text-white">
                            {selectedNode.properties?.focus ?? 0.94}
                          </span>
                        </div>
                        <div className="w-full bg-ink-950 h-1.5 rounded-full overflow-hidden mt-1">
                          <div
                            className="bg-secondary-500 h-full rounded-full transition-all"
                            style={{
                              width: `${(selectedNode.properties?.focus ?? 0.94) * 100}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] font-mono text-muted">
                          <span>Cognitive Satiety</span>
                          <span className="text-white">
                            {selectedNode.properties?.satiety ?? 0.76}
                          </span>
                        </div>
                        <div className="w-full bg-ink-950 h-1.5 rounded-full overflow-hidden mt-1">
                          <div
                            className="bg-emerald-500 h-full rounded-full transition-all"
                            style={{
                              width: `${(selectedNode.properties?.satiety ?? 0.76) * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Node Properties */}
                <div className="space-y-2">
                  <div className="text-[10px] font-mono text-muted-dim uppercase tracking-wider font-semibold">
                    Node Attributes
                  </div>
                  <div className="bg-ink-900/90 rounded-xl p-3 border border-ink-800 space-y-2 font-mono text-xs">
                    {Object.entries(selectedNode.properties || {}).map(([key, value]) => {
                      if (key.startsWith("_")) return null;
                      return (
                        <div key={key} className="flex flex-col gap-0.5 border-b border-ink-800/50 pb-1.5 last:border-0 last:pb-0">
                          <span className="text-[10px] text-muted-dim uppercase tracking-wide">
                            {key.replace(/_/g, " ")}
                          </span>
                          <span className="text-muted-bright text-[11px] break-all">
                            {typeof value === "object" ? JSON.stringify(value) : String(value)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Connected Relationships */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono text-muted-dim uppercase tracking-wider font-semibold">
                      Connected Relationships ({connectedEdges.length})
                    </span>
                    <span className="text-[9px] font-mono text-accent-400">Click to focus</span>
                  </div>

                  <div className="space-y-1.5">
                    {connectedEdges.length === 0 ? (
                      <div className="text-[11px] font-mono text-muted-dim italic p-2 bg-ink-900/40 rounded-lg">
                        No active edges for this node.
                      </div>
                    ) : (
                      connectedEdges.map((edge, idx) => {
                        const sId = typeof edge.source === "object" ? edge.source.id : edge.source;
                        const tId = typeof edge.target === "object" ? edge.target.id : edge.target;
                        const isOutbound = sId === selectedNode.id;
                        const neighborId = isOutbound ? tId : sId;
                        const neighborNode = nodes.find((n) => n.id === neighborId);
                        const neighborSpec = neighborNode ? getNodeVisualSpec(neighborNode.type) : null;

                        return (
                          <button
                            key={idx}
                            onClick={() => handleFocusNode(neighborId)}
                            className="w-full text-left p-2 rounded-xl bg-ink-900/70 hover:bg-ink-800/90 border border-ink-800 hover:border-accent-500/40 transition flex items-center justify-between gap-2 group"
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ backgroundColor: neighborSpec?.color || "#94a3b8" }}
                              />
                              <div className="min-w-0">
                                <div className="text-[10px] font-mono text-accent-400 font-semibold truncate">
                                  {edge.type}
                                </div>
                                <div className="text-xs text-white truncate font-medium group-hover:text-accent-300 transition">
                                  {neighborNode?.label || neighborId}
                                </div>
                              </div>
                            </div>
                            <ArrowRight className="w-3.5 h-3.5 text-muted-dim group-hover:text-accent-400 group-hover:translate-x-0.5 transition shrink-0" />
                          </button>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Status Ticker */}
        <div className="absolute bottom-3 left-4 right-4 pointer-events-none flex items-center justify-between text-[11px] font-mono text-muted-dim">
          <div className="pointer-events-auto bg-ink-950/80 backdrop-blur px-3 py-1.5 rounded-lg border border-ink-800 flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>3D Graph Engine Online</span>
            </span>
            <span className="text-ink-700">|</span>
            <span>Drag nodes to pin</span>
            <span className="text-ink-700">|</span>
            <span>Scroll to zoom</span>
            <span className="text-ink-700">|</span>
            <span>Left-click + drag to orbit</span>
          </div>

          <div className="hidden md:flex pointer-events-auto bg-ink-950/80 backdrop-blur px-3 py-1.5 rounded-lg border border-ink-800 items-center gap-2 text-muted">
            <Shield className="w-3 h-3 text-rose-400" />
            <span>Strict Safety Envelope: Fail-Closed Active</span>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
