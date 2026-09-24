"use client";

import React, { useEffect, useRef, useImperativeHandle, forwardRef } from "react";
import * as THREE from "three";

export interface GraphNodeData {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
  [key: string]: any;
}

export interface GraphEdgeData {
  source: string | GraphNodeData;
  target: string | GraphNodeData;
  type: string;
  [key: string]: any;
}

export interface Cognitive3DGraphRef {
  zoomIn: () => void;
  zoomOut: () => void;
  resetCamera: () => void;
  focusNode: (nodeId: string) => void;
  toggleAutoRotate: (enabled: boolean) => void;
  togglePause: (paused: boolean) => void;
}

interface Cognitive3DGraphProps {
  nodes: GraphNodeData[];
  edges: GraphEdgeData[];
  selectedNode: GraphNodeData | null;
  onSelectNode: (node: GraphNodeData | null) => void;
  searchQuery?: string;
  filterCategory?: string;
  autoRotate?: boolean;
  isPaused?: boolean;
}

export function getNodeVisualSpec(type: string): {
  category: string;
  color: string;
  particleColor: string;
  size: number;
  isMajor: boolean;
} {
  const t = (type || "").toLowerCase();
  if (t.includes("beingcore") || t.includes("being_core") || t.includes("beingidentity")) {
    return { category: "Cognitive Core", color: "#a855f7", particleColor: "#c084fc", size: 12, isMajor: true };
  }
  if (t.includes("beingmind") || t.includes("cognitivestate")) {
    return { category: "Being Mind", color: "#8b5cf6", particleColor: "#a78bfa", size: 9, isMajor: true };
  }
  if (t.includes("learnedfact") || t.includes("fact")) {
    return { category: "Learned Facts", color: "#6366f1", particleColor: "#818cf8", size: 4.8, isMajor: false };
  }
  if (t.includes("workspace") || t.includes("workstationworkspace") || t.includes("repository")) {
    return { category: "Workspaces", color: "#0ea5e9", particleColor: "#38bdf8", size: 10, isMajor: true };
  }
  if (t.includes("desktop") || t.includes("desktopplane") || t.includes("display")) {
    return { category: "Desktop Plane", color: "#06b6d4", particleColor: "#22d3ee", size: 8, isMajor: true };
  }
  if (t.includes("operator") || t.includes("terminal") || t.includes("operatorplane")) {
    return { category: "Operator Plane", color: "#14b8a6", particleColor: "#2dd4bf", size: 8, isMajor: true };
  }
  if (t.includes("craft") || t.includes("toolsmith") || t.includes("script")) {
    return { category: "Toolsmith & Craft", color: "#f59e0b", particleColor: "#fbbf24", size: 6.5, isMajor: false };
  }
  if (t.includes("service") || t.includes("runtimeservice") || t.includes("port")) {
    return { category: "Runtime Services", color: "#3b82f6", particleColor: "#60a5fa", size: 6.5, isMajor: false };
  }
  if (t.includes("evidence") || t.includes("receipt") || t.includes("verified")) {
    return { category: "Verified Evidence", color: "#10b981", particleColor: "#34d399", size: 7, isMajor: false };
  }
  if (t.includes("safety") || t.includes("policy") || t.includes("guard")) {
    return { category: "Safety Envelope", color: "#f43f5e", particleColor: "#fb7185", size: 9, isMajor: true };
  }
  return { category: "System Entity", color: "#94a3b8", particleColor: "#cbd5e1", size: 5, isMajor: false };
}

function createTextSprite(text: string, color: string = "#ffffff", fontSize: number = 22): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.Sprite();

  canvas.width = 512;
  canvas.height = 128;
  ctx.fillStyle = "rgba(0,0,0,0)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.font = `600 ${fontSize}px "JetBrains Mono", Consolas, Menlo, monospace, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const metrics = ctx.measureText(text);
  const textWidth = Math.min(metrics.width + 32, 480);
  const x = canvas.width / 2 - textWidth / 2;
  const y = 36;
  const h = fontSize + 18;
  const r = 8;

  // Background pill with border
  ctx.fillStyle = "rgba(7, 10, 20, 0.85)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, textWidth, h, r);
  } else {
    ctx.rect(x, y, textWidth, h);
  }
  ctx.fill();
  ctx.stroke();

  // Crisp text
  ctx.fillStyle = color;
  ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
  ctx.shadowBlur = 6;
  ctx.fillText(text, canvas.width / 2, y + h / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;

  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });

  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(38, 9.5, 1);
  return sprite;
}

export const Cognitive3DGraph = forwardRef<Cognitive3DGraphRef, Cognitive3DGraphProps>(
  (
    {
      nodes,
      edges,
      selectedNode,
      onSelectNode,
      searchQuery = "",
      filterCategory = "ALL",
      autoRotate = true,
      isPaused = false,
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const graphInstanceRef = useRef<any>(null);
    const rotationFrameRef = useRef<number | null>(null);
    const isPausedRef = useRef<boolean>(isPaused);
    const autoRotateRef = useRef<boolean>(autoRotate);
    const selectedNodeRef = useRef<GraphNodeData | null>(selectedNode);

    isPausedRef.current = isPaused;
    autoRotateRef.current = autoRotate;
    selectedNodeRef.current = selectedNode;

    // Filter nodes based on Category and Search Query
    const query = searchQuery.trim().toLowerCase();
    const visibleNodeIds = new Set<string>();

    nodes.forEach((node) => {
      const spec = getNodeVisualSpec(node.type);
      const categoryMatch =
        filterCategory === "ALL" ||
        spec.category.toLowerCase() === filterCategory.toLowerCase() ||
        (filterCategory === "COGNITIVE" && (spec.category.includes("Cognitive") || spec.category.includes("Mind") || spec.category.includes("Facts"))) ||
        (filterCategory === "ARCHITECTURE" && (spec.category.includes("Workspace") || spec.category.includes("Plane") || spec.category.includes("Service"))) ||
        (filterCategory === "TOOLSMITH" && spec.category.includes("Toolsmith")) ||
        (filterCategory === "EVIDENCE" && spec.category.includes("Evidence")) ||
        (filterCategory === "SAFETY" && spec.category.includes("Safety"));

      const searchMatch =
        !query ||
        node.label.toLowerCase().includes(query) ||
        node.type.toLowerCase().includes(query) ||
        node.id.toLowerCase().includes(query) ||
        JSON.stringify(node.properties || {}).toLowerCase().includes(query);

      if (categoryMatch && searchMatch) {
        visibleNodeIds.add(node.id);
      }
    });

    // If query matches nodes, also include their direct neighbors so graph remains cohesive
    if (query && visibleNodeIds.size > 0) {
      edges.forEach((edge) => {
        const sId = typeof edge.source === "object" ? edge.source.id : edge.source;
        const tId = typeof edge.target === "object" ? edge.target.id : edge.target;
        if (visibleNodeIds.has(sId) || visibleNodeIds.has(tId)) {
          visibleNodeIds.add(sId);
          visibleNodeIds.add(tId);
        }
      });
    }

    const filteredNodes = nodes.filter((n) => visibleNodeIds.has(n.id));
    const filteredEdges = edges.filter((e) => {
      const sId = typeof e.source === "object" ? e.source.id : e.source;
      const tId = typeof e.target === "object" ? e.target.id : e.target;
      return visibleNodeIds.has(sId) && visibleNodeIds.has(tId);
    });

    // Expose camera controls to parent
    useImperativeHandle(ref, () => ({
      zoomIn: () => {
        if (!graphInstanceRef.current) return;
        const currentPos = graphInstanceRef.current.cameraPosition();
        graphInstanceRef.current.cameraPosition(
          { x: currentPos.x * 0.75, y: currentPos.y * 0.75, z: currentPos.z * 0.75 },
          undefined,
          600
        );
      },
      zoomOut: () => {
        if (!graphInstanceRef.current) return;
        const currentPos = graphInstanceRef.current.cameraPosition();
        graphInstanceRef.current.cameraPosition(
          { x: currentPos.x * 1.3, y: currentPos.y * 1.3, z: currentPos.z * 1.3 },
          undefined,
          600
        );
      },
      resetCamera: () => {
        if (!graphInstanceRef.current) return;
        graphInstanceRef.current.cameraPosition({ x: 0, y: 40, z: 280 }, { x: 0, y: 0, z: 0 }, 1200);
      },
      focusNode: (nodeId: string) => {
        if (!graphInstanceRef.current) return;
        const targetNode = filteredNodes.find((n) => n.id === nodeId);
        if (targetNode && targetNode.x !== undefined && targetNode.y !== undefined && targetNode.z !== undefined) {
          const distance = 90;
          const distRatio = 1 + distance / Math.hypot(targetNode.x, targetNode.y, targetNode.z);
          graphInstanceRef.current.cameraPosition(
            { x: targetNode.x * distRatio, y: targetNode.y * distRatio + 15, z: targetNode.z * distRatio },
            targetNode,
            1200
          );
        }
      },
      toggleAutoRotate: (enabled: boolean) => {
        autoRotateRef.current = enabled;
      },
      togglePause: (paused: boolean) => {
        isPausedRef.current = paused;
        if (graphInstanceRef.current) {
          if (paused) {
            graphInstanceRef.current.pauseAnimation();
          } else {
            graphInstanceRef.current.resumeAnimation();
          }
        }
      },
    }));

    // Initialize 3D Force Graph
    useEffect(() => {
      let isMounted = true;
      let graph: any = null;

      const initGraph = async () => {
        if (!containerRef.current) return;

        // Dynamic client-side import of 3d-force-graph
        const ForceGraph3DModule = await import("3d-force-graph");
        const ForceGraph3D = (ForceGraph3DModule.default || ForceGraph3DModule) as any;

        if (!isMounted || !containerRef.current) return;

        // Clean any existing canvas in container
        containerRef.current.innerHTML = "";

        const width = containerRef.current.clientWidth || window.innerWidth;
        const height = containerRef.current.clientHeight || 750;

        const factory = typeof ForceGraph3D === "function" ? ForceGraph3D : ForceGraph3D.default;
        graph = factory()(containerRef.current)
          .width(width)
          .height(height)
          .backgroundColor("#05070e")
          .showNavInfo(false)
          .enableNodeDrag(true)
          .nodeId("id")
          .nodeVal((node: any) => {
            const spec = getNodeVisualSpec(node.type);
            return spec.size;
          })
          .nodeThreeObject((node: any) => {
            const spec = getNodeVisualSpec(node.type);
            const isSelected = selectedNodeRef.current?.id === node.id;
            const isHighlighted = query ? node.label.toLowerCase().includes(query) : false;

            const group = new THREE.Group();
            const radius = spec.size * (isSelected ? 1.3 : 1.0);

            // Core sphere with emissive glow
            const sphereGeo = new THREE.SphereGeometry(radius, 24, 24);
            const sphereMat = new THREE.MeshStandardMaterial({
              color: isSelected ? "#ffffff" : spec.color,
              roughness: 0.25,
              metalness: 0.8,
              emissive: spec.color,
              emissiveIntensity: isSelected ? 0.9 : isHighlighted ? 0.75 : 0.45,
            });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            group.add(sphere);

            // Glowing Outer Ring for major components or selected node
            if (spec.isMajor || isSelected) {
              const ringGeo = new THREE.RingGeometry(radius * 1.35, radius * 1.6, 32);
              const ringMat = new THREE.MeshBasicMaterial({
                color: isSelected ? "#38bdf8" : spec.color,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: isSelected ? 0.8 : 0.35,
              });
              const ring = new THREE.Mesh(ringGeo, ringMat);
              ring.rotation.x = Math.PI / 2;
              group.add(ring);
            }

            // Floating Label Sprite
            const labelText = node.label || node.id;
            const displayLabel = labelText.length > 28 ? labelText.slice(0, 25) + "…" : labelText;
            const sprite = createTextSprite(displayLabel, isSelected ? "#38bdf8" : spec.color, 22);
            sprite.position.y = radius + 6.5;
            group.add(sprite);

            return group;
          })
          .linkWidth((link: any) => {
            const isConnectedToSelected =
              selectedNodeRef.current &&
              (link.source.id === selectedNodeRef.current.id || link.target.id === selectedNodeRef.current.id);
            return isConnectedToSelected ? 2.5 : 1.2;
          })
          .linkColor((link: any) => {
            const isConnectedToSelected =
              selectedNodeRef.current &&
              (link.source.id === selectedNodeRef.current.id || link.target.id === selectedNodeRef.current.id);
            return isConnectedToSelected ? "rgba(56, 189, 248, 0.85)" : "rgba(99, 102, 241, 0.35)";
          })
          .linkDirectionalParticles(2)
          .linkDirectionalParticleSpeed(0.006)
          .linkDirectionalParticleWidth(2.2)
          .linkDirectionalParticleColor((link: any) => {
            const targetSpec = getNodeVisualSpec(link.target?.type || "");
            return targetSpec.particleColor || "#38bdf8";
          })
          .onNodeClick((node: any) => {
            onSelectNode(node);
            // Smoothly move camera towards clicked node
            if (node.x !== undefined && node.y !== undefined && node.z !== undefined) {
              const distance = 85;
              const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
              graph.cameraPosition(
                { x: node.x * distRatio, y: node.y * distRatio + 12, z: node.z * distRatio },
                node,
                1200
              );
            }
          })
          .onBackgroundClick(() => {
            onSelectNode(null);
          });

        // Set warm ambient lighting for 3D glow
        const scene = graph.scene();
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0x38bdf8, 2.5, 400);
        pointLight.position.set(0, 50, 100);
        scene.add(pointLight);

        // Load data
        graph.graphData({
          nodes: filteredNodes.map((n) => ({ ...n })),
          links: filteredEdges.map((e) => ({ ...e })),
        });

        graphInstanceRef.current = graph;

        // Auto-rotation loop
        let angle = 0;
        const rotate = () => {
          if (autoRotateRef.current && !selectedNodeRef.current && graphInstanceRef.current) {
            angle += 0.0018;
            const distance = 280;
            graphInstanceRef.current.cameraPosition({
              x: distance * Math.sin(angle),
              y: 40 + Math.sin(angle * 0.5) * 15,
              z: distance * Math.cos(angle),
            });
          }
          rotationFrameRef.current = requestAnimationFrame(rotate);
        };
        rotationFrameRef.current = requestAnimationFrame(rotate);

        // Handle window resize
        const handleResize = () => {
          if (containerRef.current && graphInstanceRef.current) {
            graphInstanceRef.current.width(containerRef.current.clientWidth);
            graphInstanceRef.current.height(containerRef.current.clientHeight);
          }
        };
        window.addEventListener("resize", handleResize);
      };

      initGraph();

      return () => {
        isMounted = false;
        if (rotationFrameRef.current) {
          cancelAnimationFrame(rotationFrameRef.current);
        }
        if (graphInstanceRef.current) {
          try {
            graphInstanceRef.current._destructor();
          } catch {
            // cleanup safely
          }
        }
      };
    }, []);

    // Update graph data when nodes or edges change
    useEffect(() => {
      if (!graphInstanceRef.current) return;
      graphInstanceRef.current.graphData({
        nodes: filteredNodes.map((n) => ({ ...n })),
        links: filteredEdges.map((e) => ({ ...e })),
      });
      // Re-trigger node styling refresh
      graphInstanceRef.current.nodeThreeObject(graphInstanceRef.current.nodeThreeObject());
    }, [nodes, edges, searchQuery, filterCategory, selectedNode]);

    // Handle pause/resume physics
    useEffect(() => {
      if (!graphInstanceRef.current) return;
      if (isPaused) {
        graphInstanceRef.current.pauseAnimation();
      } else {
        graphInstanceRef.current.resumeAnimation();
      }
    }, [isPaused]);

    return (
      <div className="relative w-full h-full min-h-[680px] rounded-2xl overflow-hidden border border-ink-800/80 bg-[#05070e] shadow-2xl">
        <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      </div>
    );
  }
);

Cognitive3DGraph.displayName = "Cognitive3DGraph";
