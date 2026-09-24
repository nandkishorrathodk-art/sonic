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
  hexColor: number;
  particleColor: string;
  size: number;
  isMajor: boolean;
} {
  const t = (type || "").toLowerCase();
  if (t.includes("beingcore") || t.includes("being_core") || t.includes("beingidentity")) {
    return { category: "Cognitive Core", color: "#a855f7", hexColor: 0xa855f7, particleColor: "#c084fc", size: 10, isMajor: true };
  }
  if (t.includes("beingmind") || t.includes("cognitivestate")) {
    return { category: "Being Mind", color: "#8b5cf6", hexColor: 0x8b5cf6, particleColor: "#a78bfa", size: 8, isMajor: true };
  }
  if (t.includes("learnedfact") || t.includes("fact")) {
    return { category: "Learned Facts", color: "#6366f1", hexColor: 0x6366f1, particleColor: "#818cf8", size: 4.8, isMajor: false };
  }
  if (t.includes("workspace") || t.includes("workstationworkspace") || t.includes("repository")) {
    return { category: "Workspaces", color: "#0ea5e9", hexColor: 0x0ea5e9, particleColor: "#38bdf8", size: 9, isMajor: true };
  }
  if (t.includes("desktop") || t.includes("desktopplane") || t.includes("display")) {
    return { category: "Desktop Plane", color: "#06b6d4", hexColor: 0x06b6d4, particleColor: "#22d3ee", size: 7.5, isMajor: true };
  }
  if (t.includes("operator") || t.includes("terminal") || t.includes("operatorplane")) {
    return { category: "Operator Plane", color: "#14b8a6", hexColor: 0x14b8a6, particleColor: "#2dd4bf", size: 7.5, isMajor: true };
  }
  if (t.includes("craft") || t.includes("toolsmith") || t.includes("script")) {
    return { category: "Toolsmith & Craft", color: "#f59e0b", hexColor: 0xf59e0b, particleColor: "#fbbf24", size: 6, isMajor: false };
  }
  if (t.includes("service") || t.includes("runtimeservice") || t.includes("port")) {
    return { category: "Runtime Services", color: "#3b82f6", hexColor: 0x3b82f6, particleColor: "#60a5fa", size: 6, isMajor: false };
  }
  if (t.includes("evidence") || t.includes("receipt") || t.includes("verified")) {
    return { category: "Verified Evidence", color: "#10b981", hexColor: 0x10b981, particleColor: "#34d399", size: 6.5, isMajor: false };
  }
  if (t.includes("safety") || t.includes("policy") || t.includes("guard")) {
    return { category: "Safety Envelope", color: "#f43f5e", hexColor: 0xf43f5e, particleColor: "#fb7185", size: 8.5, isMajor: true };
  }
  return { category: "System Entity", color: "#94a3b8", hexColor: 0x94a3b8, particleColor: "#cbd5e1", size: 5, isMajor: false };
}

function createTextSprite(text: string, color: string = "#ffffff", fontSize: number = 22): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.Sprite();

  canvas.width = 512;
  canvas.height = 128;
  ctx.fillStyle = "rgba(0,0,0,0)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.font = `600 ${fontSize}px "JetBrains Mono", Consolas, monospace, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const metrics = ctx.measureText(text);
  const textWidth = Math.min(metrics.width + 36, 480);
  const x = canvas.width / 2 - textWidth / 2;
  const y = 36;
  const h = fontSize + 18;
  const r = 8;

  // Background pill with glow
  ctx.fillStyle = "rgba(7, 10, 20, 0.88)";
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

interface InternalNode extends GraphNodeData {
  meshGroup: THREE.Group;
  sphereMesh: THREE.Mesh;
  haloMesh?: THREE.Mesh;
  sprite?: THREE.Sprite;
  baseSize: number;
  baseColor: string;
  baseHex: number;
}

interface InternalEdge {
  sourceId: string;
  targetId: string;
  sourceNode?: InternalNode;
  targetNode?: InternalNode;
  lineMesh: THREE.Line;
  particleMesh: THREE.Mesh;
  particleT: number;
  particleSpeed: number;
  type: string;
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
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const controlsRef = useRef<any>(null);
    const animationFrameRef = useRef<number | null>(null);

    const internalNodesRef = useRef<Map<string, InternalNode>>(new Map());
    const internalEdgesRef = useRef<InternalEdge[]>([]);

    const isPausedRef = useRef<boolean>(isPaused);
    const autoRotateRef = useRef<boolean>(autoRotate);
    const selectedNodeRef = useRef<GraphNodeData | null>(selectedNode);
    const cameraTargetRef = useRef<THREE.Vector3 | null>(null);
    const cameraLookAtRef = useRef<THREE.Vector3 | null>(null);

    isPausedRef.current = isPaused;
    autoRotateRef.current = autoRotate;
    selectedNodeRef.current = selectedNode;

    // Filter active nodes
    const query = searchQuery.trim().toLowerCase();
    const visibleNodeIds = new Set<string>();

    nodes.forEach((node) => {
      const spec = getNodeVisualSpec(node.type);
      const categoryMatch =
        filterCategory === "ALL" ||
        spec.category.toLowerCase() === filterCategory.toLowerCase() ||
        (filterCategory === "COGNITIVE" &&
          (spec.category.includes("Cognitive") ||
            spec.category.includes("Mind") ||
            spec.category.includes("Facts"))) ||
        (filterCategory === "ARCHITECTURE" &&
          (spec.category.includes("Workspace") ||
            spec.category.includes("Plane") ||
            spec.category.includes("Service"))) ||
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

    // Expose camera controls to parent
    useImperativeHandle(ref, () => ({
      zoomIn: () => {
        if (!cameraRef.current) return;
        cameraRef.current.position.multiplyScalar(0.8);
      },
      zoomOut: () => {
        if (!cameraRef.current) return;
        cameraRef.current.position.multiplyScalar(1.25);
      },
      resetCamera: () => {
        if (!cameraRef.current || !controlsRef.current) return;
        cameraTargetRef.current = new THREE.Vector3(0, 30, 260);
        cameraLookAtRef.current = new THREE.Vector3(0, 0, 0);
      },
      focusNode: (nodeId: string) => {
        const target = internalNodesRef.current.get(nodeId);
        if (target && target.meshGroup) {
          const pos = target.meshGroup.position;
          const dist = 75;
          const dir = pos.clone().normalize();
          if (dir.lengthSq() === 0) dir.set(0, 0, 1);
          cameraTargetRef.current = pos.clone().add(dir.multiplyScalar(dist)).add(new THREE.Vector3(0, 12, 0));
          cameraLookAtRef.current = pos.clone();
        }
      },
      toggleAutoRotate: (enabled: boolean) => {
        autoRotateRef.current = enabled;
        if (controlsRef.current) {
          controlsRef.current.autoRotate = enabled;
        }
      },
      togglePause: (paused: boolean) => {
        isPausedRef.current = paused;
      },
    }));

    // Setup Three.js scene, camera, renderer, controls
    useEffect(() => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      container.innerHTML = "";

      const width = container.clientWidth || window.innerWidth;
      const height = container.clientHeight || 750;

      // 1. Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x05070e);
      scene.fog = new THREE.FogExp2(0x05070e, 0.0015);
      sceneRef.current = scene;

      // 2. Camera
      const camera = new THREE.PerspectiveCamera(55, width / height, 1, 3000);
      camera.position.set(0, 35, 260);
      cameraRef.current = camera;

      // 3. Renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // 4. OrbitControls (loaded dynamically on client side)
      import("three/examples/jsm/controls/OrbitControls.js").then(({ OrbitControls }) => {
        if (!containerRef.current) return;
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.maxDistance = 800;
        controls.minDistance = 30;
        controls.autoRotate = autoRotateRef.current;
        controls.autoRotateSpeed = 0.8;
        controlsRef.current = controls;
      });

      // 5. Lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
      scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.8);
      dirLight.position.set(100, 200, 150);
      scene.add(dirLight);

      const pointLight = new THREE.PointLight(0xa855f7, 2.2, 600);
      pointLight.position.set(0, 0, 0);
      scene.add(pointLight);

      // 6. Raycasting for mouse click & hover
      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2();

      let hoveredMesh: THREE.Mesh | null = null;

      const onPointerMove = (e: MouseEvent) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const meshes: THREE.Mesh[] = [];
        internalNodesRef.current.forEach((n) => {
          if (n.sphereMesh.visible) meshes.push(n.sphereMesh);
        });

        const intersects = raycaster.intersectObjects(meshes);
        if (intersects.length > 0) {
          renderer.domElement.style.cursor = "pointer";
          const hit = intersects[0].object as THREE.Mesh;
          if (hoveredMesh !== hit) {
            hoveredMesh = hit;
          }
        } else {
          renderer.domElement.style.cursor = "grab";
          hoveredMesh = null;
        }
      };

      const onPointerDown = () => {
        renderer.domElement.style.cursor = "grabbing";
      };

      const onClick = (e: MouseEvent) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const meshes: THREE.Mesh[] = [];
        internalNodesRef.current.forEach((n) => {
          if (n.sphereMesh.visible) meshes.push(n.sphereMesh);
        });

        const intersects = raycaster.intersectObjects(meshes);
        if (intersects.length > 0) {
          const hit = intersects[0].object as THREE.Mesh;
          const nodeId = hit.userData?.nodeId;
          const node = internalNodesRef.current.get(nodeId);
          if (node) {
            onSelectNode(node);
            const pos = node.meshGroup.position;
            const dist = 75;
            const dir = pos.clone().normalize();
            if (dir.lengthSq() === 0) dir.set(0, 0, 1);
            cameraTargetRef.current = pos.clone().add(dir.multiplyScalar(dist)).add(new THREE.Vector3(0, 12, 0));
            cameraLookAtRef.current = pos.clone();
          }
        }
      };

      renderer.domElement.addEventListener("mousemove", onPointerMove);
      renderer.domElement.addEventListener("mousedown", onPointerDown);
      renderer.domElement.addEventListener("click", onClick);

      // 7. Resize handler
      const handleResize = () => {
        if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
        const w = containerRef.current.clientWidth;
        const h = containerRef.current.clientHeight;
        cameraRef.current.aspect = w / h;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(w, h);
      };
      window.addEventListener("resize", handleResize);

      // 8. Animation & Physics Simulation Loop
      const animate = () => {
        animationFrameRef.current = requestAnimationFrame(animate);

        // Smooth camera transition if target is set
        if (cameraTargetRef.current && cameraLookAtRef.current && cameraRef.current && controlsRef.current) {
          cameraRef.current.position.lerp(cameraTargetRef.current, 0.06);
          controlsRef.current.target.lerp(cameraLookAtRef.current, 0.06);
          if (cameraRef.current.position.distanceTo(cameraTargetRef.current) < 1.0) {
            cameraTargetRef.current = null;
            cameraLookAtRef.current = null;
          }
        }

        if (controlsRef.current) {
          controlsRef.current.update();
        }

        // Run Force-Directed Physics if not paused
        if (!isPausedRef.current) {
          const nodeList = Array.from(internalNodesRef.current.values()).filter((n) => n.meshGroup.visible);
          const nCount = nodeList.length;

          // Coulomb Repulsion between all nodes
          for (let i = 0; i < nCount; i++) {
            const na = nodeList[i];
            for (let j = i + 1; j < nCount; j++) {
              const nb = nodeList[j];
              const dx = na.x! - nb.x!;
              const dy = na.y! - nb.y!;
              const dz = na.z! - nb.z!;
              const distSq = dx * dx + dy * dy + dz * dz + 120;
              const force = 180 / distSq;
              const dist = Math.sqrt(distSq);

              const fx = (dx / dist) * force;
              const fy = (dy / dist) * force;
              const fz = (dz / dist) * force;

              na.vx! += fx;
              na.vy! += fy;
              na.vz! += fz;

              nb.vx! -= fx;
              nb.vy! -= fy;
              nb.vz! -= fz;
            }
          }

          // Hooke's Attraction along active edges
          internalEdgesRef.current.forEach((edge) => {
            const na = edge.sourceNode;
            const nb = edge.targetNode;
            if (na && nb && na.meshGroup.visible && nb.meshGroup.visible) {
              const dx = nb.x! - na.x!;
              const dy = nb.y! - na.y!;
              const dz = nb.z! - na.z!;
              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
              const targetDist = 65;
              const force = (dist - targetDist) * 0.025;

              const fx = (dx / dist) * force;
              const fy = (dy / dist) * force;
              const fz = (dz / dist) * force;

              na.vx! += fx;
              na.vy! += fy;
              na.vz! += fz;

              nb.vx! -= fx;
              nb.vy! -= fy;
              nb.vz! -= fz;
            }
          });

          // Center Gravity and Velocity Damping
          nodeList.forEach((n) => {
            n.vx! -= n.x! * 0.005;
            n.vy! -= n.y! * 0.005;
            n.vz! -= n.z! * 0.005;

            n.vx! *= 0.88;
            n.vy! *= 0.88;
            n.vz! *= 0.88;

            n.x! += n.vx!;
            n.y! += n.vy!;
            n.z! += n.vz!;

            n.meshGroup.position.set(n.x!, n.y!, n.z!);

            // Rotate halo ring slowly
            if (n.haloMesh) {
              n.haloMesh.rotation.z += 0.01;
            }
          });

          // Update edge lines and particle flow
          internalEdgesRef.current.forEach((edge) => {
            const na = edge.sourceNode;
            const nb = edge.targetNode;
            if (na && nb && na.meshGroup.visible && nb.meshGroup.visible) {
              edge.lineMesh.visible = true;
              edge.particleMesh.visible = true;

              const positions = (edge.lineMesh.geometry as THREE.BufferGeometry).attributes.position;
              positions.setXYZ(0, na.x!, na.y!, na.z!);
              positions.setXYZ(1, nb.x!, nb.y!, nb.z!);
              positions.needsUpdate = true;

              // Animate streaming particle along edge
              edge.particleT = (edge.particleT + edge.particleSpeed) % 1.0;
              const px = THREE.MathUtils.lerp(na.x!, nb.x!, edge.particleT);
              const py = THREE.MathUtils.lerp(na.y!, nb.y!, edge.particleT);
              const pz = THREE.MathUtils.lerp(na.z!, nb.z!, edge.particleT);
              edge.particleMesh.position.set(px, py, pz);
            } else {
              edge.lineMesh.visible = false;
              edge.particleMesh.visible = false;
            }
          });
        }

        renderer.render(scene, camera);
      };

      animate();

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        window.removeEventListener("resize", handleResize);
        renderer.domElement.removeEventListener("mousemove", onPointerMove);
        renderer.domElement.removeEventListener("mousedown", onPointerDown);
        renderer.domElement.removeEventListener("click", onClick);
        if (controlsRef.current) {
          controlsRef.current.dispose();
        }
        renderer.dispose();
      };
    }, []);

    // Build or update 3D node and edge meshes when props change
    useEffect(() => {
      const scene = sceneRef.current;
      if (!scene) return;

      // 1. Clear existing node and edge meshes
      internalNodesRef.current.forEach((n) => {
        scene.remove(n.meshGroup);
      });
      internalEdgesRef.current.forEach((e) => {
        scene.remove(e.lineMesh);
        scene.remove(e.particleMesh);
      });

      internalNodesRef.current.clear();
      internalEdgesRef.current = [];

      // 2. Build nodes in 3D
      nodes.forEach((node, index) => {
        const spec = getNodeVisualSpec(node.type);
        const group = new THREE.Group();

        // Initial sphere layout in 3D
        const phi = Math.acos(-1 + (2 * index) / Math.max(nodes.length, 1));
        const theta = Math.sqrt(nodes.length * Math.PI) * phi;
        const radiusLayout = 90;

        const posX = node.x ?? radiusLayout * Math.cos(theta) * Math.sin(phi);
        const posY = node.y ?? radiusLayout * Math.sin(theta) * Math.sin(phi);
        const posZ = node.z ?? radiusLayout * Math.cos(phi);

        group.position.set(posX, posY, posZ);

        // Core Sphere Mesh
        const sphereGeo = new THREE.SphereGeometry(spec.size, 24, 24);
        const sphereMat = new THREE.MeshStandardMaterial({
          color: spec.hexColor,
          roughness: 0.25,
          metalness: 0.8,
          emissive: spec.hexColor,
          emissiveIntensity: 0.45,
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.userData = { nodeId: node.id };
        group.add(sphere);

        // Glowing outer halo for key nodes
        let haloMesh: THREE.Mesh | undefined;
        if (spec.isMajor) {
          const ringGeo = new THREE.RingGeometry(spec.size * 1.35, spec.size * 1.6, 32);
          const ringMat = new THREE.MeshBasicMaterial({
            color: spec.hexColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.35,
          });
          haloMesh = new THREE.Mesh(ringGeo, ringMat);
          haloMesh.rotation.x = Math.PI / 2;
          group.add(haloMesh);
        }

        // Floating Billboard Sprite
        const labelText = node.label || node.id;
        const displayLabel = labelText.length > 28 ? labelText.slice(0, 25) + "…" : labelText;
        const sprite = createTextSprite(displayLabel, spec.color, 22);
        sprite.position.y = spec.size + 6.5;
        group.add(sprite);

        // Visibility based on filters
        const isVisible = visibleNodeIds.has(node.id);
        group.visible = isVisible;

        scene.add(group);

        const internalNode: InternalNode = {
          ...node,
          x: posX,
          y: posY,
          z: posZ,
          vx: 0,
          vy: 0,
          vz: 0,
          meshGroup: group,
          sphereMesh: sphere,
          haloMesh,
          sprite,
          baseSize: spec.size,
          baseColor: spec.color,
          baseHex: spec.hexColor,
        };

        internalNodesRef.current.set(node.id, internalNode);
      });

      // 3. Build edges & particles in 3D
      edges.forEach((edge) => {
        const sId = typeof edge.source === "object" ? edge.source.id : edge.source;
        const tId = typeof edge.target === "object" ? edge.target.id : edge.target;

        const sourceNode = internalNodesRef.current.get(sId);
        const targetNode = internalNodesRef.current.get(tId);

        if (!sourceNode || !targetNode) return;

        // Line mesh
        const geom = new THREE.BufferGeometry();
        geom.setAttribute(
          "position",
          new THREE.Float32BufferAttribute([sourceNode.x!, sourceNode.y!, sourceNode.z!, targetNode.x!, targetNode.y!, targetNode.z!], 3)
        );
        const mat = new THREE.LineBasicMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.35,
        });
        const line = new THREE.Line(geom, mat);
        scene.add(line);

        // Streaming particle
        const pGeo = new THREE.SphereGeometry(1.2, 12, 12);
        const pMat = new THREE.MeshBasicMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.85,
        });
        const particle = new THREE.Mesh(pGeo, pMat);
        scene.add(particle);

        const isVisible = visibleNodeIds.has(sId) && visibleNodeIds.has(tId);
        line.visible = isVisible;
        particle.visible = isVisible;

        internalEdgesRef.current.push({
          sourceId: sId,
          targetId: tId,
          sourceNode,
          targetNode,
          lineMesh: line,
          particleMesh: particle,
          particleT: Math.random(),
          particleSpeed: 0.005 + Math.random() * 0.004,
          type: edge.type,
        });
      });
    }, [nodes, edges, searchQuery, filterCategory]);

    // Update node styling on selection or highlight
    useEffect(() => {
      internalNodesRef.current.forEach((n) => {
        const isSelected = selectedNode?.id === n.id;
        const isMatch = query ? n.label.toLowerCase().includes(query) : false;

        const mat = n.sphereMesh.material as THREE.MeshStandardMaterial;
        if (isSelected) {
          mat.color.setHex(0xffffff);
          mat.emissive.setHex(0x38bdf8);
          mat.emissiveIntensity = 0.95;
          n.meshGroup.scale.set(1.35, 1.35, 1.35);
        } else if (isMatch) {
          mat.color.setHex(n.baseHex);
          mat.emissive.setHex(n.baseHex);
          mat.emissiveIntensity = 0.85;
          n.meshGroup.scale.set(1.15, 1.15, 1.15);
        } else {
          mat.color.setHex(n.baseHex);
          mat.emissive.setHex(n.baseHex);
          mat.emissiveIntensity = 0.45;
          n.meshGroup.scale.set(1.0, 1.0, 1.0);
        }
      });

      // Highlight links connected to selected node
      internalEdgesRef.current.forEach((edge) => {
        const isConnected = selectedNode && (edge.sourceId === selectedNode.id || edge.targetId === selectedNode.id);
        const lineMat = edge.lineMesh.material as THREE.LineBasicMaterial;
        const pMat = edge.particleMesh.material as THREE.MeshBasicMaterial;

        if (isConnected) {
          lineMat.color.setHex(0x38bdf8);
          lineMat.opacity = 0.9;
          pMat.color.setHex(0x38bdf8);
          pMat.opacity = 1.0;
        } else {
          lineMat.color.setHex(0x6366f1);
          lineMat.opacity = 0.3;
          pMat.color.setHex(0x818cf8);
          pMat.opacity = 0.6;
        }
      });
    }, [selectedNode, searchQuery]);

    return (
      <div className="relative w-full h-full min-h-[680px] rounded-2xl overflow-hidden border border-ink-800/80 bg-[#05070e] shadow-2xl">
        <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      </div>
    );
  }
);

Cognitive3DGraph.displayName = "Cognitive3DGraph";
