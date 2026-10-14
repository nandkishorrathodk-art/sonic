"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export function HeroScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [webglUnavailable, setWebglUnavailable] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
    } catch (error) {
      console.warn("SONIC hero WebGL scene unavailable; using the CSS fallback.", error);
      setWebglUnavailable(true);
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(0, 0, 10);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.setAttribute("aria-hidden", "true");
    container.appendChild(renderer.domElement);

    const root = new THREE.Group();
    scene.add(root);

    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.18, 5),
      new THREE.MeshStandardMaterial({
        color: 0x66f0e2,
        emissive: 0x087f83,
        emissiveIntensity: 1.35,
        metalness: 0.68,
        roughness: 0.22,
      }),
    );
    root.add(core);

    const wireframe = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.23, 2),
      new THREE.MeshBasicMaterial({ color: 0xa6fff4, wireframe: true, transparent: true, opacity: 0.19 }),
    );
    root.add(wireframe);

    const rings = [0, 1, 2].map((index) => {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(1.82 + index * 0.29, index === 1 ? 0.009 : 0.014, 8, 180),
        new THREE.MeshBasicMaterial({
          color: 0x64eadd,
          transparent: true,
          opacity: 0.72,
          side: THREE.DoubleSide,
        }),
      );
      ring.rotation.set(0.52 + index * 0.62, 0.28 + index * 0.83, index * 0.55);
      root.add(ring);
      return ring;
    });

    const satelliteMaterial = new THREE.MeshStandardMaterial({
      color: 0xff9e86,
      emissive: 0x9d3f3a,
      emissiveIntensity: 1.6,
      metalness: 0.55,
      roughness: 0.3,
    });
    const satellites = [
      new THREE.Mesh(new THREE.SphereGeometry(0.095, 20, 20), satelliteMaterial),
      new THREE.Mesh(new THREE.SphereGeometry(0.065, 16, 16), satelliteMaterial),
      new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 16), satelliteMaterial),
    ];
    satellites[0].position.set(2.08, 0, 0);
    satellites[1].position.set(-1.04, 1.1, 1.8);
    satellites[2].position.set(-1.1, -1.1, -0.8);
    satellites.forEach((satellite) => root.add(satellite));

    const starPositions = new Float32Array(420 * 3);
    for (let index = 0; index < starPositions.length; index += 3) {
      starPositions[index] = (Math.random() - 0.5) * 11;
      starPositions[index + 1] = (Math.random() - 0.5) * 8;
      starPositions[index + 2] = (Math.random() - 0.5) * 5 - 1;
    }
    const starsGeometry = new THREE.BufferGeometry();
    starsGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
    const stars = new THREE.Points(
      starsGeometry,
      new THREE.PointsMaterial({ color: 0x83d9d2, size: 0.018, transparent: true, opacity: 0.52 }),
    );
    scene.add(stars);

    scene.add(new THREE.AmbientLight(0x9feee4, 1.2));
    const keyLight = new THREE.PointLight(0x63eee1, 32, 12);
    keyLight.position.set(2, 2, 4);
    scene.add(keyLight);
    const rimLight = new THREE.PointLight(0x8972ff, 20, 12);
    rimLight.position.set(-3, -2, -2);
    scene.add(rimLight);

    let isVisible = true;
    let frameId = 0;
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    const resizeObserver = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (motionQuery.matches) renderer.render(scene, camera);
    });
    resizeObserver.observe(container);

    const visibilityObserver = new IntersectionObserver(([entry]) => {
      isVisible = entry.isIntersecting;
      if (!isVisible && frameId) {
        cancelAnimationFrame(frameId);
        frameId = 0;
      } else if (isVisible && !motionQuery.matches && !document.hidden && !frameId) {
        frameId = requestAnimationFrame(animate);
      } else if (isVisible && motionQuery.matches) {
        renderer.render(scene, camera);
      }
    });
    visibilityObserver.observe(container);

    const renderFrame = (time: number) => {
      if (!isVisible || document.hidden) return;

      const seconds = time * 0.001;
      if (!motionQuery.matches) {
        root.rotation.y = seconds * 0.12;
        root.rotation.x = Math.sin(seconds * 0.17) * 0.07;
        wireframe.rotation.copy(root.rotation);
        rings.forEach((ring, index) => {
          ring.rotation.z += (index % 2 === 0 ? 1 : -1) * 0.0008;
        });
        satellites[0].position.set(Math.cos(seconds * 0.62) * 2.08, Math.sin(seconds * 0.62) * 0.55, Math.sin(seconds * 0.62) * 1.45);
        satellites[1].position.set(Math.cos(seconds * -0.43 + 2) * 2.5, Math.sin(seconds * -0.43 + 2) * 1.1, Math.cos(seconds * -0.43 + 2) * 0.7);
        satellites[2].position.set(Math.cos(seconds * 0.32 + 4) * 1.68, Math.sin(seconds * 0.32 + 4) * 1.35, Math.sin(seconds * 0.32 + 4) * 0.8);
        stars.rotation.y = seconds * 0.008;
      }
      renderer.render(scene, camera);
    };
    const animate = (time: number) => {
      renderFrame(time);
      frameId = !motionQuery.matches && isVisible && !document.hidden
        ? requestAnimationFrame(animate)
        : 0;
    };
    const onMotionPreferenceChange = () => {
      if (motionQuery.matches) {
        if (frameId) cancelAnimationFrame(frameId);
        frameId = 0;
        renderFrame(0);
      } else if (isVisible && !document.hidden && !frameId) {
        frameId = requestAnimationFrame(animate);
      }
    };
    const onDocumentVisibilityChange = () => {
      if (document.hidden) {
        if (frameId) cancelAnimationFrame(frameId);
        frameId = 0;
      } else if (isVisible && !motionQuery.matches && !frameId) {
        frameId = requestAnimationFrame(animate);
      } else if (isVisible && motionQuery.matches) {
        renderFrame(0);
      }
    };
    motionQuery.addEventListener("change", onMotionPreferenceChange);
    document.addEventListener("visibilitychange", onDocumentVisibilityChange);
    if (motionQuery.matches) renderFrame(0);
    else frameId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      visibilityObserver.disconnect();
      motionQuery.removeEventListener("change", onMotionPreferenceChange);
      document.removeEventListener("visibilitychange", onDocumentVisibilityChange);
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className="sonic-hero-scene" ref={containerRef} aria-hidden="true">
      {webglUnavailable && <div className="sonic-hero-scene-fallback" />}
    </div>
  );
}
