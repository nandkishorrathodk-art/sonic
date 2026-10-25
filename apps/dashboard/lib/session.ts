// Active workstation session shared across dashboard routes.
//
// The Workstation view mints/attaches a session (e.g. `ws-<hex>`) while the
// Agents registry previously always queried the literal `default` session, so
// the Boss workers the operator had just launched were invisible there. This
// keeps one active session id that both views read and write.

const STORAGE_KEY = "sonic.active_session_id";
const SESSION_EVENT = "sonic:session-changed";

export function getActiveSessionId(): string {
  if (typeof window === "undefined") return "default";
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored && stored.trim() ? stored.trim() : "default";
  } catch {
    return "default";
  }
}

export function setActiveSessionId(sessionId: string): void {
  if (typeof window === "undefined") return;
  const value = (sessionId || "").trim();
  if (!value) return;
  try {
    if (window.localStorage.getItem(STORAGE_KEY) === value) return;
    window.localStorage.setItem(STORAGE_KEY, value);
  } catch {
    // Storage may be unavailable (private mode); in-memory listeners still fire.
  }
  window.dispatchEvent(new CustomEvent(SESSION_EVENT, { detail: value }));
}

// Subscribe to active-session changes, including cross-tab updates.
export function subscribeToActiveSession(
  listener: (sessionId: string) => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const onCustom = (event: Event) => {
    const detail = (event as CustomEvent<string>).detail;
    listener(detail || getActiveSessionId());
  };
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) {
      listener(event.newValue || "default");
    }
  };
  window.addEventListener(SESSION_EVENT, onCustom);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(SESSION_EVENT, onCustom);
    window.removeEventListener("storage", onStorage);
  };
}
