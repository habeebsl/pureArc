import React, {
  createContext,
  useContext,
  useState,
  useRef,
  useCallback,
  useEffect,
} from "react";
import {
  Shot,
  SessionStartResponse,
  getSessionShots,
  getShotDetail,
  generateMockShot,
  USE_MOCK,
} from "../../api";

interface SessionState {
  session: SessionStartResponse | null;
  shots: Shot[];
  isPolling: boolean;
  latestShot: Shot | null;
  newShotFlash: boolean; // triggers brief visual flash
}

interface SessionContextValue extends SessionState {
  setSession: (s: SessionStartResponse) => void;
  addShot: (shot: Shot) => void;
  startPolling: () => void;
  stopPolling: () => void;
  clearSession: () => void;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<SessionState>({
    session: null,
    shots: [],
    isPolling: false,
    latestShot: null,
    newShotFlash: false,
  });

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const demoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const knownShotIds = useRef<Set<string>>(new Set());
  const flashTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerFlash = useCallback(() => {
    setState((prev) => ({ ...prev, newShotFlash: true }));
    if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = setTimeout(() => {
      setState((prev) => ({ ...prev, newShotFlash: false }));
    }, 1200);
  }, []);

  const addShot = useCallback(
    (shot: Shot) => {
      if (knownShotIds.current.has(shot.id)) return;
      knownShotIds.current.add(shot.id);
      setState((prev) => ({
        ...prev,
        shots: [shot, ...prev.shots],
        latestShot: shot,
      }));
      triggerFlash();
    },
    [triggerFlash]
  );

  // Demo mode: auto-generate shots on a randomized interval
  const scheduleDemoShot = useCallback(
    (sessionId: string) => {
      const delay = 7000 + Math.random() * 6000; // 7–13 s
      demoTimerRef.current = setTimeout(() => {
        const shot = generateMockShot(sessionId);
        addShot(shot);
        scheduleDemoShot(sessionId); // schedule next
      }, delay);
    },
    [addShot]
  );

  const startPolling = useCallback(() => {
    if (!state.session) return;
    setState((prev) => ({ ...prev, isPolling: true }));

    if (USE_MOCK) {
      scheduleDemoShot(state.session.session_id);
    } else {
      const buildWsUrl = (session: SessionStartResponse): string => {
        const raw = session.ws_url || `/session/${session.session_id}/events`;

        if (/^wss?:\/\//i.test(raw)) {
          return raw;
        }

        // App runs through Vite HTTPS URL in dev; use same-origin WS via /api proxy.
        const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const normalizedPath = raw.startsWith("/") ? raw : `/${raw}`;
        return `${wsProto}//${window.location.host}/api${normalizedPath}`;
      };

      const connectWs = () => {
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }

        try {
          const ws = new WebSocket(buildWsUrl(state.session!));
          wsRef.current = ws;

          ws.onmessage = async (evt) => {
            try {
              const data = JSON.parse(evt.data);
              if (data?.event !== "shot_result") return;
              const shotId = data?.payload?.shot_id;
              if (!shotId || knownShotIds.current.has(shotId)) return;
              const shot = await getShotDetail(shotId);
              addShot(shot);
            } catch {
              // ignore malformed or transient websocket event errors
            }
          };

          ws.onerror = () => {
            // keep polling fallback active
          };
        } catch {
          // keep polling fallback active
        }
      };

      connectWs();

      // Polling fallback and reconciliation (handles missed ws events)
      pollingRef.current = setInterval(async () => {
        try {
          const shots = await getSessionShots(state.session!.session_id);
          shots.forEach((shot) => addShot(shot));
        } catch {
          // silently ignore transient errors
        }
      }, 5000);
    }
  }, [state.session, addShot, scheduleDemoShot]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    if (demoTimerRef.current) clearTimeout(demoTimerRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState((prev) => ({ ...prev, isPolling: false }));
  }, []);

  const setSession = useCallback((s: SessionStartResponse) => {
    knownShotIds.current = new Set();
    setState({
      session: s,
      shots: [],
      isPolling: false,
      latestShot: null,
      newShotFlash: false,
    });
  }, []);

  const clearSession = useCallback(() => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    if (demoTimerRef.current) clearTimeout(demoTimerRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    knownShotIds.current = new Set();
    setState({
      session: null,
      shots: [],
      isPolling: false,
      latestShot: null,
      newShotFlash: false,
    });
  }, []);

  // Start polling automatically once session is set
  useEffect(() => {
    if (state.session && !state.isPolling) {
      startPolling();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.session]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (demoTimerRef.current) clearTimeout(demoTimerRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  return (
    <SessionContext.Provider
      value={{
        ...state,
        setSession,
        addShot,
        startPolling,
        stopPolling,
        clearSession,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used inside SessionProvider");
  return ctx;
}
