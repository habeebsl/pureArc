import { checkHealth, getLatestSession, startSession, USE_MOCK } from "../../api";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import {
  AlertTriangle,
  ChevronRight,
  RefreshCw,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";
import { useSession } from "../context/SessionContext";

type HealthState = "checking" | "online" | "offline";

export function LaunchScreen() {
  const navigate = useNavigate();
  const { setSession } = useSession();
  const [health, setHealth] = useState<HealthState>("checking");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runHealthCheck = async () => {
    setHealth("checking");
    setError(null);
    try {
      const res = await checkHealth();
      setHealth(res.status === "offline" ? "offline" : "online");
    } catch {
      setHealth("offline");
    }
  };

  useEffect(() => {
    runHealthCheck();
  }, []);

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const session = await startSession({
        user_id: "demo-user",
        device: navigator.userAgent.includes("Mobile") ? "mobile" : "desktop",
        fps: 30,
        resolution: [1280, 720],
      });

      let chosenSession = session;

      if (!USE_MOCK) {
        try {
          const latest = await getLatestSession();
          if (latest?.session_id && latest.session_id !== session.session_id) {
            chosenSession = {
              session_id: latest.session_id,
              ws_url: `/session/${latest.session_id}/events`,
            };
          }
        } catch {
          // keep started session if latest lookup fails
        }
      }

      setSession(chosenSession);
      navigate("/session");
    } catch (e: any) {
      setError("Failed to start session. Check connection and retry.");
      setStarting(false);
    }
  };

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden"
      style={{ background: "#0A0A0A", fontFamily: "'Inter', sans-serif" }}
    >
      {/* Subtle court-line background */}
      <div className="absolute inset-0 pointer-events-none select-none overflow-hidden opacity-[0.03]">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50%" cy="50%" r="180" fill="none" stroke="#E8602C" strokeWidth="2" />
          <circle cx="50%" cy="50%" r="60" fill="none" stroke="#E8602C" strokeWidth="2" />
          <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#E8602C" strokeWidth="1.5" />
          <rect x="35%" y="20%" width="30%" height="30%" fill="none" stroke="#E8602C" strokeWidth="1.5" />
          <rect x="35%" y="50%" width="30%" height="30%" fill="none" stroke="#E8602C" strokeWidth="1.5" />
        </svg>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-md px-6 flex flex-col items-center gap-8"
      >
        {/* Logo mark */}
        <div className="flex flex-col items-center gap-3">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center"
            style={{ background: "#E8602C" }}
          >
            <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
              <circle cx="17" cy="17" r="13" stroke="white" strokeWidth="2" />
              <path d="M4 17 Q17 4 30 17" stroke="white" strokeWidth="2" fill="none" />
              <path d="M4 17 Q17 30 30 17" stroke="white" strokeWidth="2" fill="none" />
              <line x1="17" y1="4" x2="17" y2="30" stroke="white" strokeWidth="2" />
            </svg>
          </div>
          <div className="text-center">
            <p
              className="tracking-[0.2em] uppercase text-xs mb-1"
              style={{ color: "#E8602C", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600 }}
            >
              Shot Coaching
            </p>
            <h1
              className="tracking-tight"
              style={{
                fontFamily: "'Barlow Condensed', sans-serif",
                fontWeight: 700,
                fontSize: "2.8rem",
                color: "#F2EDE8",
                lineHeight: 1,
              }}
            >
              PUREARC
            </h1>
          </div>
        </div>

        {/* Status card */}
        <div
          className="w-full rounded-2xl p-5 flex flex-col gap-4"
          style={{ background: "#141414", border: "1px solid #252525" }}
        >
          <AnimatePresence mode="wait">
            {health === "checking" && (
              <motion.div
                key="checking"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-3"
              >
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-8 h-8 rounded-full border-2 border-t-transparent"
                  style={{ borderColor: "#2C2C2C", borderTopColor: "#E8602C" }}
                />
                <div>
                  <p className="text-sm" style={{ color: "#F2EDE8" }}>
                    Checking service…
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "#6E6862" }}>
                    Connecting to PureArc backend
                  </p>
                </div>
              </motion.div>
            )}

            {health === "online" && (
              <motion.div
                key="online"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-3"
              >
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ background: "#1A2E23" }}
                >
                  <Wifi className="w-4 h-4" style={{ color: "#3DB87A" }} />
                </div>
                <div>
                  <p className="text-sm" style={{ color: "#F2EDE8" }}>
                    Ready to Start
                    {USE_MOCK && (
                      <span
                        className="ml-2 text-xs px-1.5 py-0.5 rounded"
                        style={{ background: "#2A1F0F", color: "#E8602C" }}
                      >
                        DEMO
                      </span>
                    )}
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "#3DB87A" }}>
                    {USE_MOCK ? "Demo mode — shots auto-generated" : "Backend online · All systems operational"}
                  </p>
                </div>
              </motion.div>
            )}

            {health === "offline" && (
              <motion.div
                key="offline"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col gap-3"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ background: "#2A1818" }}
                  >
                    <WifiOff className="w-4 h-4" style={{ color: "#D95C5C" }} />
                  </div>
                  <div>
                    <p className="text-sm" style={{ color: "#F2EDE8" }}>
                      Service Offline
                    </p>
                    <p className="text-xs mt-0.5" style={{ color: "#D95C5C" }}>
                      Cannot reach PureArc backend
                    </p>
                  </div>
                </div>
                <div
                  className="rounded-xl p-3 flex gap-2"
                  style={{ background: "#1A1410", border: "1px solid #2E2218" }}
                >
                  <AlertTriangle
                    className="w-4 h-4 flex-shrink-0 mt-0.5"
                    style={{ color: "#E8602C" }}
                  />
                  <div className="text-xs" style={{ color: "#8A847D" }}>
                    <p>Troubleshooting:</p>
                    <ul className="mt-1 space-y-0.5 list-disc list-inside">
                      <li>Confirm backend is running on the correct port</li>
                      <li>Set <code className="px-1 rounded" style={{ background: "#252525" }}>VITE_API_BASE_URL</code> in your .env file</li>
                      <li>Check for CORS issues in browser console</li>
                    </ul>
                  </div>
                </div>
                <button
                  onClick={runHealthCheck}
                  className="flex items-center justify-center gap-2 rounded-xl py-2.5 text-sm transition-colors"
                  style={{ background: "#1C1C1C", color: "#F2EDE8", border: "1px solid #2C2C2C" }}
                >
                  <RefreshCw className="w-4 h-4" />
                  Retry Connection
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Error state */}
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm text-center px-4 py-3 rounded-xl w-full"
            style={{ background: "#2A1818", color: "#D95C5C", border: "1px solid #3D2020" }}
          >
            {error}
          </motion.p>
        )}

        {/* Start Session button */}
        <motion.button
          onClick={handleStart}
          disabled={health === "checking" || starting}
          whileHover={health !== "checking" && !starting ? { scale: 1.02 } : {}}
          whileTap={health !== "checking" && !starting ? { scale: 0.98 } : {}}
          className="w-full flex items-center justify-center gap-3 rounded-2xl py-4 transition-all"
          style={{
            background:
              health === "checking" || starting ? "#2A2A2A" : "#E8602C",
            color: health === "checking" || starting ? "#4A4A4A" : "#fff",
            fontFamily: "'Barlow Condensed', sans-serif",
            fontWeight: 700,
            fontSize: "1.15rem",
            letterSpacing: "0.08em",
            cursor:
              health === "checking" || starting ? "not-allowed" : "pointer",
          }}
        >
          {starting ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                className="w-5 h-5 rounded-full border-2 border-t-transparent"
                style={{ borderColor: "#6A6A6A", borderTopColor: "#AAA" }}
              />
              STARTING SESSION…
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              START SESSION
              <ChevronRight className="w-5 h-5" />
            </>
          )}
        </motion.button>

        <p className="text-xs text-center" style={{ color: "#4A4540" }}>
          No account required · Sessions are not stored
        </p>
      </motion.div>
    </div>
  );
}