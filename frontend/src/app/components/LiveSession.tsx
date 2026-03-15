import { Shot, USE_MOCK } from "../../api";
import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import {
  CameraOff,
  Video,
  Square,
  ChevronRight,
  TrendingUp,
  Clock,
  X,
} from "lucide-react";
import { useSession } from "../context/SessionContext";
import { MetricTile } from "./MetricTile";
import { ShotHistoryCard } from "./ShotHistoryCard";
import { ReplayPanel } from "./ReplayPanel";

function useSessionTimer() {
  const startRef = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(iv);
  }, []);
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function metricHighlight(
  key: "release_angle" | "arc" | "tempo" | "drift",
  value: number,
  result: "made" | "miss"
): "good" | "warn" | "neutral" {
  if (result === "made") return "good";
  if (key === "release_angle" && value < 48) return "warn";
  if (key === "arc" && value < 42) return "warn";
  if (key === "tempo" && (value < 0.70 || value > 1.1)) return "warn";
  if (key === "drift" && Math.abs(value) > 4) return "warn";
  return "neutral";
}

export function LiveSession() {
  const navigate = useNavigate();
  const { session, shots, latestShot, newShotFlash, stopPolling } = useSession();
  const [selectedShot, setSelectedShot] = useState<Shot | null>(null);
  const [showMobileResults, setShowMobileResults] = useState(false);
  const [cameraState, setCameraState] = useState<"idle" | "ready" | "error">("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);

  // Redirect if no session
  useEffect(() => {
    if (!session) navigate("/");
  }, [session, navigate]);

  const timer = useSessionTimer();
  const makes = shots.filter((s) => s.result === "made").length;
  const attempts = shots.length;
  const pct = attempts > 0 ? Math.round((makes / attempts) * 100) : 0;

  useEffect(() => {
    if (!session) return;

    let isCancelled = false;

    const startCamera = async () => {
      try {
        setCameraState("idle");
        setCameraError(null);

        let mediaStream: MediaStream | null = null;

        const baseVideo = {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        };

        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
              ...baseVideo,
              facingMode: { exact: "environment" },
            },
            audio: false,
          });
        } catch {
          try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
              video: {
                ...baseVideo,
                facingMode: { ideal: "environment" },
              },
              audio: false,
            });
          } catch {
            mediaStream = await navigator.mediaDevices.getUserMedia({
              video: {
                ...baseVideo,
                facingMode: "user",
              },
              audio: false,
            });
          }
        }

        if (isCancelled) {
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }

        cameraStreamRef.current = mediaStream;
        setCameraState("ready");
      } catch {
        if (isCancelled) return;
        setCameraState("error");
        setCameraError("Camera unavailable. Check browser permission and device access.");
      }
    };

    startCamera();

    return () => {
      isCancelled = true;
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((track) => track.stop());
        cameraStreamRef.current = null;
      }
    };
  }, [session]);

  useEffect(() => {
    const video = cameraVideoRef.current;
    const stream = cameraStreamRef.current;
    if (!video || !stream) return;

    video.srcObject = stream;
    video.play().catch(() => null);
  }, [cameraState]);

  const handleEnd = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    stopPolling();
    navigate("/summary");
  };

  if (!session) return null;

  return (
    <div
      className="h-[100dvh] overflow-hidden flex flex-col"
      style={{ background: "#0A0A0A", fontFamily: "'Inter', sans-serif" }}
    >
      {/* Shot Flash Overlay */}
      <AnimatePresence>
        {newShotFlash && latestShot && (
          <motion.div
            key="flash"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 pointer-events-none z-30"
            style={{
              background:
                latestShot.result === "made"
                  ? "rgba(61,184,122,0.07)"
                  : "rgba(217,92,92,0.07)",
            }}
          />
        )}
      </AnimatePresence>

      {/* Header */}
      <header
        className="flex items-center px-5 py-3 gap-4 flex-shrink-0"
        style={{ borderBottom: "1px solid #1A1A1A", background: "#0C0C0C" }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: "#E8602C" }}
          >
            <svg width="16" height="16" viewBox="0 0 34 34" fill="none">
              <circle cx="17" cy="17" r="13" stroke="white" strokeWidth="2.5" />
              <path d="M4 17 Q17 4 30 17" stroke="white" strokeWidth="2" fill="none" />
              <path d="M4 17 Q17 30 30 17" stroke="white" strokeWidth="2" fill="none" />
              <line x1="17" y1="4" x2="17" y2="30" stroke="white" strokeWidth="2" />
            </svg>
          </div>
          <span
            style={{
              fontFamily: "'Barlow Condensed', sans-serif",
              fontWeight: 700,
              fontSize: "1.15rem",
              color: "#F2EDE8",
              letterSpacing: "0.05em",
            }}
          >
            PUREARC
          </span>
        </div>

        {/* Live pulse + status */}
        <div className="hidden sm:flex items-center gap-2 ml-2">
          <span className="relative flex h-2 w-2">
            <span
              className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ background: "#3DB87A" }}
            />
            <span
              className="relative inline-flex rounded-full h-2 w-2"
              style={{ background: "#3DB87A" }}
            />
          </span>
          <span className="text-xs" style={{ color: "#3DB87A" }}>
            SESSION ACTIVE
          </span>
          {USE_MOCK && (
            <span
              className="text-xs px-1.5 py-0.5 rounded ml-1"
              style={{ background: "#1E1400", color: "#E8602C", border: "1px solid #2E1E00" }}
            >
              DEMO
            </span>
          )}
        </div>

        <div className="sm:hidden flex items-center gap-2 ml-2">
          <span className="relative flex h-2 w-2">
            <span
              className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ background: "#3DB87A" }}
            />
            <span
              className="relative inline-flex rounded-full h-2 w-2"
              style={{ background: "#3DB87A" }}
            />
          </span>
          <span className="text-xs" style={{ color: "#3DB87A" }}>
            LIVE
          </span>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Timer */}
        <div className="flex items-center gap-1.5" style={{ color: "#6E6862" }}>
          <Clock className="w-3.5 h-3.5" />
          <span
            style={{
              fontFamily: "'Barlow Condensed', sans-serif",
              fontWeight: 600,
              fontSize: "1rem",
              color: "#8A847D",
            }}
          >
            {timer}
          </span>
        </div>

        {/* End Session (desktop) */}
        <button
          onClick={handleEnd}
          className="hidden sm:flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm transition-colors"
          style={{
            background: "#1C1C1C",
            color: "#D95C5C",
            border: "1px solid #2A2020",
            fontFamily: "'Barlow Condensed', sans-serif",
            fontWeight: 600,
            letterSpacing: "0.04em",
          }}
        >
          <Square className="w-3.5 h-3.5" />
          END SESSION
        </button>
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* LEFT — Camera + Stats */}
        <div
          className="lg:w-[380px] xl:w-[420px] flex flex-col gap-4 p-5 flex-shrink-0"
          style={{ borderRight: "1px solid #1A1A1A" }}
        >
          {/* Camera preview */}
          <div
            className="relative rounded-2xl overflow-hidden aspect-[3/4] sm:aspect-video flex items-center justify-center"
            style={{ background: "#0F0F0F", border: "1px solid #1E1E1E" }}
          >
            <video
              ref={cameraVideoRef}
              autoPlay
              muted
              playsInline
              className="absolute inset-0 w-full h-full object-cover"
              style={{ opacity: cameraState === "ready" ? 1 : 0 }}
            />

            {/* Viewfinder corners */}
            {[
              "top-2 left-2 border-t-2 border-l-2",
              "top-2 right-2 border-t-2 border-r-2",
              "bottom-2 left-2 border-b-2 border-l-2",
              "bottom-2 right-2 border-b-2 border-r-2",
            ].map((cls, i) => (
              <div
                key={i}
                className={`absolute w-5 h-5 ${cls} pointer-events-none`}
                style={{ borderColor: "#E8602C", opacity: 0.6 }}
              />
            ))}

            {/* Camera icon + label */}
            {cameraState !== "ready" && (
              <div className="flex flex-col items-center gap-2 opacity-70 px-4 text-center">
                {cameraState === "error" ? (
                  <CameraOff className="w-8 h-8" style={{ color: "#D95C5C" }} />
                ) : (
                  <Video className="w-8 h-8" style={{ color: "#8A847D" }} />
                )}
                <p className="text-xs" style={{ color: cameraState === "error" ? "#D95C5C" : "#6E6862" }}>
                  {cameraState === "error" ? cameraError : "Starting camera feed..."}
                </p>
              </div>
            )}

            {/* REC indicator */}
            <div className="absolute top-3 left-3 flex items-center gap-1.5">
              <span className="relative flex h-2 w-2">
                <span
                  className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                  style={{ background: "#D95C5C" }}
                />
                <span
                  className="relative inline-flex rounded-full h-2 w-2"
                  style={{ background: "#D95C5C" }}
                />
              </span>
              <span
                className="text-xs"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 600,
                  color: "#D95C5C",
                  letterSpacing: "0.1em",
                }}
              >
                REC
              </span>
            </div>
          </div>

          {/* Session counters */}
          <div className="grid grid-cols-3 gap-3">
            <div
              className="rounded-xl p-3 flex flex-col gap-1"
              style={{ background: "#141414", border: "1px solid #1E1E1E" }}
            >
              <p
                className="uppercase"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 500,
                  fontSize: "0.6rem",
                  color: "#6E6862",
                  letterSpacing: "0.12em",
                }}
              >
                Attempts
              </p>
              <p
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 700,
                  fontSize: "2rem",
                  color: "#F2EDE8",
                  lineHeight: 1,
                }}
              >
                {attempts}
              </p>
            </div>
            <div
              className="rounded-xl p-3 flex flex-col gap-1"
              style={{ background: "#141414", border: "1px solid #1E1E1E" }}
            >
              <p
                className="uppercase"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 500,
                  fontSize: "0.6rem",
                  color: "#6E6862",
                  letterSpacing: "0.12em",
                }}
              >
                Makes
              </p>
              <p
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 700,
                  fontSize: "2rem",
                  color: makes > 0 ? "#3DB87A" : "#F2EDE8",
                  lineHeight: 1,
                }}
              >
                {makes}
              </p>
            </div>
            <div
              className="rounded-xl p-3 flex flex-col gap-1"
              style={{ background: "#141414", border: "1px solid #1E1E1E" }}
            >
              <p
                className="uppercase"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 500,
                  fontSize: "0.6rem",
                  color: "#6E6862",
                  letterSpacing: "0.12em",
                }}
              >
                FG %
              </p>
              <p
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 700,
                  fontSize: "2rem",
                  color: pct >= 50 ? "#3DB87A" : pct >= 35 ? "#E8602C" : "#D95C5C",
                  lineHeight: 1,
                }}
              >
                {attempts > 0 ? pct : "—"}
                {attempts > 0 && (
                  <span
                    style={{ fontSize: "0.85rem", fontWeight: 500, color: "#6E6862", marginLeft: 1 }}
                  >
                    %
                  </span>
                )}
              </p>
            </div>
          </div>

          {/* Shot history label */}
          {shots.length > 0 && (
            <div className="flex items-center gap-2 mt-1">
              <TrendingUp className="w-3.5 h-3.5" style={{ color: "#6E6862" }} />
              <p
                className="text-xs uppercase tracking-widest"
                style={{ color: "#6E6862", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600 }}
              >
                Recent Shots
              </p>
            </div>
          )}

          {/* Shot history list — desktop only, scrollable */}
          <div className="hidden lg:flex flex-col gap-2 overflow-y-auto flex-1 pr-1" style={{ maxHeight: 320 }}>
            {shots.map((shot, i) => (
              <ShotHistoryCard
                key={shot.id}
                shot={shot}
                index={i}
                isLatest={i === 0}
                onClick={() => setSelectedShot(shot)}
              />
            ))}
          </div>
        </div>

        {/* RIGHT — Latest shot */}
        <div className="hidden sm:flex flex-1 flex-col p-5 gap-5 pb-5 overflow-y-auto overscroll-contain">
          <AnimatePresence mode="wait">
            {!latestShot ? (
              <motion.div
                key="waiting"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex-1 flex flex-col items-center justify-center gap-4 text-center py-16"
              >
                {/* Animated hoop arc */}
                <motion.div
                  animate={{ scale: [1, 1.04, 1] }}
                  transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                >
                  <svg width="80" height="56" viewBox="0 0 80 56" fill="none" opacity={0.25}>
                    <path
                      d="M10 46 Q40 4 70 46"
                      stroke="#E8602C"
                      strokeWidth="3"
                      fill="none"
                      strokeLinecap="round"
                    />
                    <circle cx="40" cy="48" r="12" stroke="#E8602C" strokeWidth="2.5" fill="none" />
                    <line x1="28" y1="48" x2="52" y2="48" stroke="#E8602C" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                </motion.div>
                <div>
                  <p
                    style={{
                      fontFamily: "'Barlow Condensed', sans-serif",
                      fontWeight: 600,
                      fontSize: "1.2rem",
                      color: "#6E6862",
                      letterSpacing: "0.04em",
                    }}
                  >
                    Recording… take your first shot.
                  </p>
                  <p className="text-sm mt-1" style={{ color: "#4A4540" }}>
                    {USE_MOCK
                      ? "Demo mode — a shot will appear in a few seconds"
                      : "Results appear automatically as shots are detected"}
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key={latestShot.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.35, ease: "easeOut" }}
                className="flex flex-col gap-5"
              >
                {/* Result hero */}
                <div
                  className="rounded-2xl p-6 flex flex-col gap-3"
                  style={{
                    background: latestShot.result === "made" ? "#0E1F17" : "#1A0E0E",
                    border: `1px solid ${latestShot.result === "made" ? "#1E4030" : "#3D2020"}`,
                  }}
                >
                  <div className="flex items-center gap-4">
                    {/* Big result label */}
                    <div
                      className="rounded-xl px-5 py-3 flex items-center gap-3"
                      style={{
                        background: latestShot.result === "made" ? "#163025" : "#2D1515",
                      }}
                    >
                      <span
                        style={{
                          fontFamily: "'Barlow Condensed', sans-serif",
                          fontWeight: 700,
                          fontSize: "2.4rem",
                          color: latestShot.result === "made" ? "#3DB87A" : "#D95C5C",
                          lineHeight: 1,
                          letterSpacing: "0.03em",
                        }}
                      >
                        {latestShot.result === "made" ? "MADE" : "MISS"}
                      </span>
                      <span
                        style={{ fontSize: "1.8rem" }}
                      >
                        {latestShot.result === "made" ? "✓" : "✕"}
                      </span>
                    </div>

                    {/* Quality score */}
                    <div className="flex flex-col gap-0.5">
                      <p
                        className="uppercase"
                        style={{
                          fontFamily: "'Barlow Condensed', sans-serif",
                          fontSize: "0.6rem",
                          fontWeight: 600,
                          color: "#6E6862",
                          letterSpacing: "0.12em",
                        }}
                      >
                        Quality Score
                      </p>
                      <p
                        style={{
                          fontFamily: "'Barlow Condensed', sans-serif",
                          fontWeight: 700,
                          fontSize: "2rem",
                          color: latestShot.quality_score >= 70 ? "#3DB87A" : latestShot.quality_score >= 50 ? "#E8602C" : "#D95C5C",
                          lineHeight: 1,
                        }}
                      >
                        {latestShot.quality_score}
                        <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "#6E6862" }}>/100</span>
                      </p>
                    </div>
                  </div>

                  {/* Coaching cue */}
                  <div
                    className="rounded-xl p-4"
                    style={{ background: "#0A0A0A", border: "1px solid #1E1E1E" }}
                  >
                    <p
                      className="text-xs uppercase mb-1.5"
                      style={{
                        fontFamily: "'Barlow Condensed', sans-serif",
                        fontWeight: 600,
                        color: "#E8602C",
                        letterSpacing: "0.12em",
                      }}
                    >
                      Coaching Cue
                    </p>
                    <p className="text-sm leading-relaxed" style={{ color: "#D4CEC8" }}>
                      {latestShot.coaching_cue}
                    </p>
                  </div>
                </div>

                {/* Metrics grid */}
                <div>
                  <p
                    className="text-xs uppercase mb-2.5"
                    style={{
                      fontFamily: "'Barlow Condensed', sans-serif",
                      fontWeight: 600,
                      color: "#6E6862",
                      letterSpacing: "0.12em",
                    }}
                  >
                    Metrics Snapshot
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <MetricTile
                      label="Release Angle"
                      value={latestShot.metrics.release_angle.toFixed(1)}
                      unit="°"
                      highlight={metricHighlight("release_angle", latestShot.metrics.release_angle, latestShot.result)}
                    />
                    <MetricTile
                      label="Arc"
                      value={latestShot.metrics.arc.toFixed(1)}
                      unit="°"
                      highlight={metricHighlight("arc", latestShot.metrics.arc, latestShot.result)}
                    />
                    <MetricTile
                      label="Tempo"
                      value={latestShot.metrics.tempo.toFixed(2)}
                      unit="s"
                      highlight={metricHighlight("tempo", latestShot.metrics.tempo, latestShot.result)}
                    />
                    <MetricTile
                      label="Drift"
                      value={
                        (latestShot.metrics.drift >= 0 ? "+" : "") +
                        latestShot.metrics.drift.toFixed(1)
                      }
                      unit="cm"
                      highlight={metricHighlight("drift", latestShot.metrics.drift, latestShot.result)}
                    />
                  </div>
                </div>

                {/* Action button */}
                <button
                  onClick={() => setSelectedShot(latestShot)}
                  className="self-start flex items-center gap-2 rounded-xl px-5 py-3 transition-colors"
                  style={{
                    background: "#1C1C1C",
                    color: "#F2EDE8",
                    border: "1px solid #2C2C2C",
                    fontFamily: "'Barlow Condensed', sans-serif",
                    fontWeight: 600,
                    fontSize: "0.95rem",
                    letterSpacing: "0.04em",
                  }}
                >
                  VIEW LAST SHOT
                  <ChevronRight className="w-4 h-4" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Mobile shot history */}
          {shots.length > 0 && (
            <div className="lg:hidden flex flex-col gap-2 mt-2">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-3.5 h-3.5" style={{ color: "#6E6862" }} />
                <p
                  className="text-xs uppercase tracking-widest"
                  style={{ color: "#6E6862", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600 }}
                >
                  Recent Shots
                </p>
              </div>
              {shots.slice(0, 5).map((shot, i) => (
                <ShotHistoryCard
                  key={shot.id}
                  shot={shot}
                  index={i}
                  isLatest={i === 0}
                  onClick={() => setSelectedShot(shot)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Replay Panel */}
      <AnimatePresence>
        {selectedShot && (
          <ReplayPanel
            shot={selectedShot}
            onClose={() => setSelectedShot(null)}
          />
        )}
      </AnimatePresence>

      {/* Mobile action bar */}
      <div
        className="sm:hidden fixed bottom-0 left-0 right-0 p-4"
        style={{
          background: "linear-gradient(to top, rgba(10,10,10,0.95), rgba(10,10,10,0.75), rgba(10,10,10,0))",
          zIndex: 20,
        }}
      >
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setShowMobileResults(true)}
            className="w-full flex items-center justify-center gap-2 rounded-xl py-3 text-sm transition-colors"
            style={{
              background: "#1C1C1C",
              color: "#F2EDE8",
              border: "1px solid #2C2C2C",
              fontFamily: "'Barlow Condensed', sans-serif",
              fontWeight: 600,
              letterSpacing: "0.04em",
            }}
          >
            <TrendingUp className="w-4 h-4" />
            RESULTS
          </button>
          <button
            onClick={handleEnd}
            className="w-full flex items-center justify-center gap-2 rounded-xl py-3 text-sm transition-colors"
            style={{
              background: "#1C1C1C",
              color: "#D95C5C",
              border: "1px solid #2A2020",
              fontFamily: "'Barlow Condensed', sans-serif",
              fontWeight: 600,
              letterSpacing: "0.04em",
            }}
          >
            <Square className="w-4 h-4" />
            END SESSION
          </button>
        </div>
      </div>

      {/* Mobile results sheet */}
      <AnimatePresence>
        {showMobileResults && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowMobileResults(false)}
              className="sm:hidden fixed inset-0"
              style={{ background: "rgba(0,0,0,0.72)", zIndex: 24 }}
            />

            <motion.div
              initial={{ y: "100%" }}
              animate={{ y: 0 }}
              exit={{ y: "100%" }}
              transition={{ type: "spring", stiffness: 320, damping: 34 }}
              className="sm:hidden fixed left-0 right-0 bottom-0 rounded-t-2xl flex flex-col"
              style={{
                height: "82dvh",
                background: "#0F0F0F",
                borderTop: "1px solid #1E1E1E",
                zIndex: 25,
              }}
            >
              <div
                className="flex items-center px-5 py-4"
                style={{ borderBottom: "1px solid #1A1A1A" }}
              >
                <p
                  className="text-xs uppercase"
                  style={{
                    fontFamily: "'Barlow Condensed', sans-serif",
                    fontWeight: 600,
                    color: "#6E6862",
                    letterSpacing: "0.12em",
                  }}
                >
                  Results
                </p>
                <div className="flex-1" />
                <button
                  onClick={() => setShowMobileResults(false)}
                  className="w-9 h-9 rounded-xl flex items-center justify-center"
                  style={{ background: "#1C1C1C", color: "#8A847D", border: "1px solid #262626" }}
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-5 pb-28 space-y-5">
                <AnimatePresence mode="wait">
                  {!latestShot ? (
                    <motion.div
                      key="waiting-mobile"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex flex-col items-center justify-center gap-4 text-center py-16"
                    >
                      <motion.div
                        animate={{ scale: [1, 1.04, 1] }}
                        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                      >
                        <svg width="80" height="56" viewBox="0 0 80 56" fill="none" opacity={0.25}>
                          <path d="M10 46 Q40 4 70 46" stroke="#E8602C" strokeWidth="3" fill="none" strokeLinecap="round" />
                          <circle cx="40" cy="48" r="12" stroke="#E8602C" strokeWidth="2.5" fill="none" />
                          <line x1="28" y1="48" x2="52" y2="48" stroke="#E8602C" strokeWidth="3" strokeLinecap="round" />
                        </svg>
                      </motion.div>
                      <div>
                        <p
                          style={{
                            fontFamily: "'Barlow Condensed', sans-serif",
                            fontWeight: 600,
                            fontSize: "1.2rem",
                            color: "#6E6862",
                            letterSpacing: "0.04em",
                          }}
                        >
                          Recording… take your first shot.
                        </p>
                        <p className="text-sm mt-1" style={{ color: "#4A4540" }}>
                          {USE_MOCK
                            ? "Demo mode — a shot will appear in a few seconds"
                            : "Results appear automatically as shots are detected"}
                        </p>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key={latestShot.id + "-mobile"}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.35, ease: "easeOut" }}
                      className="flex flex-col gap-5"
                    >
                      <div
                        className="rounded-2xl p-6 flex flex-col gap-3"
                        style={{
                          background: latestShot.result === "made" ? "#0E1F17" : "#1A0E0E",
                          border: `1px solid ${latestShot.result === "made" ? "#1E4030" : "#3D2020"}`,
                        }}
                      >
                        <div className="flex items-center gap-4">
                          <div
                            className="rounded-xl px-5 py-3 flex items-center gap-3"
                            style={{ background: latestShot.result === "made" ? "#163025" : "#2D1515" }}
                          >
                            <span
                              style={{
                                fontFamily: "'Barlow Condensed', sans-serif",
                                fontWeight: 700,
                                fontSize: "2.4rem",
                                color: latestShot.result === "made" ? "#3DB87A" : "#D95C5C",
                                lineHeight: 1,
                                letterSpacing: "0.03em",
                              }}
                            >
                              {latestShot.result === "made" ? "MADE" : "MISS"}
                            </span>
                            <span style={{ fontSize: "1.8rem" }}>{latestShot.result === "made" ? "✓" : "✕"}</span>
                          </div>
                          <div className="flex flex-col gap-0.5">
                            <p
                              className="uppercase"
                              style={{
                                fontFamily: "'Barlow Condensed', sans-serif",
                                fontSize: "0.6rem",
                                fontWeight: 600,
                                color: "#6E6862",
                                letterSpacing: "0.12em",
                              }}
                            >
                              Quality Score
                            </p>
                            <p
                              style={{
                                fontFamily: "'Barlow Condensed', sans-serif",
                                fontWeight: 700,
                                fontSize: "2rem",
                                color: latestShot.quality_score >= 70 ? "#3DB87A" : latestShot.quality_score >= 50 ? "#E8602C" : "#D95C5C",
                                lineHeight: 1,
                              }}
                            >
                              {latestShot.quality_score}
                              <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "#6E6862" }}>/100</span>
                            </p>
                          </div>
                        </div>

                        <div className="rounded-xl p-4" style={{ background: "#0A0A0A", border: "1px solid #1E1E1E" }}>
                          <p
                            className="text-xs uppercase mb-1.5"
                            style={{
                              fontFamily: "'Barlow Condensed', sans-serif",
                              fontWeight: 600,
                              color: "#E8602C",
                              letterSpacing: "0.12em",
                            }}
                          >
                            Coaching Cue
                          </p>
                          <p className="text-sm leading-relaxed" style={{ color: "#D4CEC8" }}>
                            {latestShot.coaching_cue}
                          </p>
                        </div>
                      </div>

                      <div>
                        <p
                          className="text-xs uppercase mb-2.5"
                          style={{
                            fontFamily: "'Barlow Condensed', sans-serif",
                            fontWeight: 600,
                            color: "#6E6862",
                            letterSpacing: "0.12em",
                          }}
                        >
                          Metrics Snapshot
                        </p>
                        <div className="grid grid-cols-2 gap-3">
                          <MetricTile
                            label="Release Angle"
                            value={latestShot.metrics.release_angle.toFixed(1)}
                            unit="°"
                            highlight={metricHighlight("release_angle", latestShot.metrics.release_angle, latestShot.result)}
                          />
                          <MetricTile
                            label="Arc"
                            value={latestShot.metrics.arc.toFixed(1)}
                            unit="°"
                            highlight={metricHighlight("arc", latestShot.metrics.arc, latestShot.result)}
                          />
                          <MetricTile
                            label="Tempo"
                            value={latestShot.metrics.tempo.toFixed(2)}
                            unit="s"
                            highlight={metricHighlight("tempo", latestShot.metrics.tempo, latestShot.result)}
                          />
                          <MetricTile
                            label="Drift"
                            value={(latestShot.metrics.drift >= 0 ? "+" : "") + latestShot.metrics.drift.toFixed(1)}
                            unit="cm"
                            highlight={metricHighlight("drift", latestShot.metrics.drift, latestShot.result)}
                          />
                        </div>
                      </div>

                      <button
                        onClick={() => {
                          setSelectedShot(latestShot)
                          setShowMobileResults(false)
                        }}
                        className="self-start flex items-center gap-2 rounded-xl px-5 py-3 transition-colors"
                        style={{
                          background: "#1C1C1C",
                          color: "#F2EDE8",
                          border: "1px solid #2C2C2C",
                          fontFamily: "'Barlow Condensed', sans-serif",
                          fontWeight: 600,
                          fontSize: "0.95rem",
                          letterSpacing: "0.04em",
                        }}
                      >
                        VIEW LAST SHOT
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>

                {shots.length > 0 && (
                  <div className="flex flex-col gap-2 mt-2">
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="w-3.5 h-3.5" style={{ color: "#6E6862" }} />
                      <p
                        className="text-xs uppercase tracking-widest"
                        style={{ color: "#6E6862", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600 }}
                      >
                        Recent Shots
                      </p>
                    </div>
                    {shots.slice(0, 8).map((shot, i) => (
                      <ShotHistoryCard
                        key={shot.id}
                        shot={shot}
                        index={i}
                        isLatest={i === 0}
                        onClick={() => {
                          setSelectedShot(shot)
                          setShowMobileResults(false)
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}