import { useState, useEffect } from "react";
import { motion } from "motion/react";
import {
  X,
  Play,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  ChevronRight,
  Loader2,
  Video,
} from "lucide-react";
import {
  Shot,
  ShotDetail,
  ReplayAnalysis,
  getShotDetail,
  analyzeReplay,
  generateMockShotDetail,
} from "../../api";
import { MetricTile } from "./MetricTile";
import { USE_MOCK } from "../../api";

interface ReplayPanelProps {
  shot: Shot;
  onClose: () => void;
}

type AnalysisState = "idle" | "loading" | "done" | "error";

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

export function ReplayPanel({ shot, onClose }: ReplayPanelProps) {
  const [detail, setDetail] = useState<ShotDetail | null>(null);
  const [analysis, setAnalysis] = useState<ReplayAnalysis | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [clipError, setClipError] = useState(false);

  useEffect(() => {
    // Disable body scroll
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; };
  }, []);

  useEffect(() => {
    const load = async () => {
      if (USE_MOCK) {
        setDetail(generateMockShotDetail(shot));
        return;
      }
      try {
        const d = await getShotDetail(shot.id);
        setDetail(d);
        if (!d.clip_url) setClipError(true);
      } catch {
        // Fallback: use existing shot data
        setDetail(generateMockShotDetail(shot));
      }
    };
    load();
  }, [shot]);

  const handleAnalyze = async () => {
    setAnalysisState("loading");
    try {
      const res = await analyzeReplay(shot.id, {
        include_drill: true,
        detail_level: "detailed",
      });
      setAnalysis(res);
      setAnalysisState("done");
    } catch {
      // Fallback: still show existing data
      setAnalysisState("error");
    }
  };

  const isMade = shot.result === "made";

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-40"
        style={{ background: "rgba(0,0,0,0.75)" }}
      />

      {/* Panel */}
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", stiffness: 320, damping: 34 }}
        className="fixed top-0 right-0 h-full z-50 flex flex-col overflow-hidden"
        style={{
          width: "min(560px, 100vw)",
          background: "#0F0F0F",
          borderLeft: "1px solid #1E1E1E",
          fontFamily: "'Inter', sans-serif",
        }}
      >
        {/* Panel header */}
        <div
          className="flex items-center px-5 py-4 flex-shrink-0"
          style={{ borderBottom: "1px solid #1A1A1A" }}
        >
          <div>
            <p
              className="uppercase tracking-widest text-xs"
              style={{
                fontFamily: "'Barlow Condensed', sans-serif",
                fontWeight: 600,
                color: "#6E6862",
                letterSpacing: "0.12em",
              }}
            >
              Shot Review
            </p>
            <div className="flex items-center gap-3 mt-0.5">
              <span
                className="px-2 py-0.5 rounded"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 700,
                  fontSize: "1rem",
                  background: isMade ? "#122418" : "#221616",
                  color: isMade ? "#3DB87A" : "#D95C5C",
                  letterSpacing: "0.04em",
                }}
              >
                {isMade ? "MADE" : "MISS"}
              </span>
              <span className="text-sm" style={{ color: "#6E6862" }}>
                Q{" "}
                <span style={{ color: "#C8C2BC" }}>{shot.quality_score}</span>
                /100
              </span>
            </div>
          </div>
          <div className="flex-1" />
          <button
            onClick={onClose}
            className="w-9 h-9 rounded-xl flex items-center justify-center transition-colors"
            style={{ background: "#1C1C1C", color: "#8A847D", border: "1px solid #262626" }}
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
          {/* Clip section */}
          <div
            className="rounded-2xl overflow-hidden"
            style={{ border: "1px solid #1E1E1E" }}
          >
            {clipError || !detail?.clip_url ? (
              <div
                className="h-44 flex flex-col items-center justify-center gap-2"
                style={{ background: "#0C0C0C" }}
              >
                <Video className="w-8 h-8 opacity-20" style={{ color: "#8A847D" }} />
                <p className="text-xs" style={{ color: "#4A4540" }}>
                  Clip not available
                </p>
              </div>
            ) : (
              <div
                className="h-44 flex items-center justify-center relative"
                style={{ background: "#0C0C0C" }}
              >
                <button
                  className="w-14 h-14 rounded-full flex items-center justify-center"
                  style={{ background: "#E8602C" }}
                >
                  <Play className="w-6 h-6 text-white ml-0.5" />
                </button>
                <p
                  className="absolute bottom-3 right-3 text-xs"
                  style={{ color: "#4A4540" }}
                >
                  Tap to play
                </p>
              </div>
            )}
          </div>

          {/* Metrics */}
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
              Metrics
            </p>
            <div className="grid grid-cols-2 gap-2.5">
              <MetricTile
                label="Release Angle"
                value={shot.metrics.release_angle.toFixed(1)}
                unit="°"
                highlight={metricHighlight("release_angle", shot.metrics.release_angle, shot.result)}
              />
              <MetricTile
                label="Arc"
                value={shot.metrics.arc.toFixed(1)}
                unit="°"
                highlight={metricHighlight("arc", shot.metrics.arc, shot.result)}
              />
              <MetricTile
                label="Tempo"
                value={shot.metrics.tempo.toFixed(2)}
                unit="s"
                highlight={metricHighlight("tempo", shot.metrics.tempo, shot.result)}
              />
              <MetricTile
                label="Drift"
                value={(shot.metrics.drift >= 0 ? "+" : "") + shot.metrics.drift.toFixed(1)}
                unit="cm"
                highlight={metricHighlight("drift", shot.metrics.drift, shot.result)}
              />
            </div>
          </div>

          {/* Deterministic feedback */}
          {detail && (
            <>
              {detail.mistakes.length > 0 && (
                <div
                  className="rounded-xl p-4 space-y-2"
                  style={{ background: "#140E0E", border: "1px solid #2A1818" }}
                >
                  <p
                    className="text-xs uppercase mb-3"
                    style={{
                      fontFamily: "'Barlow Condensed', sans-serif",
                      fontWeight: 600,
                      color: "#D95C5C",
                      letterSpacing: "0.12em",
                    }}
                  >
                    Fix First
                  </p>
                  {detail.mistakes.map((m, i) => (
                    <div key={i} className="flex items-start gap-2.5">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "#D95C5C" }} />
                      <p className="text-sm" style={{ color: "#C8C2BC" }}>
                        {m}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {isMade && (
                <div
                  className="rounded-xl p-4 space-y-2"
                  style={{ background: "#0E1A13", border: "1px solid #1E3A28" }}
                >
                  <p
                    className="text-xs uppercase mb-3"
                    style={{
                      fontFamily: "'Barlow Condensed', sans-serif",
                      fontWeight: 600,
                      color: "#3DB87A",
                      letterSpacing: "0.12em",
                    }}
                  >
                    What Went Well
                  </p>
                  <div className="flex items-start gap-2.5">
                    <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "#3DB87A" }} />
                    <p className="text-sm" style={{ color: "#C8C2BC" }}>
                      {detail.context}
                    </p>
                  </div>
                </div>
              )}
            </>
          )}

          {/* AI Analysis section */}
          {analysisState === "idle" && (
            <button
              onClick={handleAnalyze}
              className="w-full flex items-center justify-center gap-2.5 rounded-2xl py-4 transition-all"
              style={{
                background: "#141414",
                border: "1px solid #2C2C2C",
                color: "#F2EDE8",
                fontFamily: "'Barlow Condensed', sans-serif",
                fontWeight: 600,
                fontSize: "0.95rem",
                letterSpacing: "0.06em",
              }}
            >
              <Sparkles className="w-4 h-4" style={{ color: "#E8602C" }} />
              ANALYZE REPLAY
              <ChevronRight className="w-4 h-4" style={{ color: "#6E6862" }} />
            </button>
          )}

          {analysisState === "loading" && (
            <div
              className="rounded-2xl p-6 flex flex-col items-center gap-3"
              style={{ background: "#141414", border: "1px solid #1E1E1E" }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Loader2 className="w-6 h-6" style={{ color: "#E8602C" }} />
              </motion.div>
              <div className="text-center">
                <p className="text-sm" style={{ color: "#F2EDE8" }}>
                  Running AI analysis…
                </p>
                <p className="text-xs mt-0.5" style={{ color: "#6E6862" }}>
                  Comparing against model biomechanics
                </p>
              </div>
            </div>
          )}

          {analysisState === "error" && (
            <div
              className="rounded-xl p-4"
              style={{ background: "#1A1410", border: "1px solid #2E2218" }}
            >
              <p className="text-sm" style={{ color: "#C8A87A" }}>
                AI analysis unavailable — showing fallback coaching above.
              </p>
              <button
                onClick={handleAnalyze}
                className="text-xs mt-2 underline"
                style={{ color: "#E8602C" }}
              >
                Retry
              </button>
            </div>
          )}

          {analysisState === "done" && analysis && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
              className="space-y-4"
            >
              {/* AI badge */}
              <div className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5" style={{ color: "#E8602C" }} />
                <p
                  className="text-xs uppercase tracking-widest"
                  style={{
                    fontFamily: "'Barlow Condensed', sans-serif",
                    fontWeight: 600,
                    color: "#E8602C",
                    letterSpacing: "0.12em",
                  }}
                >
                  AI Analysis
                </p>
              </div>

              {/* What went well */}
              {analysis.what_went_well.length > 0 && (
                <div
                  className="rounded-xl p-4 space-y-2"
                  style={{ background: "#0E1A13", border: "1px solid #1E3A28" }}
                >
                  <p
                    className="text-xs uppercase mb-2"
                    style={{
                      fontFamily: "'Barlow Condensed', sans-serif",
                      fontWeight: 600,
                      color: "#3DB87A",
                      letterSpacing: "0.12em",
                    }}
                  >
                    What Went Well
                  </p>
                  {analysis.what_went_well.map((w, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: "#3DB87A" }} />
                      <p className="text-sm" style={{ color: "#C8C2BC" }}>
                        {w}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* What to fix */}
              {analysis.what_to_fix.length > 0 && (
                <div
                  className="rounded-xl p-4 space-y-2"
                  style={{ background: "#140E0E", border: "1px solid #2A1818" }}
                >
                  <p
                    className="text-xs uppercase mb-2"
                    style={{
                      fontFamily: "'Barlow Condensed', sans-serif",
                      fontWeight: 600,
                      color: "#D95C5C",
                      letterSpacing: "0.12em",
                    }}
                  >
                    What to Fix
                  </p>
                  {analysis.what_to_fix.map((f, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: "#D95C5C" }} />
                      <p className="text-sm" style={{ color: "#C8C2BC" }}>
                        {f}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* Drills */}
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
                  Drills
                </p>
                <div className="space-y-2.5">
                  {[
                    { label: "Primary Drill", drill: analysis.primary_drill },
                    { label: "Backup Drill", drill: analysis.backup_drill },
                  ].map(({ label, drill }) => (
                    <a
                      key={label}
                      href={drill.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex flex-col gap-1 rounded-xl p-4 group transition-colors block"
                      style={{
                        background: "#141414",
                        border: "1px solid #222",
                        textDecoration: "none",
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <p
                          className="text-xs uppercase"
                          style={{
                            fontFamily: "'Barlow Condensed', sans-serif",
                            fontWeight: 600,
                            color: "#E8602C",
                            letterSpacing: "0.08em",
                          }}
                        >
                          {label}
                        </p>
                        <ExternalLink className="w-3 h-3 opacity-40 group-hover:opacity-100 transition-opacity" style={{ color: "#E8602C" }} />
                      </div>
                      <p className="text-sm" style={{ color: "#F2EDE8" }}>
                        {drill.name}
                      </p>
                      <p className="text-xs leading-relaxed" style={{ color: "#6E6862" }}>
                        {drill.description}
                      </p>
                    </a>
                  ))}
                </div>
              </div>

              {/* Next shot focus */}
              <div
                className="rounded-xl p-4"
                style={{
                  background: "#141018",
                  border: "1px solid #2A2040",
                }}
              >
                <p
                  className="text-xs uppercase mb-2"
                  style={{
                    fontFamily: "'Barlow Condensed', sans-serif",
                    fontWeight: 600,
                    color: "#9B8BCC",
                    letterSpacing: "0.12em",
                  }}
                >
                  Next Shot Focus
                </p>
                <p className="text-sm leading-relaxed" style={{ color: "#C8C2BC" }}>
                  {analysis.next_shot_focus}
                </p>
              </div>
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <div
          className="flex-shrink-0 px-5 py-4 flex gap-3"
          style={{ borderTop: "1px solid #1A1A1A" }}
        >
          <button
            onClick={onClose}
            className="flex-1 rounded-xl py-3 text-sm transition-colors"
            style={{
              background: "#1C1C1C",
              color: "#F2EDE8",
              border: "1px solid #2C2C2C",
              fontFamily: "'Barlow Condensed', sans-serif",
              fontWeight: 600,
              fontSize: "0.95rem",
              letterSpacing: "0.06em",
            }}
          >
            ← CONTINUE SHOOTING
          </button>
        </div>
      </motion.div>
    </>
  );
}
