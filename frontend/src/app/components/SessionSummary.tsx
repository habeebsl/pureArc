import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { TrendingUp, RotateCcw, AlertCircle, Star, Target, ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import { useSession } from "../context/SessionContext";
import { Shot, Drill, ShotDetail, ReplayAnalysis, getShotDetail, analyzeReplay, generateMockShotDetail, USE_MOCK } from "../../api";
import { ReplayPanel } from "./ReplayPanel";

// ── Stat block ────────────────────────────────────────────────────────────────

function SummaryStatBlock({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-2xl p-5" style={{ background: "#141414", border: "1px solid #1E1E1E" }}>
      <p className="uppercase" style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, fontSize: "0.62rem", color: "#6E6862", letterSpacing: "0.14em" }}>
        {label}
      </p>
      <p className="leading-none" style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "3rem", color: color ?? "#F2EDE8", lineHeight: 1 }}>
        {value}
      </p>
      {sub && <p className="text-xs mt-0.5" style={{ color: "#6E6862" }}>{sub}</p>}
    </div>
  );
}

// ── Drill card ────────────────────────────────────────────────────────────────

function DrillCard({ drill, index }: { drill: Drill; index: number }) {
  const [expanded, setExpanded] = useState(index === 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.3, ease: "easeOut" }}
      className="rounded-2xl overflow-hidden"
      style={{ background: "#141018", border: "1px solid #2A2040" }}
    >
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-4 text-left"
      >
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
               style={{ background: "#9B8BCC22", border: "1px solid #9B8BCC44" }}>
            <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "0.7rem", color: "#9B8BCC" }}>
              {index + 1}
            </span>
          </div>
          <p className="text-sm font-medium" style={{ color: "#F2EDE8" }}>{drill.name}</p>
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "#1E1A30", color: "#9B8BCC" }}>
            {drill.duration_min} min
          </span>
        </div>
        {expanded
          ? <ChevronUp  className="w-4 h-4 flex-shrink-0" style={{ color: "#6E6862" }} />
          : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: "#6E6862" }} />}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 flex flex-col gap-3">
              {drill.steps.length > 0 && (
                <ol className="flex flex-col gap-1.5">
                  {drill.steps.map((step, i) => (
                    <li key={i} className="flex gap-2.5 text-sm" style={{ color: "#C8C2BC" }}>
                      <span className="flex-shrink-0 font-medium" style={{ color: "#9B8BCC" }}>{i + 1}.</span>
                      {step}
                    </li>
                  ))}
                </ol>
              )}
              {drill.links.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-1">
                  {drill.links.map((link, i) => (
                    <a key={i} href={link} target="_blank" rel="noopener noreferrer"
                       className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full transition-colors"
                       style={{ background: "#1E1A30", color: "#9B8BCC", border: "1px solid #2A2040" }}
                       onClick={(e) => e.stopPropagation()}>
                      <ExternalLink className="w-3 h-3" /> Watch Drill
                    </a>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function SessionSummary() {
  const navigate                                     = useNavigate();
  const { shots, drills, analysisResult, clearAnalysis } = useSession();
  const [selectedShot, setSelectedShot]              = useState<Shot | null>(null);

  useEffect(() => {
    if (shots.length === 0 && !analysisResult) navigate("/");
  }, [shots, analysisResult, navigate]);

  const attempts = shots.length;
  const makes    = shots.filter((s) => s.result === "made").length;
  const pct      = attempts > 0 ? Math.round((makes / attempts) * 100) : 0;

  const misses = shots.filter((s) => s.result === "miss");
  const topIssue =
    misses.length > 0
      ? misses[0].coaching_cue
      : "Excellent session — keep building on this consistency.";

  const bestShot =
    shots.length > 0
      ? shots.reduce((best, s) => (s.quality_score > best.quality_score ? s : best))
      : null;

  const handleNewSession = () => {
    clearAnalysis();
    navigate("/");
  };

  return (
    <>
      <div
        className="min-h-screen flex flex-col items-center justify-start pt-10 pb-16 px-5"
        style={{ background: "#0A0A0A", fontFamily: "'Inter', sans-serif" }}
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="w-full max-w-xl flex flex-col gap-8"
        >
          {/* Header */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: "#E8602C" }}>
                <svg width="16" height="16" viewBox="0 0 34 34" fill="none">
                  <circle cx="17" cy="17" r="13" stroke="white" strokeWidth="2.5" />
                  <path d="M4 17 Q17 4 30 17"  stroke="white" strokeWidth="2" fill="none" />
                  <path d="M4 17 Q17 30 30 17" stroke="white" strokeWidth="2" fill="none" />
                  <line x1="17" y1="4" x2="17" y2="30" stroke="white" strokeWidth="2" />
                </svg>
              </div>
              <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "1.1rem", color: "#F2EDE8", letterSpacing: "0.05em" }}>
                PUREARC
              </span>
            </div>
            <h1 className="mt-2" style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "2rem", color: "#F2EDE8", lineHeight: 1.1 }}>
              Analysis Complete
            </h1>
            {analysisResult?.summary && (
              <p className="text-sm" style={{ color: "#6E6862" }}>{analysisResult.summary}</p>
            )}
          </div>

          {/* Stats */}
          {attempts > 0 && (
            <div className="grid grid-cols-3 gap-3">
              <SummaryStatBlock label="Attempts" value={attempts} />
              <SummaryStatBlock label="Makes"    value={makes} color={makes > 0 ? "#3DB87A" : "#F2EDE8"} />
              <SummaryStatBlock label="FG %"     value={`${pct}%`}
                color={pct >= 50 ? "#3DB87A" : pct >= 35 ? "#E8602C" : "#D95C5C"} />
            </div>
          )}

          {/* No shots detected */}
          {attempts === 0 && (
            <div className="rounded-2xl p-5" style={{ background: "#1A1614", border: "1px solid #2E2018" }}>
              <p className="text-sm leading-relaxed" style={{ color: "#C8C2BC" }}>
                No shot attempts were detected. Try a clip where both the shooter and hoop are clearly visible.
              </p>
            </div>
          )}

          {/* AI Drills — 1 to 5 targeted drills */}
          {drills.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Target className="w-4 h-4 flex-shrink-0" style={{ color: "#9B8BCC" }} />
                <p className="text-xs uppercase"
                   style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, color: "#9B8BCC", letterSpacing: "0.12em" }}>
                  Targeted Drills ({drills.length})
                </p>
              </div>
              <div className="flex flex-col gap-3">
                {drills.map((drill, i) => (
                  <DrillCard key={i} drill={drill} index={i} />
                ))}
              </div>
            </div>
          )}

          {/* Top issue */}
          {attempts > 0 && (
            <div className="rounded-2xl p-5 space-y-3" style={{ background: "#141414", border: "1px solid #1E1E1E" }}>
              <div className="flex items-center gap-2.5">
                <AlertCircle className="w-4 h-4 flex-shrink-0" style={{ color: "#E8602C" }} />
                <p className="text-xs uppercase"
                   style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, color: "#E8602C", letterSpacing: "0.12em" }}>
                  Top Recurring Issue
                </p>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: "#C8C2BC" }}>{topIssue}</p>
            </div>
          )}

          {/* Best shot */}
          {bestShot && (
            <div className="rounded-2xl p-5 space-y-2" style={{ background: "#0E1A13", border: "1px solid #1E3A28" }}>
              <div className="flex items-center gap-2.5">
                <Star className="w-4 h-4 flex-shrink-0" style={{ color: "#3DB87A" }} />
                <p className="text-xs uppercase"
                   style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, color: "#3DB87A", letterSpacing: "0.12em" }}>
                  Best Shot
                </p>
              </div>
              <div className="flex items-center gap-4">
                <p style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "2.5rem", color: "#3DB87A", lineHeight: 1 }}>
                  {bestShot.quality_score}
                  <span style={{ fontSize: "1rem", fontWeight: 500, color: "#6E6862", marginLeft: 2 }}>/100</span>
                </p>
                <div className="grid grid-cols-2 gap-x-5 gap-y-0.5 text-xs" style={{ color: "#6E6862" }}>
                  <span>Release <span style={{ color: "#C8C2BC" }}>{bestShot.metrics.release_angle.toFixed(1)}°</span></span>
                  <span>Arc     <span style={{ color: "#C8C2BC" }}>{bestShot.metrics.arc.toFixed(1)}</span></span>
                  <span>Tempo   <span style={{ color: "#C8C2BC" }}>{bestShot.metrics.tempo.toFixed(2)}s</span></span>
                  <span>Drift   <span style={{ color: "#C8C2BC" }}>{bestShot.metrics.drift >= 0 ? "+" : ""}{bestShot.metrics.drift.toFixed(1)}cm</span></span>
                </div>
              </div>
            </div>
          )}

          {/* Shot breakdown + replay */}
          {shots.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-3.5 h-3.5" style={{ color: "#6E6862" }} />
                <p className="text-xs uppercase"
                   style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, color: "#6E6862", letterSpacing: "0.12em" }}>
                  Shot Breakdown
                </p>
                <span className="text-xs ml-1" style={{ color: "#4A4540" }}>· click to replay &amp; analyze</span>
              </div>
              <div className="flex gap-2 flex-wrap">
                {shots
                  .slice()
                  .reverse()
                  .map((shot) => (
                    <button
                      key={shot.id}
                      onClick={() => setSelectedShot(shot)}
                      className="w-9 h-9 rounded-lg flex items-center justify-center transition-transform hover:scale-110"
                      style={{
                        background: shot.result === "made" ? "#122418" : "#221616",
                        border: `1px solid ${shot.result === "made" ? "#1E4030" : "#3D2020"}`,
                        cursor: "pointer",
                      }}
                      title={shot.result}
                    >
                      <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "0.65rem",
                                     color: shot.result === "made" ? "#3DB87A" : "#D95C5C" }}>
                        {shot.result === "made" ? "✓" : "✕"}
                      </span>
                    </button>
                  ))}
              </div>
            </div>
          )}

          {/* Analyze another video */}
          <motion.button
            onClick={handleNewSession}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full flex items-center justify-center gap-2.5 rounded-2xl py-4 mt-2"
            style={{ background: "#E8602C", color: "#fff", fontFamily: "'Barlow Condensed', sans-serif",
                     fontWeight: 700, fontSize: "1.1rem", letterSpacing: "0.08em" }}
          >
            <RotateCcw className="w-4 h-4" />
            ANALYZE ANOTHER VIDEO
          </motion.button>
        </motion.div>
      </div>

      {/* ReplayPanel overlay */}
      <AnimatePresence>
        {selectedShot && (
          <ReplayPanel shot={selectedShot} onClose={() => setSelectedShot(null)} />
        )}
      </AnimatePresence>
    </>
  );
}
