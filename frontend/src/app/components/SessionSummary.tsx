import { useEffect } from "react";
import { useNavigate } from "react-router";
import { motion } from "motion/react";
import { TrendingUp, RotateCcw, AlertCircle, Star, Target } from "lucide-react";
import { useSession } from "../context/SessionContext";

function SummaryStatBlock({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  return (
    <div
      className="flex flex-col gap-1 rounded-2xl p-5"
      style={{ background: "#141414", border: "1px solid #1E1E1E" }}
    >
      <p
        className="uppercase"
        style={{
          fontFamily: "'Barlow Condensed', sans-serif",
          fontWeight: 600,
          fontSize: "0.62rem",
          color: "#6E6862",
          letterSpacing: "0.14em",
        }}
      >
        {label}
      </p>
      <p
        className="leading-none"
        style={{
          fontFamily: "'Barlow Condensed', sans-serif",
          fontWeight: 700,
          fontSize: "3rem",
          color: color ?? "#F2EDE8",
          lineHeight: 1,
        }}
      >
        {value}
      </p>
      {sub && (
        <p className="text-xs mt-0.5" style={{ color: "#6E6862" }}>
          {sub}
        </p>
      )}
    </div>
  );
}

export function SessionSummary() {
  const navigate = useNavigate();
  const { shots, clearSession } = useSession();

  useEffect(() => {
    if (shots.length === 0) navigate("/");
  }, [shots, navigate]);

  const attempts = shots.length;
  const makes = shots.filter((s) => s.result === "made").length;
  const pct = attempts > 0 ? Math.round((makes / attempts) * 100) : 0;

  // Compute top recurring issue from miss shots
  const misses = shots.filter((s) => s.result === "miss");
  const topIssue =
    misses.length > 0
      ? misses[0].coaching_cue
      : "Excellent session — keep building on this consistency.";

  // Compute best shot by quality score
  const bestShot =
    shots.length > 0
      ? shots.reduce((best, s) =>
          s.quality_score > best.quality_score ? s : best
        )
      : null;

  // Recommend drill based on most common miss metric
  const flatArcMisses = misses.filter((s) => s.metrics.arc < 42).length;
  const driftMisses = misses.filter((s) => Math.abs(s.metrics.drift) > 4).length;
  const tempMisses = misses.filter((s) => s.metrics.tempo < 0.70).length;

  let recommendedDrill = "One-Hand Form Shooting — rebuild your release shape.";
  if (flatArcMisses >= driftMisses && flatArcMisses >= tempMisses)
    recommendedDrill = "Arc Building Drill — shoot up and over a high object from close range.";
  else if (driftMisses >= flatArcMisses && driftMisses >= tempMisses)
    recommendedDrill = "Wall Shooting Drill — stand close to a wall so your guide hand can't push.";
  else if (tempMisses > 0)
    recommendedDrill = "Slow-Motion Shooting — exaggerate your tempo, pause at the top before releasing.";

  const handleNewSession = () => {
    clearSession();
    navigate("/");
  };

  return (
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
                fontSize: "1.1rem",
                color: "#F2EDE8",
                letterSpacing: "0.05em",
              }}
            >
              PUREARC
            </span>
          </div>
          <h1
            className="mt-2"
            style={{
              fontFamily: "'Barlow Condensed', sans-serif",
              fontWeight: 700,
              fontSize: "2rem",
              color: "#F2EDE8",
              lineHeight: 1.1,
            }}
          >
            Session Complete
          </h1>
          <p className="text-sm" style={{ color: "#6E6862" }}>
            Here's what happened on the court.
          </p>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-3 gap-3">
          <SummaryStatBlock label="Attempts" value={attempts} />
          <SummaryStatBlock
            label="Makes"
            value={makes}
            color={makes > 0 ? "#3DB87A" : "#F2EDE8"}
          />
          <SummaryStatBlock
            label="FG %"
            value={`${pct}%`}
            color={pct >= 50 ? "#3DB87A" : pct >= 35 ? "#E8602C" : "#D95C5C"}
          />
        </div>

        {/* Top recurring issue */}
        <div
          className="rounded-2xl p-5 space-y-3"
          style={{ background: "#141414", border: "1px solid #1E1E1E" }}
        >
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 flex-shrink-0" style={{ color: "#E8602C" }} />
            <p
              className="text-xs uppercase"
              style={{
                fontFamily: "'Barlow Condensed', sans-serif",
                fontWeight: 600,
                color: "#E8602C",
                letterSpacing: "0.12em",
              }}
            >
              Top Recurring Issue
            </p>
          </div>
          <p className="text-sm leading-relaxed" style={{ color: "#C8C2BC" }}>
            {topIssue}
          </p>
        </div>

        {/* Best shot */}
        {bestShot && (
          <div
            className="rounded-2xl p-5 space-y-2"
            style={{ background: "#0E1A13", border: "1px solid #1E3A28" }}
          >
            <div className="flex items-center gap-2.5">
              <Star className="w-4 h-4 flex-shrink-0" style={{ color: "#3DB87A" }} />
              <p
                className="text-xs uppercase"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 600,
                  color: "#3DB87A",
                  letterSpacing: "0.12em",
                }}
              >
                Best Shot
              </p>
            </div>
            <div className="flex items-center gap-4">
              <p
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 700,
                  fontSize: "2.5rem",
                  color: "#3DB87A",
                  lineHeight: 1,
                }}
              >
                {bestShot.quality_score}
                <span style={{ fontSize: "1rem", fontWeight: 500, color: "#6E6862", marginLeft: 2 }}>
                  /100
                </span>
              </p>
              <div className="grid grid-cols-2 gap-x-5 gap-y-0.5 text-xs" style={{ color: "#6E6862" }}>
                <span>
                  Release{" "}
                  <span style={{ color: "#C8C2BC" }}>
                    {bestShot.metrics.release_angle.toFixed(1)}°
                  </span>
                </span>
                <span>
                  Arc{" "}
                  <span style={{ color: "#C8C2BC" }}>
                    {bestShot.metrics.arc.toFixed(1)}°
                  </span>
                </span>
                <span>
                  Tempo{" "}
                  <span style={{ color: "#C8C2BC" }}>
                    {bestShot.metrics.tempo.toFixed(2)}s
                  </span>
                </span>
                <span>
                  Drift{" "}
                  <span style={{ color: "#C8C2BC" }}>
                    {bestShot.metrics.drift >= 0 ? "+" : ""}
                    {bestShot.metrics.drift.toFixed(1)}cm
                  </span>
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Recommended drill */}
        <div
          className="rounded-2xl p-5 space-y-3"
          style={{ background: "#141018", border: "1px solid #2A2040" }}
        >
          <div className="flex items-center gap-2.5">
            <Target className="w-4 h-4 flex-shrink-0" style={{ color: "#9B8BCC" }} />
            <p
              className="text-xs uppercase"
              style={{
                fontFamily: "'Barlow Condensed', sans-serif",
                fontWeight: 600,
                color: "#9B8BCC",
                letterSpacing: "0.12em",
              }}
            >
              Recommended Next Drill
            </p>
          </div>
          <p className="text-sm leading-relaxed" style={{ color: "#C8C2BC" }}>
            {recommendedDrill}
          </p>
        </div>

        {/* Shot breakdown mini list */}
        {shots.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-3.5 h-3.5" style={{ color: "#6E6862" }} />
              <p
                className="text-xs uppercase"
                style={{
                  fontFamily: "'Barlow Condensed', sans-serif",
                  fontWeight: 600,
                  color: "#6E6862",
                  letterSpacing: "0.12em",
                }}
              >
                Shot Breakdown
              </p>
            </div>
            <div className="flex gap-1 flex-wrap">
              {shots
                .slice()
                .reverse()
                .map((shot) => (
                  <div
                    key={shot.id}
                    className="w-7 h-7 rounded flex items-center justify-center"
                    style={{
                      background:
                        shot.result === "made" ? "#122418" : "#221616",
                      border: `1px solid ${shot.result === "made" ? "#1E4030" : "#3D2020"}`,
                      title: shot.result,
                    }}
                    title={shot.result}
                  >
                    <span
                      style={{
                        fontFamily: "'Barlow Condensed', sans-serif",
                        fontWeight: 700,
                        fontSize: "0.6rem",
                        color: shot.result === "made" ? "#3DB87A" : "#D95C5C",
                      }}
                    >
                      {shot.result === "made" ? "✓" : "✕"}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Start New Session */}
        <motion.button
          onClick={handleNewSession}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="w-full flex items-center justify-center gap-2.5 rounded-2xl py-4 mt-2"
          style={{
            background: "#E8602C",
            color: "#fff",
            fontFamily: "'Barlow Condensed', sans-serif",
            fontWeight: 700,
            fontSize: "1.1rem",
            letterSpacing: "0.08em",
          }}
        >
          <RotateCcw className="w-5 h-5" />
          START NEW SESSION
        </motion.button>
      </motion.div>
    </div>
  );
}
