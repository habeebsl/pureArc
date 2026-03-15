import { motion } from "motion/react";
import { Shot } from "../../api";
import { formatDistanceToNow } from "date-fns";

interface ShotHistoryCardProps {
  shot: Shot;
  index: number;
  isLatest: boolean;
  onClick: () => void;
}

export function ShotHistoryCard({
  shot,
  index,
  isLatest,
  onClick,
}: ShotHistoryCardProps) {
  const isMade = shot.result === "made";

  return (
    <motion.button
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.04, duration: 0.3, ease: "easeOut" }}
      onClick={onClick}
      className="w-full text-left rounded-xl px-4 py-3 flex items-center gap-4 transition-colors group"
      style={{
        background: isLatest ? "#1C1410" : "#141414",
        border: `1px solid ${isLatest ? "#3A2A1A" : "#1E1E1E"}`,
        cursor: "pointer",
      }}
    >
      {/* Result badge */}
      <div
        className="w-10 h-10 rounded-lg flex-shrink-0 flex items-center justify-center"
        style={{
          background: isMade ? "#122418" : "#221616",
          border: `1px solid ${isMade ? "#1E4030" : "#3D2020"}`,
        }}
      >
        <span
          style={{
            fontFamily: "'Barlow Condensed', sans-serif",
            fontWeight: 700,
            fontSize: "0.68rem",
            letterSpacing: "0.06em",
            color: isMade ? "#3DB87A" : "#D95C5C",
          }}
        >
          {isMade ? "MADE" : "MISS"}
        </span>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p
            className="text-sm truncate"
            style={{ color: "#C8C2BC" }}
          >
            {shot.coaching_cue.length > 52
              ? shot.coaching_cue.slice(0, 52) + "…"
              : shot.coaching_cue}
          </p>
        </div>
        <div className="flex items-center gap-3 mt-0.5">
          <p className="text-xs" style={{ color: "#4A4540" }}>
            {formatDistanceToNow(new Date(shot.timestamp), { addSuffix: true })}
          </p>
          <span style={{ color: "#2A2A2A" }}>·</span>
          <p className="text-xs" style={{ color: "#4A4540" }}>
            {shot.metrics.release_angle.toFixed(1)}° release
          </p>
          <span style={{ color: "#2A2A2A" }}>·</span>
          <p
            className="text-xs"
            style={{ color: isMade ? "#3DB87A" : "#4A4540" }}
          >
            Q {shot.quality_score}
          </p>
        </div>
      </div>

      {/* Arrow */}
      <div
        className="text-xs flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ color: "#E8602C" }}
      >
        →
      </div>
    </motion.button>
  );
}
