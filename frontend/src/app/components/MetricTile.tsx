interface MetricTileProps {
  label: string;
  value: string | number;
  unit?: string;
  highlight?: "good" | "warn" | "neutral";
}

export function MetricTile({ label, value, unit, highlight = "neutral" }: MetricTileProps) {
  const valueColor =
    highlight === "good"
      ? "#3DB87A"
      : highlight === "warn"
      ? "#D95C5C"
      : "#F2EDE8";

  return (
    <div
      className="flex flex-col gap-1 rounded-xl p-3"
      style={{ background: "#1C1C1C", border: "1px solid #262626" }}
    >
      <p
        className="uppercase tracking-widest"
        style={{
          fontFamily: "'Barlow Condensed', sans-serif",
          fontWeight: 500,
          fontSize: "0.62rem",
          color: "#6E6862",
          letterSpacing: "0.12em",
        }}
      >
        {label}
      </p>
      <p
        className="leading-none"
        style={{
          fontFamily: "'Barlow Condensed', sans-serif",
          fontWeight: 700,
          fontSize: "1.7rem",
          color: valueColor,
        }}
      >
        {value}
        {unit && (
          <span
            style={{
              fontSize: "0.85rem",
              fontWeight: 500,
              color: "#6E6862",
              marginLeft: 2,
            }}
          >
            {unit}
          </span>
        )}
      </p>
    </div>
  );
}
