import "./StatsGrid.css";

export default function StatsGrid({ stats }) {
  const entries = Object.entries(stats || {});
  if (entries.length === 0) return null;

  return (
    <div className="stats-grid">
      {entries.map(([key, value]) => (
        <div className="stat-item" key={key}>
          <span className="k">{key.replace(/_/g, " ")}</span>
          <span className="v">{typeof value === "number" ? value.toFixed(3) : String(value)}</span>
        </div>
      ))}
    </div>
  );
}