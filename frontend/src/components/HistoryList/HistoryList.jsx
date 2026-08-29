import { imageUrl } from "../../api/client";
import { QUALITY_LABEL_CLASS } from "../../config";
import "./HistoryList.css";

export default function HistoryList({ items, loading, error, onSelect }) {
  return (
    <section className="panel history" aria-label="Analysis history">
      <h2 className="panel-title">
        <span className="idx">03</span> Log
      </h2>

      <div className="history-list">
        {loading && <p className="muted small">Loading history…</p>}
        {error && <p className="muted small">Could not load history — backend unreachable.</p>}
        {!loading && !error && items.length === 0 && (
          <p className="muted small">No inspections logged yet.</p>
        )}
        {!loading && !error && items.map((item) => (
          <div className="history-item" key={item.id} onClick={() => onSelect(item.id)}>
            <img className="history-thumb" src={imageUrl(item.image_url)} alt="" />
            <div className="history-meta">
              <div className="history-name">{item.filename}</div>
              <div className="history-sub">
                {item.issue_count} issue(s) · {new Date(item.created_at).toLocaleString()}
              </div>
            </div>
            <div className={`history-score ${QUALITY_LABEL_CLASS[item.quality_label] || "degraded"}`}>
              {item.quality_score}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}