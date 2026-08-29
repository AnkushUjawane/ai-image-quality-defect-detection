import IssueCard from "../IssueCard/IssueCard";
import StatsGrid from "../StatsGrid/StatsGrid";
import { QUALITY_LABEL_CLASS } from "../../config";
import "./ResultsPanel.css";

export default function ResultsPanel({ status, error, result }) {
  return (
    <section className="panel results" aria-label="Analysis results">
      <h2 className="panel-title">
        <span className="idx">02</span> Readout
      </h2>

      {status === "idle" && (
        <div className="results-empty">
          <p>No inspection run yet.</p>
          <p className="muted">Upload an image and run inspection to see the quality readout.</p>
        </div>
      )}

      {status === "loading" && (
        <div className="results-loading">
          <div className="pulse-bar"><div className="pulse-fill" /></div>
          <p>Extracting features &amp; scoring…</p>
        </div>
      )}

      {status === "error" && (
        <div className="results-error">ERROR — {error}</div>
      )}

      {status === "success" && result && (
        <div className="results-data">
          <ScoreBlock result={result} />

          <div className="issues-block">
            <h3 className="sub-title">Detected issues</h3>
            <div className="issues-list">
              {result.issues.length === 0 ? (
                <div className="no-issues">✓ No issues detected — clean signal across all checks.</div>
              ) : (
                result.issues.map((issue, i) => <IssueCard issue={issue} key={`${issue.type}-${i}`} />)
              )}
            </div>
          </div>

          <div className="stats-block">
            <h3 className="sub-title">Image statistics</h3>
            <StatsGrid stats={result.image_stats} />
          </div>
        </div>
      )}
    </section>
  );
}

function ScoreBlock({ result }) {
  const cls = QUALITY_LABEL_CLASS[result.quality_label] || "degraded";
  return (
    <div className="score-block">
      <div className={`score-ring ${cls}`}>
        <span>{result.quality_score}</span>
      </div>
      <div className="score-meta">
        <span className={`label-tag ${cls}`}>{result.quality_label}</span>
        <span className="score-caption">quality score / 100</span>
      </div>
    </div>
  );
}