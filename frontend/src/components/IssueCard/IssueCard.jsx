import "./IssueCard.css";

export default function IssueCard({ issue }) {
  const { type, severity, confidence, explanation } = issue;
  return (
    <div className={`issue-card ${severity}`}>
      <div className="issue-head">
        <span className="issue-type">{type}</span>
        <span className="issue-tags">
          <span className={`sev-badge ${severity}`}>{severity}</span>
          <span>conf {(confidence * 100).toFixed(0)}%</span>
        </span>
      </div>
      <div className="issue-explanation">{explanation}</div>
    </div>
  );
}