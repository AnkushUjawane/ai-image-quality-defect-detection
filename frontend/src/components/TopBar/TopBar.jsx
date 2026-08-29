import { useHealthCheck } from "../../hooks/useHealthCheck";
import "./TopBar.css";

export default function TopBar() {
  const { online, modelLoaded, checking } = useHealthCheck();

  let dotClass = "dot";
  let text = "connecting…";
  if (!checking) {
    if (online) {
      dotClass += " online";
      text = modelLoaded ? "system nominal" : "model not loaded";
    } else {
      dotClass += " offline";
      text = "backend unreachable";
    }
  }

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">◎</span>
        <span className="brand-text">
          APERTURE<span className="brand-sub">/ inspection unit</span>
        </span>
      </div>
      <div className="status-pill">
        <span className={dotClass} />
        <span>{text}</span>
      </div>
    </header>
  );
}