import { useRef, useState } from "react";
import { ALLOWED_MIME_TYPES, MAX_FILE_SIZE_BYTES } from "../../config";
import "./Dropzone.css";

export default function Dropzone({ previewSrc, onFileSelected, onAnalyze, analyzing, canAnalyze }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [hint, setHint] = useState({ text: "", error: false });

  function validateAndSelect(file) {
    if (!file) return;
    if (!ALLOWED_MIME_TYPES.includes(file.type)) {
      setHint({ text: `Unsupported type: ${file.type || "unknown"}`, error: true });
      return;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setHint({ text: "File exceeds 15MB limit", error: true });
      return;
    }
    setHint({ text: "", error: false });
    onFileSelected(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    validateAndSelect(e.dataTransfer.files?.[0]);
  }

  return (
    <section className="panel intake" aria-label="Image intake">
      <h2 className="panel-title">
        <span className="idx">01</span> Intake
      </h2>

      <div
        className={`dropzone${dragOver ? " dragover" : ""}`}
        tabIndex={0}
        role="button"
        aria-label="Upload image for analysis"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={(e) => { e.preventDefault(); setDragOver(false); }}
        onDrop={handleDrop}
      >
        {!previewSrc ? (
          <div className="dz-empty">
            <div className="reticle">
              <span className="corner tl" /><span className="corner tr" />
              <span className="corner bl" /><span className="corner br" />
              <span className="crosshair" />
            </div>
            <p className="dz-title">Drop image or click to select</p>
            <p className="dz-sub">JPEG · PNG · WEBP · BMP · TIFF — up to 15MB</p>
          </div>
        ) : (
          <div className="dz-preview">
            <div className="frame">
              <img src={previewSrc} alt="Uploaded preview" />
              <div className="reticle overlay">
                <span className="corner tl" /><span className="corner tr" />
                <span className="corner bl" /><span className="corner br" />
              </div>
              <div className={`scanline${analyzing ? " active" : ""}`} />
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_MIME_TYPES.join(",")}
          hidden
          onChange={(e) => validateAndSelect(e.target.files?.[0])}
        />
      </div>

      <button className="btn-primary" disabled={!canAnalyze || analyzing} onClick={onAnalyze}>
        {analyzing ? "Inspecting…" : "Run inspection"}
      </button>
      <p className={`hint${hint.error ? " err" : ""}`}>{hint.text}</p>
    </section>
  );
}