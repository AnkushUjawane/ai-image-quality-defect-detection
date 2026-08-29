import { useCallback, useEffect, useState } from "react";
import "./App.css";
import TopBar from "./components/TopBar/TopBar";
import Dropzone from "./components/Dropzone/Dropzone";
import ResultsPanel from "./components/ResultsPanel/ResultsPanel";
import HistoryList from "./components/HistoryList/HistoryList";
import { analyzeImage, getAnalysis, imageUrl, listAnalyses } from "./api/client";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);

  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState(false);

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(false);
    try {
      const items = await listAnalyses(25);
      setHistory(items);
    } catch {
      setHistoryError(true);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    // Fetch-on-mount: intentional async data load, not a state sync effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshHistory();
  }, [refreshHistory]);

  function handleFileSelected(file) {
    setSelectedFile(file);
    setPreviewSrc(URL.createObjectURL(file));
  }

  async function handleAnalyze() {
    if (!selectedFile) return;
    setStatus("loading");
    setError("");
    try {
      const data = await analyzeImage(selectedFile);
      setResult(data);
      setStatus("success");
      refreshHistory();
    } catch (e) {
      setError(e.message || "Unexpected error during analysis");
      setStatus("error");
    }
  }

  async function handleSelectHistoryItem(id) {
    setStatus("loading");
    setError("");
    try {
      const data = await getAnalysis(id);
      setResult(data);
      setStatus("success");
      if (data.image_url) setPreviewSrc(imageUrl(data.image_url));
    } catch (e) {
      setError(e.message || "Could not load that analysis");
      setStatus("error");
    }
  }

  return (
    <>
      <div className="scan-overlay" aria-hidden="true" />
      <TopBar />
      <main className="layout">
        <Dropzone
          previewSrc={previewSrc}
          onFileSelected={handleFileSelected}
          onAnalyze={handleAnalyze}
          analyzing={status === "loading"}
          canAnalyze={!!selectedFile}
        />
        <ResultsPanel status={status} error={error} result={result} />
        <HistoryList
          items={history}
          loading={historyLoading}
          error={historyError}
          onSelect={handleSelectHistoryItem}
        />
      </main>
      <footer className="footer">
        <span>
          AI-Powered Image Quality &amp; Defect Detection — hybrid CV + ML pipeline
          (RandomForest ensembles, 13 engineered features)
        </span>
      </footer>
    </>
  );
}