import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";
import {
  buildFormData,
  buildHeaders,
  buildUrl,
  getDefaultApiBase,
  normalizeBase,
  readJson,
} from "./api.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Tooltip,
  Legend
);

const CHART_THEME = {
  axis: "#9ea9be",
  grid: "rgba(255, 255, 255, 0.08)",
  accent: "#3a4bff",
  accentSoft: "rgba(58, 75, 255, 0.4)",
  warn: "#ff8b5d",
};

const DEFAULT_STATUS = {
  api: "Not connected",
  detail: "Start the backend to see live status.",
};

const LS_KEY_BASE = "rigvisionx_api_base";
const LS_KEY_API = "rigvisionx_api_key";

const emptyArray = [];
const PREDICT_ENDPOINTS = ["/predict/failure-risk", "/failure-risk/predict"];
const HISTORY_ENDPOINT = "/history?limit=20";

const formatNumber = (value) => {
  if (value === null || value === undefined) return "-";
  if (Number.isNaN(value)) return "-";
  if (typeof value !== "number") return String(value);
  return value.toFixed(3).replace(/\.0+$/, "");
};

const formatTimestamp = (value) => {
  if (!value) return "";
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
  } catch {
    return String(value);
  }
};

const shortLabel = (value, index) => {
  const base = formatTimestamp(value) || `#${index + 1}`;
  return base.length > 16 ? `${base.slice(0, 16)}...` : base;
};

const summarizeMetadata = (metadata) => {
  if (!metadata) return "";
  const parts = [];
  if (metadata.training_type) parts.push(`type=${metadata.training_type}`);
  if (metadata.window_size) parts.push(`window=${metadata.window_size}`);
  if (metadata.step) parts.push(`step=${metadata.step}`);
  if (metadata.threshold !== undefined) parts.push(`threshold=${metadata.threshold}`);
  if (metadata.filename) parts.push(`file=${metadata.filename}`);
  return parts.join(" | ");
};

const baseChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { enabled: true },
  },
  scales: {
    x: {
      ticks: { color: CHART_THEME.axis },
      grid: { color: CHART_THEME.grid },
    },
    y: {
      ticks: { color: CHART_THEME.axis },
      grid: { color: CHART_THEME.grid },
    },
  },
};

const lineChartOptions = {
  ...baseChartOptions,
  elements: {
    line: { tension: 0.3 },
    point: { radius: 2 },
  },
};

export default function App() {
  const edaSectionRef = useRef(null);
  const trainSectionRef = useRef(null);
  const predSectionRef = useRef(null);
  const edaPathRef = useRef(null);
  const trainTypeRef = useRef(null);
  const predWindowRef = useRef(null);
  const [apiBase, setApiBase] = useState(() => {
    const saved = window.localStorage.getItem(LS_KEY_BASE);
    return saved || getDefaultApiBase();
  });
  const [apiKey, setApiKey] = useState("");
  const [apiKeyLoaded, setApiKeyLoaded] = useState(false);
  const [keychainAvailable, setKeychainAvailable] = useState(false);
  const [status, setStatus] = useState(DEFAULT_STATUS);
  const [checking, setChecking] = useState(false);

  const [history, setHistory] = useState(emptyArray);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const [edaPath, setEdaPath] = useState("");
  const [edaFile, setEdaFile] = useState(null);
  const [edaResult, setEdaResult] = useState(null);
  const [edaError, setEdaError] = useState("");
  const [edaLoading, setEdaLoading] = useState(false);

  const [trainType, setTrainType] = useState("all");
  const [trainWindow, setTrainWindow] = useState(24);
  const [trainSave, setTrainSave] = useState(true);
  const [trainOutputDir, setTrainOutputDir] = useState("models/");
  const [trainPath, setTrainPath] = useState("");
  const [trainFile, setTrainFile] = useState(null);
  const [trainAsync, setTrainAsync] = useState(true);
  const [trainResult, setTrainResult] = useState(null);
  const [trainError, setTrainError] = useState("");
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainJob, setTrainJob] = useState(null);
  const [trainPolling, setTrainPolling] = useState(false);

  const [predWindow, setPredWindow] = useState(24);
  const [predStep, setPredStep] = useState(6);
  const [predThreshold, setPredThreshold] = useState(0.7);
  const [predPath, setPredPath] = useState("");
  const [predFile, setPredFile] = useState(null);
  const [predResult, setPredResult] = useState(null);
  const [predError, setPredError] = useState("");
  const [predLoading, setPredLoading] = useState(false);

  const base = useMemo(() => normalizeBase(apiBase), [apiBase]);
  const headers = useMemo(() => buildHeaders(apiKey), [apiKey]);

  const requestWithFallback = async (paths, options = {}) => {
    let lastResult = null;
    for (let index = 0; index < paths.length; index += 1) {
      const path = paths[index];
      const response = await fetch(buildUrl(base, path), options);
      const payload = await readJson(response);
      lastResult = { response, payload };
      if (response.status === 404 && index < paths.length - 1) {
        continue;
      }
      return lastResult;
    }
    return lastResult;
  };

  useEffect(() => {
    window.localStorage.setItem(LS_KEY_BASE, apiBase);
  }, [apiBase]);

  useEffect(() => {
    let alive = true;
    const loadApiKey = async () => {
      const secrets = window.rigvisionx?.secrets;
      if (secrets?.getApiKey) {
        let available = true;
        if (secrets.isKeychainAvailable) {
          try {
            available = await secrets.isKeychainAvailable();
          } catch {
            available = false;
          }
        }
        if (alive) setKeychainAvailable(available);
        if (available) {
          try {
            const stored = await secrets.getApiKey();
            if (alive && stored) setApiKey(stored);
          } catch {
            // Ignore keychain failures and continue without stored key.
          }
        } else {
          const saved = window.localStorage.getItem(LS_KEY_API);
          if (alive && saved) setApiKey(saved);
        }
      } else {
        const saved = window.localStorage.getItem(LS_KEY_API);
        if (alive && saved) setApiKey(saved);
      }
      if (alive) setApiKeyLoaded(true);
    };
    loadApiKey();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!apiKeyLoaded) return;
    if (keychainAvailable && window.rigvisionx?.secrets?.setApiKey) {
      if (apiKey) {
        window.rigvisionx.secrets.setApiKey(apiKey);
      } else if (window.rigvisionx?.secrets?.clearApiKey) {
        window.rigvisionx.secrets.clearApiKey();
      }
      return;
    }
    window.localStorage.setItem(LS_KEY_API, apiKey);
  }, [apiKey, apiKeyLoaded]);

  const checkHealth = async () => {
    if (!base) {
      setStatus({ api: "Missing URL", detail: "Set the API base URL." });
      return;
    }
    setChecking(true);
    try {
      const response = await fetch(buildUrl(base, "/health"), { headers });
      if (!response.ok) {
        const payload = await readJson(response);
        const detail = payload?.detail || response.statusText;
        setStatus({ api: "Error", detail });
        return;
      }
      const payload = await response.json();
      setStatus({ api: payload.status || "ok", detail: "Backend is reachable." });
    } catch (error) {
      setStatus({ api: "Offline", detail: error?.message || "Connection failed." });
    } finally {
      setChecking(false);
    }
  };

  const fetchHistory = async () => {
    if (!base) return;
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await fetch(buildUrl(base, HISTORY_ENDPOINT), { headers });
      const payload = await readJson(response);
      if (response.status === 404) {
        setHistory(emptyArray);
        setHistoryError("History endpoint is not available on this API.");
        return;
      }
      if (!response.ok) {
        throw new Error(payload?.detail || "History request failed.");
      }
      setHistory(payload?.records || emptyArray);
    } catch (error) {
      setHistoryError(error?.message || "Failed to load history.");
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    fetchHistory();
  }, [base, apiKey]);

  const submitEda = async (event) => {
    event.preventDefault();
    setEdaError("");
    setEdaResult(null);
    if (!base) {
      setEdaError("Set the API base URL first.");
      return;
    }
    if (!edaPath && !edaFile) {
      setEdaError("Provide a data path or upload a CSV file.");
      return;
    }
    setEdaLoading(true);
    try {
      const form = buildFormData({ data_path: edaPath }, edaFile);
      const response = await fetch(buildUrl(base, "/eda/summary"), {
        method: "POST",
        body: form,
        headers,
      });
      const payload = await readJson(response);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("EDA endpoint is unavailable on Docker API. Use Desktop backend on port 8081.");
        }
        throw new Error(payload?.detail || "EDA request failed.");
      }
      setEdaResult(payload);
      fetchHistory();
    } catch (error) {
      setEdaError(error?.message || "EDA failed.");
    } finally {
      setEdaLoading(false);
    }
  };

  const submitTraining = async (event) => {
    event.preventDefault();
    setTrainError("");
    setTrainResult(null);
    setTrainJob(null);
    if (!base) {
      setTrainError("Set the API base URL first.");
      return;
    }
    if (!trainPath && !trainFile) {
      setTrainError("Provide a data path or upload a CSV file.");
      return;
    }
    setTrainLoading(true);
    try {
      const form = buildFormData(
        {
          training_type: trainType,
          data_path: trainPath,
          save_models: trainSave,
          output_dir: trainOutputDir,
          window_size: trainWindow,
        },
        trainFile
      );
      const endpoint = trainAsync ? "/train/async" : "/train";
      const response = await fetch(buildUrl(base, endpoint), {
        method: "POST",
        body: form,
        headers,
      });
      const payload = await readJson(response);
      if (!response.ok) {
        throw new Error(payload?.detail || "Training failed.");
      }
      if (trainAsync) {
        setTrainJob(payload);
      } else {
        setTrainResult(payload);
      }
      fetchHistory();
    } catch (error) {
      setTrainError(error?.message || "Training failed.");
    } finally {
      setTrainLoading(false);
    }
  };

  const pollTraining = async (jobId) => {
    try {
      const response = await fetch(buildUrl(base, `/train/status/${jobId}`), {
        headers,
      });
      const payload = await readJson(response);
      if (!response.ok) {
        throw new Error(payload?.detail || "Failed to read job status.");
      }
      setTrainJob(payload);
      if (payload.status === "completed") {
        setTrainResult(payload.result);
      }
      if (payload.status === "failed") {
        setTrainError(payload.error || "Training failed.");
      }
    } catch (error) {
      setTrainError(error?.message || "Failed to read job status.");
    }
  };

  useEffect(() => {
    if (!trainJob?.job_id) return undefined;
    if (trainJob.status === "completed" || trainJob.status === "failed") {
      setTrainPolling(false);
      return undefined;
    }
    setTrainPolling(true);
    const timer = setTimeout(() => {
      pollTraining(trainJob.job_id);
    }, 2000);
    return () => clearTimeout(timer);
  }, [trainJob, base, apiKey]);

  const submitPrediction = async (event) => {
    event.preventDefault();
    setPredError("");
    setPredResult(null);
    if (!base) {
      setPredError("Set the API base URL first.");
      return;
    }
    if (!predPath && !predFile) {
      setPredError("Provide a data path or upload a CSV file.");
      return;
    }
    setPredLoading(true);
    try {
      const form = buildFormData(
        {
          window_size: predWindow,
          step: predStep,
          threshold: predThreshold,
          data_path: predPath,
        },
        predFile
      );
      const { response, payload } = await requestWithFallback(PREDICT_ENDPOINTS, {
        method: "POST",
        body: form,
        headers,
      });
      if (!response) {
        throw new Error("Prediction request failed.");
      }
      if (!response.ok) {
        throw new Error(payload?.detail || "Prediction failed.");
      }
      setPredResult(payload);
      fetchHistory();
    } catch (error) {
      setPredError(error?.message || "Prediction failed.");
    } finally {
      setPredLoading(false);
    }
  };

  const shellInfo = window.rigvisionx
    ? `${window.rigvisionx.platform} v${window.rigvisionx.version}`
    : "browser";

  const edaSummary = edaResult?.summary || null;
  const missingRows = edaSummary
    ? Object.entries(edaSummary.missing || {}).map(([name, count]) => ({
        name,
        count: Number(count) || 0,
      }))
    : emptyArray;
  const totalRows = edaResult?.dataset?.rows || 0;
  const missingSorted = [...missingRows].sort((a, b) => b.count - a.count).slice(0, 8);

  const describeRows = edaSummary
    ? Object.entries(edaSummary.describe || {}).slice(0, 8)
    : emptyArray;

  const correlationPairs = useMemo(() => {
    if (!edaSummary?.correlations) return emptyArray;
    const pairs = [];
    const entries = Object.entries(edaSummary.correlations);
    entries.forEach(([col, values]) => {
      if (!values || typeof values !== "object") return;
      Object.entries(values).forEach(([other, value]) => {
        if (col >= other) return;
        if (typeof value !== "number") return;
        pairs.push({ col, other, value, score: Math.abs(value) });
      });
    });
    return pairs.sort((a, b) => b.score - a.score).slice(0, 6);
  }, [edaSummary]);

  const missingChartData = useMemo(() => {
    return {
      labels: missingSorted.map((item) => item.name),
      datasets: [
        {
          label: "Missing",
          data: missingSorted.map((item) => item.count),
          backgroundColor: CHART_THEME.accentSoft,
          borderColor: CHART_THEME.accent,
          borderWidth: 1,
        },
      ],
    };
  }, [missingSorted]);

  const correlationChartData = useMemo(() => {
    return {
      labels: correlationPairs.map((item) => `${item.col}/${item.other}`),
      datasets: [
        {
          label: "Correlation",
          data: correlationPairs.map((item) => item.value),
          backgroundColor: CHART_THEME.warn,
        },
      ],
    };
  }, [correlationPairs]);

  const predictions = predResult?.predictions || emptyArray;
  const highRiskCount = predictions.filter((item) => item.high_risk).length;
  const averageRisk = predictions.length
    ? predictions.reduce((sum, item) => sum + (item.risk_score || 0), 0) / predictions.length
    : 0;
  const topPredictions = [...predictions]
    .sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0))
    .slice(0, 8);

  const riskSeries = useMemo(() => {
    return predictions.slice(0, 40).map((item, idx) => ({
      label: shortLabel(item.timestamp, idx),
      value: Number(item.risk_score) || 0,
    }));
  }, [predictions]);

  const riskChartData = useMemo(() => {
    return {
      labels: riskSeries.map((item) => item.label),
      datasets: [
        {
          label: "Risk score",
          data: riskSeries.map((item) => item.value),
          borderColor: CHART_THEME.accent,
          backgroundColor: CHART_THEME.accentSoft,
          fill: true,
        },
      ],
    };
  }, [riskSeries]);

  const trainingOutput = trainResult || trainJob?.result || null;

  const focusSection = (sectionRef, inputRef) => {
    const section = sectionRef?.current;
    if (section?.scrollIntoView) {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (inputRef?.current?.focus) {
      inputRef.current.focus({ preventScroll: true });
    }
  };

  return (
    <div className="app">
      <header className="header">
        <p className="eyebrow">TerraEnergy AI</p>
        <h1>Desktop Control Hub</h1>
        <p className="subhead">
          Local-first dashboard for models, datasets, and operational insight.
        </p>
      </header>

      <section className="panel">
        <div>
          <p className="label">Shell</p>
          <p className="value">{shellInfo}</p>
        </div>
        <div>
          <p className="label">Backend</p>
          <p className="value">{status.api}</p>
          <p className="hint">{status.detail}</p>
        </div>
      </section>

      <section className="panel form-panel">
        <div className="stack">
          <p className="label">API base URL</p>
          <input
            className="input"
            value={apiBase}
            onChange={(event) => setApiBase(event.target.value)}
            placeholder="http://127.0.0.1:8081 or http://127.0.0.1:8080"
          />
        </div>
        <div className="stack">
          <p className="label">API key or bearer token (optional)</p>
          <input
            className="input"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="x-api-key or JWT / Bearer <token>"
          />
          <p className="hint">Desktop backend uses x-api-key. Docker API uses JWT bearer token.</p>
        </div>
        <button className="button" onClick={checkHealth} disabled={checking}>
          {checking ? "Checking..." : "Check connection"}
        </button>
      </section>
      <section className="grid">
        <button
          type="button"
          className="card"
          onClick={() => focusSection(edaSectionRef, edaPathRef)}
        >
          <h2>EDA Workspace</h2>
          <p>Inspect datasets, correlations, and missing values.</p>
        </button>
        <button
          type="button"
          className="card"
          onClick={() => focusSection(trainSectionRef, trainTypeRef)}
        >
          <h2>Model Training</h2>
          <p>Run failure risk, RUL, and anomaly models via desktop or Docker API.</p>
        </button>
        <button
          type="button"
          className="card"
          onClick={() => focusSection(predSectionRef, predWindowRef)}
        >
          <h2>Predictions</h2>
          <p>Generate risk scores and export operational summaries.</p>
        </button>
      </section>

      <section className="grid grid-forms">
        <div className="card form-card" ref={edaSectionRef}>
          <h2>EDA Summary</h2>
          <p className="hint">Available on Desktop backend (`:8081`).</p>
          <form onSubmit={submitEda} className="form">
            <div className="field">
              <label className="label">CSV path (optional)</label>
              <input
                className="input"
                ref={edaPathRef}
                value={edaPath}
                onChange={(event) => setEdaPath(event.target.value)}
                placeholder="data/processed/training_data.csv"
              />
            </div>
            <div className="field">
              <label className="label">Upload CSV</label>
              <input
                className="input"
                type="file"
                accept=".csv"
                onChange={(event) => setEdaFile(event.target.files?.[0] || null)}
              />
            </div>
            <button className="button primary" type="submit" disabled={edaLoading}>
              {edaLoading ? "Running..." : "Run EDA"}
            </button>
          </form>
          {edaError && <p className="error">{edaError}</p>}
          {edaResult && (
            <div className="result-block">
              <div className="stat-grid">
                <div>
                  <p className="label">Rows</p>
                  <p className="value">{edaResult.dataset?.rows}</p>
                </div>
                <div>
                  <p className="label">Columns</p>
                  <p className="value">{edaResult.dataset?.columns}</p>
                </div>
                <div>
                  <p className="label">Dataset</p>
                  <p className="value">{edaResult.dataset?.path}</p>
                </div>
              </div>
              <h3 className="section-title">Missing values</h3>
              <div className="bar-list">
                {missingSorted.length === 0 && <p className="muted">No missing values.</p>}
                {missingSorted.map((item) => {
                  const pct = totalRows ? (item.count / totalRows) * 100 : 0;
                  return (
                    <div key={item.name} className="bar-row">
                      <span>{item.name}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="bar-value">{item.count}</span>
                    </div>
                  );
                })}
              </div>
              {missingSorted.length > 0 && (
                <div className="chart-panel">
                  <Bar data={missingChartData} options={baseChartOptions} />
                </div>
              )}
              <h3 className="section-title">Top correlations</h3>
              <div className="table">
                <div className="table-row table-head">
                  <span>Metric A</span>
                  <span>Metric B</span>
                  <span>Correlation</span>
                </div>
                {correlationPairs.length === 0 && (
                  <div className="table-row">
                    <span className="muted">Not enough numeric metrics.</span>
                  </div>
                )}
                {correlationPairs.map((item) => (
                  <div className="table-row" key={`${item.col}-${item.other}`}>
                    <span>{item.col}</span>
                    <span>{item.other}</span>
                    <span>{formatNumber(item.value)}</span>
                  </div>
                ))}
              </div>
              {correlationPairs.length > 0 && (
                <div className="chart-panel">
                  <Bar data={correlationChartData} options={baseChartOptions} />
                </div>
              )}
              <h3 className="section-title">Describe snapshot</h3>
              <div className="table">
                <div className="table-row table-head">
                  <span>Column</span>
                  <span>Mean</span>
                  <span>Std</span>
                  <span>Min</span>
                  <span>Max</span>
                </div>
                {describeRows.map(([col, stats]) => (
                  <div className="table-row" key={col}>
                    <span>{col}</span>
                    <span>{formatNumber(stats?.mean)}</span>
                    <span>{formatNumber(stats?.std)}</span>
                    <span>{formatNumber(stats?.min)}</span>
                    <span>{formatNumber(stats?.max)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card form-card" ref={trainSectionRef}>
          <h2>Training</h2>
          <form onSubmit={submitTraining} className="form">
            <div className="field">
              <label className="label">Training type</label>
              <select
                className="input"
                ref={trainTypeRef}
                value={trainType}
                onChange={(event) => setTrainType(event.target.value)}
              >
                <option value="all">all</option>
                <option value="failure_risk">failure_risk</option>
                <option value="rul">rul</option>
                <option value="anomaly">anomaly</option>
              </select>
            </div>
            <div className="field split">
              <div>
                <label className="label">Window size</label>
                <input
                  className="input"
                  type="number"
                  min="1"
                  value={trainWindow}
                  onChange={(event) => setTrainWindow(Number(event.target.value))}
                />
              </div>
              <div>
                <label className="label">Output dir</label>
                <input
                  className="input"
                  value={trainOutputDir}
                  onChange={(event) => setTrainOutputDir(event.target.value)}
                />
              </div>
            </div>
            <div className="field">
              <label className="label">CSV path (optional)</label>
              <input
                className="input"
                value={trainPath}
                onChange={(event) => setTrainPath(event.target.value)}
                placeholder="data/processed/training_data.csv"
              />
            </div>
            <div className="field">
              <label className="label">Upload CSV</label>
              <input
                className="input"
                type="file"
                accept=".csv"
                onChange={(event) => setTrainFile(event.target.files?.[0] || null)}
              />
            </div>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={trainSave}
                onChange={(event) => setTrainSave(event.target.checked)}
              />
              Save models on disk
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={trainAsync}
                onChange={(event) => setTrainAsync(event.target.checked)}
              />
              Run training in background
            </label>
            <button className="button primary" type="submit" disabled={trainLoading}>
              {trainLoading
                ? "Starting..."
                : trainAsync
                  ? "Start async training"
                  : "Start training"}
            </button>
          </form>
          {trainError && <p className="error">{trainError}</p>}
          {trainJob && (
            <div className="status-row">
              <span>Status:</span>
              <span className="value">
                {trainJob.status}
                {trainPolling ? " (polling)" : ""}
              </span>
              <span className="muted">Job {trainJob.job_id}</span>
            </div>
          )}
          {trainingOutput && (
            <pre className="result">{JSON.stringify(trainingOutput, null, 2)}</pre>
          )}
        </div>

        <div className="card form-card" ref={predSectionRef}>
          <h2>Failure Risk Prediction</h2>
          <form onSubmit={submitPrediction} className="form">
            <div className="field split">
              <div>
                <label className="label">Window size</label>
                <input
                  className="input"
                  type="number"
                  min="1"
                  ref={predWindowRef}
                  value={predWindow}
                  onChange={(event) => setPredWindow(Number(event.target.value))}
                />
              </div>
              <div>
                <label className="label">Step</label>
                <input
                  className="input"
                  type="number"
                  min="1"
                  value={predStep}
                  onChange={(event) => setPredStep(Number(event.target.value))}
                />
              </div>
              <div>
                <label className="label">Threshold</label>
                <input
                  className="input"
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={predThreshold}
                  onChange={(event) => setPredThreshold(Number(event.target.value))}
                />
              </div>
            </div>
            <div className="field">
              <label className="label">CSV path (optional)</label>
              <input
                className="input"
                value={predPath}
                onChange={(event) => setPredPath(event.target.value)}
                placeholder="data/processed/training_data.csv"
              />
            </div>
            <div className="field">
              <label className="label">Upload CSV</label>
              <input
                className="input"
                type="file"
                accept=".csv"
                onChange={(event) => setPredFile(event.target.files?.[0] || null)}
              />
            </div>
            <button className="button primary" type="submit" disabled={predLoading}>
              {predLoading ? "Running..." : "Run prediction"}
            </button>
          </form>
          {predError && <p className="error">{predError}</p>}
          {predResult && (
            <div className="result-block">
              <div className="stat-grid">
                <div>
                  <p className="label">Total points</p>
                  <p className="value">{predictions.length}</p>
                </div>
                <div>
                  <p className="label">High risk</p>
                  <p className="value">{highRiskCount}</p>
                </div>
                <div>
                  <p className="label">Avg risk</p>
                  <p className="value">{formatNumber(averageRisk)}</p>
                </div>
              </div>
              {riskSeries.length > 0 && (
                <div className="chart-panel">
                  <Line data={riskChartData} options={lineChartOptions} />
                </div>
              )}
              <h3 className="section-title">Top risk points</h3>
              <div className="table">
                <div className="table-row table-head">
                  <span>Timestamp</span>
                  <span>Rig</span>
                  <span>Score</span>
                </div>
                {topPredictions.map((item, idx) => (
                  <div className="table-row" key={`${item.timestamp}-${idx}`}>
                    <span>{item.timestamp || "-"}</span>
                    <span>{item.rig_id || "-"}</span>
                    <span>{formatNumber(item.risk_score)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="card activity-card">
        <div className="activity-header">
          <h2>Recent activity</h2>
          <button className="button" onClick={fetchHistory} disabled={historyLoading}>
            {historyLoading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
        {historyError && <p className="error">{historyError}</p>}
        <div className="table">
          <div className="table-row table-head">
            <span>Time</span>
            <span>Action</span>
            <span>Dataset</span>
            <span>Notes</span>
          </div>
          {history.length === 0 && !historyLoading && (
            <div className="table-row">
              <span className="muted">No history yet.</span>
            </div>
          )}
          {history.map((item, idx) => (
            <div className="table-row" key={`${item.timestamp}-${idx}`}>
              <span>{formatTimestamp(item.timestamp)}</span>
              <span>{item.action}</span>
              <span className="truncate">{item.data_source}</span>
              <span className="muted truncate">{summarizeMetadata(item.metadata)}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}




















