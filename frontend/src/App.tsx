import { useEffect, useMemo, useRef, useState } from "react";

import {
  getHealth,
  predict,
  type HealthResponse,
  type Modality,
  type Prediction,
  type RgbModel,
} from "./api";
import { AlertIcon, ApertureIcon, CheckIcon, HeatIcon, LayersIcon } from "./components/Icons";
import ResultWorkspace from "./components/ResultWorkspace";
import UploadDropzone from "./components/UploadDropzone";

export default function App() {
  const [modality, setModality] = useState<Modality>("rgb");
  const [rgbModel, setRgbModel] = useState<RgbModel>("augmented");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<Prediction | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then(setHealth)
      .catch(() => setHealthError(true));
    return () => controller.abort();
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
    abortRef.current?.abort();
  }, [preview]);

  const checkpointState = useMemo(() => {
    if (!health) return null;
    return modality === "rgb"
      ? health.modalities.rgb.models[rgbModel]
      : health.modalities.thermal;
  }, [health, modality, rgbModel]);

  function selectFile(selected: File) {
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setPredictionResult(null);
    setError(null);
  }

  function switchModality(next: Modality) {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setModality(next);
    setPredictionResult(null);
    setError(null);
  }

  async function runInference() {
    if (!file) {
      setError("Choose an image before running inference.");
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      setPredictionResult(await predict(file, modality, rgbModel, controller.signal));
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Inference failed unexpectedly.");
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        abortRef.current = null;
      }
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="GhostVision home">
          <span className="brand-mark"><ApertureIcon /></span>
          <span><strong>GHOSTVISION</strong><small>Wildlife vision workstation</small></span>
        </a>
        <div className="topbar-status">
          <span className={`status-dot ${healthError ? "error" : health ? "online" : ""}`} />
          {healthError ? "API unavailable" : health ? "Inference API online" : "Checking backend"}
        </div>
      </header>

      <main id="top" className="workstation">
        <aside className="control-panel">
          <div className="panel-heading">
            <span className="step-number">01</span>
            <div><span className="eyebrow">Analysis mode</span><h1>Select a vision pipeline</h1></div>
          </div>

          <div className="mode-selector">
            <button type="button" className={modality === "rgb" ? "active rgb" : ""} onClick={() => switchModality("rgb")}>
              <LayersIcon /><span><strong>RGB Segmentation</strong><small>Reveal camouflaged contours</small></span>
            </button>
            <button type="button" className={modality === "thermal" ? "active thermal" : ""} onClick={() => switchModality("thermal")}>
              <HeatIcon /><span><strong>Thermal Detection</strong><small>Locate wildlife heat signatures</small></span>
            </button>
          </div>

          {modality === "rgb" && (
            <fieldset className="model-selector">
              <legend>RGB checkpoint</legend>
              {(["augmented", "baseline"] as RgbModel[]).map((model) => (
                <label key={model} className={rgbModel === model ? "selected" : ""}>
                  <input type="radio" name="rgb-model" value={model} checked={rgbModel === model} onChange={() => { setRgbModel(model); setPredictionResult(null); }} />
                  <span><strong>{model}</strong><small>{model === "augmented" ? "Epoch 7 · selected best" : "Epoch 19 · selected best"}</small></span>
                  <span className="radio-mark" />
                </label>
              ))}
            </fieldset>
          )}

          <div className="section-divider" />
          <div className="panel-heading compact">
            <span className="step-number">02</span>
            <div><span className="eyebrow">Source image</span><h2>Load a field capture</h2></div>
          </div>
          <UploadDropzone file={file} previewUrl={preview} disabled={loading} onFile={selectFile} onError={setError} />

          {checkpointState && !checkpointState.checkpoint_available && (
            <div className="checkpoint-warning"><AlertIcon /><div><strong>Checkpoint unavailable</strong><span>Configure the server path for this model before inference.</span></div></div>
          )}

          {error && <div className="error-message" role="alert"><AlertIcon /><span>{error}</span></div>}

          <button className={`run-button ${modality}`} type="button" onClick={runInference} disabled={!file || loading || checkpointState?.checkpoint_available === false}>
            {loading ? <><span className="spinner" />Processing image</> : <>Run {modality === "rgb" ? "segmentation" : "detection"}<span>→</span></>}
          </button>
          <p className="privacy-note">Your image is processed by the locally configured server. Predictions are never hardcoded in this interface.</p>
        </aside>

        <section className="visualization-panel" aria-live="polite">
          <div className="visualization-header">
            <div><span className="eyebrow">Visualization workspace</span><h2>{predictionResult ? predictionResult.modality === "rgb" ? `${predictionResult.model_name} RGB model` : "YOLO11n thermal detector" : modality === "rgb" ? "Camouflage segmentation" : "Thermal wildlife detection"}</h2></div>
            <div className="pipeline-badge"><span className={modality} />{modality === "rgb" ? "U-Net · ResNet34" : "YOLO11n"}</div>
          </div>

          {loading ? (
            <div className="workspace-empty loading-state">
              <div className="scanner"><span /></div>
              <strong>{modality === "rgb" ? "Resolving camouflage boundaries" : "Scanning thermal signatures"}</strong>
              <p>Inference is running on the server with the configured real checkpoint.</p>
            </div>
          ) : predictionResult ? (
            <ResultWorkspace
              key={preview ?? predictionResult.modality}
              prediction={predictionResult}
              originalUrl={preview ?? ""}
              originalName={file?.name ?? "wildlife-image"}
              onError={setError}
            />
          ) : preview ? (
            <div className="image-stage preview-stage">
              <img src={preview} alt="Selected wildlife image awaiting inference" />
              <div className="preview-caption"><span>Ready for analysis</span><small>{file?.name}</small></div>
            </div>
          ) : (
            <div className="workspace-empty">
              <div className="reticle"><span /><i /></div>
              <strong>Awaiting field image</strong>
              <p>Upload an RGB or thermal capture to begin server-side inference.</p>
              <div className="empty-features"><span><CheckIcon />Validated upload</span><span><CheckIcon />Real model output</span><span><CheckIcon />Downloadable evidence</span></div>
            </div>
          )}
        </section>
      </main>

      <footer>
        <span>GhostVision-AI · RGB segmentation + thermal detection</span>
        <span>Interface design informed by <a href="https://github.com/gammahazard/locate-anything" target="_blank" rel="noreferrer">locate-anything</a> · Apache-2.0</span>
      </footer>
    </div>
  );
}
