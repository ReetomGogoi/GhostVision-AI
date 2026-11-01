# GhostVision-AI React workstation

The website is a React 18, TypeScript, and Vite client for the FastAPI service
in `ghostvision/api.py`. Model inference remains server-side through the
existing `GhostVisionApplication`; the browser does not load checkpoints or
generate predictions.

## Result handling

- RGB requests use `POST /predict/rgb`. The returned base64 PNG mask and
  overlay are rendered directly, while the selected local image remains the
  original and comparison source.
- Thermal requests use `POST /predict/thermal`. The returned animal count,
  XYXY boxes, and confidence scores are rendered over the selected local image.
- Download buttons create PNG visualizations from the actual response in the
  browser. No prediction values are embedded in production frontend code.
- `GET /health` controls checkpoint availability states without loading models.

RGB segmentation masks and thermal detection boxes remain distinct result
types throughout the interface.

## Configure and start the backend

Checkpoint paths remain external to the repository:

```bash
export GHOSTVISION_RGB_BASELINE=/absolute/path/to/best_resumed_epoch_19.pth
export GHOSTVISION_RGB_AUGMENTED=/absolute/path/to/best_resumed_epoch_07.pth
export GHOSTVISION_THERMAL_CHECKPOINT=/absolute/path/to/best.pt
export GHOSTVISION_THERMAL_CONFIDENCE=0.25
export GHOSTVISION_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Install the project dependencies and start FastAPI from the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn ghostvision.api:app --host 127.0.0.1 --port 8000
```

Check configuration at <http://127.0.0.1:8000/health>. Full endpoint schemas
and examples are documented in [`api.md`](api.md).

## Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/health` and `/predict` to the local
backend. For a backend on another host, copy `.env.example` to `.env.local` and
set `VITE_API_BASE_URL` to its origin.

## Validation

Run the backend and existing Streamlit tests:

```bash
python -B -m unittest discover -s tests -v
```

Run the frontend API-contract tests, TypeScript validation, and production
build:

```bash
cd frontend
npm test
npm run typecheck
npm run build
```

The frontend tests mock HTTP responses only. They verify the committed endpoint
paths, multipart field names, health schema, both modality result shapes, and
safe error handling without loading model weights.

## Design attribution

The workstation layout and interaction language were informed by the
Apache-2.0 licensed
[`gammahazard/locate-anything`](https://github.com/gammahazard/locate-anything)
interface. GhostVision uses its own modality controls, API, result contracts,
visualizations, and styling. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
