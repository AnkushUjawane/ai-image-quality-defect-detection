# Aperture — Frontend (React + Vite)

Dependency-light React SPA for the Image Quality & Defect Detection
inspection console. No CSS framework, no state library — plain React state
and hooks.

## Structure

```
src/
├── main.jsx               entry point, mounts <App/>
├── App.jsx                top-level layout + state (selected file, result, history)
├── config.js               allowed file types, size limit, label→CSS-class map
├── api/
│   └── client.js           fetch wrapper: analyzeImage, listAnalyses, getAnalysis, checkHealth
├── hooks/
│   └── useHealthCheck.js   polls /api/health every 15s
├── components/
│   ├── TopBar.jsx
│   ├── Dropzone.jsx        upload, drag & drop, preview, "Run inspection" button
│   ├── ResultsPanel.jsx    score ring, label, issues list, stats grid
│   ├── IssueCard.jsx
│   ├── StatsGrid.jsx
│   └── HistoryList.jsx
└── styles/
    └── index.css
```

## Setup

```bash
npm install
cp .env.example .env   # edit VITE_API_BASE_URL if the backend isn't on localhost:8000
npm run dev             # http://localhost:5173
```

## Build

```bash
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
```

`VITE_API_BASE_URL` is read at **build time** (Vite inlines `import.meta.env.*`
into the bundle) — set it before running `npm run build`, or pass it as a
Docker build arg (see the repo-root README / `docker-compose.yml`).