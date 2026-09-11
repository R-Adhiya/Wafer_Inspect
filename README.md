# WaferInspect — Automated Wafer Defect Inspection & Measurement

WaferInspect is a software prototype demonstrating automated defect detection, measurement, and spatial pattern diagnosis using wafer-like grayscale imagery. It provides an end-to-end discover → filter → diagnose workflow designed to distinguish random noise from systematic defect patterns.

> **Disclaimer**: WaferInspect is an educational prototype and is not intended for industrial fab-grade inspection or production QC tools.

---

## Current Phase: Phase 1 — Project Foundation

Phase 1 establishes the foundational monorepo architecture, environment setup, database foundation, and live REST communication between the React frontend and FastAPI backend.

```
React + TypeScript (Vite + Tailwind CSS)
          │
          ▼ (REST / HTTP)
FastAPI Backend (Uvicorn + Pydantic)
          │
          ▼
SQLAlchemy + SQLite Database Foundation
```

---

## Current Status

- [x] Monorepo structure initialized with module separation (`cv`, `ml`, `db`, `services`, `routes`).
- [x] FastAPI backend application with CORS, logging, and environment configuration.
- [x] Modular Health Check API (`GET /health`).
- [x] SQLAlchemy database foundation (SQLite initialization, engine, session management, and Base models).
- [x] React + Vite + TypeScript + Tailwind CSS frontend application.
- [x] Centralized API service layer (`frontend/src/services/api.ts`).
- [x] Real-time Backend Connection Status indicator with graceful offline error handling.
- [ ] Image upload & validation *(Phase 2)*
- [ ] Classical CV preprocessing & detection *(Phases 3 & 4)*
- [ ] Defect segmentation & measurement *(Phases 5 & 6)*
- [ ] Spatial pattern diagnosis algorithm *(Phase 6B)*
- [ ] PyTorch classifier integration *(Phase 10)*

---

## Quickstart Guide

### 1. Running the Backend

```bash
# Navigate to project root
cd backend

# Create and activate Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate

# Install Phase 1 dependencies
pip install -r requirements.txt

# Start FastAPI development server
uvicorn app.main:app --reload --port 8000
```

The backend server will start at `http://localhost:8000`. You can inspect the API endpoints and interactive Swagger documentation at `http://localhost:8000/docs`.

### 2. Running the Frontend

```bash
# Navigate to frontend folder
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

The frontend application will start at `http://localhost:5173`. Open your browser to view the WaferInspect Dashboard and verify backend connection status.

---

## API Endpoints

### Health Check

- **Endpoint**: `GET /health`
- **Description**: Returns operational status, application name, and version.
- **Response**:
```json
{
  "status": "healthy",
  "service": "WaferInspect API",
  "version": "0.1.0"
}
```
