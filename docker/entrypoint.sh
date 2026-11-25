#!/bin/sh
set -e

echo "[WaferInspect] Running database migrations..."
alembic upgrade head

echo "[WaferInspect] Starting backend API server on 0.0.0.0:8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
