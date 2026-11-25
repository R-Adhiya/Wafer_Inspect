#!/usr/bin/env python3
"""End-to-end smoke test script for WaferInspect using standard library only.

Verifies:
  1. GET /health -> 200, status "ok"
  2. POST /samples/<name> -> 200, returns upload_id and inspection_id
  3. POST /inspection/run -> 200, pipeline completes
  4. GET /inspection/{id} -> 200, defect_count > 0, pattern present, severity present
  5. GET /inspection/history -> 200, contains the inspection in history list
  6. GET /inspection/{id}/image/final_overlay -> 200, non-empty image bytes
  7. DELETE /inspection/{id} -> 200, deleted: true

Exits non-zero with a clear error on any failure.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _request(
    url: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Tuple[int, bytes, Dict[str, str]]:
    """Execute an HTTP request using urllib and return (status_code, body_bytes, headers)."""
    req_headers = headers.copy() if headers else {}
    body_bytes: Optional[bytes] = None

    if data is not None:
        body_bytes = json.dumps(data).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(
        url=url,
        data=body_bytes,
        headers=req_headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, resp_body, resp_headers
    except urllib.error.HTTPError as err:
        err_body = err.read()
        err_headers = {k.lower(): v for k, v in err.headers.items()}
        return err.code, err_body, err_headers
    except Exception as exc:
        raise RuntimeError(f"Network error requesting {method} {url}: {exc}") from exc


def run_smoke_test(base_url: str, sample_name: str = "edge_ring") -> bool:
    """Run full end-to-end smoke test suite against target base URL."""
    base_url = base_url.rstrip("/")
    print(f"[*] Starting WaferInspect smoke test against {base_url} (sample: {sample_name})")

    # 1. Health check
    health_url = f"{base_url}/health"
    print(f"  [1/7] GET {health_url} ...")
    status, body, _ = _request(health_url)
    if status != 200:
        print(f"FAIL: GET /health returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    try:
        health_data = json.loads(body.decode("utf-8"))
        if health_data.get("status") not in ("ok", "healthy"):
            print(f"FAIL: Expected status 'ok' or 'healthy', got: {health_data}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"FAIL: Malformed JSON from /health: {exc}", file=sys.stderr)
        return False
    print("        -> OK (health check passed)")

    # 2. POST /samples/<name>
    sample_url = f"{base_url}/samples/{sample_name}"
    print(f"  [2/7] POST {sample_url} ...")
    status, body, _ = _request(sample_url, method="POST", data={})
    if status != 200:
        print(f"FAIL: POST /samples/{sample_name} returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    try:
        sample_data = json.loads(body.decode("utf-8"))
        inspection_id = sample_data.get("inspection_id")
        upload_id = sample_data.get("upload_id")
        if not inspection_id or not upload_id:
            print(f"FAIL: Missing inspection_id or upload_id in sample response: {sample_data}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"FAIL: Malformed JSON from /samples: {exc}", file=sys.stderr)
        return False
    print(f"        -> OK (created inspection_id={inspection_id}, upload_id={upload_id})")

    # 3. POST /inspection/run
    run_url = f"{base_url}/inspection/run"
    print(f"  [3/7] POST {run_url} ...")
    run_payload = {
        "inspection_id": inspection_id,
        "upload_id": upload_id,
    }
    status, body, _ = _request(run_url, method="POST", data=run_payload, timeout=60.0)
    if status != 200:
        print(f"FAIL: POST /inspection/run returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    try:
        run_data = json.loads(body.decode("utf-8"))
        if run_data.get("status") != "completed":
            print(f"FAIL: Inspection run status is '{run_data.get('status')}', expected 'completed'", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"FAIL: Malformed JSON from /inspection/run: {exc}", file=sys.stderr)
        return False
    print("        -> OK (inspection completed successfully)")

    # 4. GET /inspection/{id}
    report_url = f"{base_url}/inspection/{inspection_id}"
    print(f"  [4/7] GET {report_url} ...")
    status, body, _ = _request(report_url)
    if status != 200:
        print(f"FAIL: GET /inspection/{inspection_id} returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    try:
        report_data = json.loads(body.decode("utf-8"))
        summary = report_data.get("summary") or {}
        defect_count = summary.get("defect_count", 0)
        pattern = report_data.get("pattern") or summary.get("pattern")
        severity = summary.get("severity")

        if defect_count <= 0:
            print(f"FAIL: Expected defect_count > 0, got {defect_count}", file=sys.stderr)
            return False
        if not pattern:
            print("FAIL: Pattern analysis result missing from inspection report", file=sys.stderr)
            return False
        if not severity:
            print("FAIL: Severity assessment missing from inspection report", file=sys.stderr)
            return False
        print(f"        -> OK (defects={defect_count}, severity={severity}, pattern={pattern.get('pattern_type') or pattern.get('type')})")
    except Exception as exc:
        print(f"FAIL: Failed inspecting report payload: {exc}", file=sys.stderr)
        return False

    # 5. GET /inspection/history
    history_url = f"{base_url}/inspection/history"
    print(f"  [5/7] GET {history_url} ...")
    status, body, _ = _request(history_url)
    if status != 200:
        print(f"FAIL: GET /inspection/history returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    try:
        history_data = json.loads(body.decode("utf-8"))
        items = history_data.get("items", [])
        found = any(it.get("id") == inspection_id or it.get("upload_id") == upload_id for it in items)
        if not found:
            print(f"FAIL: Inspection id {inspection_id} not found in history items ({len(items)} items)", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"FAIL: Malformed JSON from history: {exc}", file=sys.stderr)
        return False
    print("        -> OK (inspection found in history)")

    # 6. GET /inspection/{id}/image/final_overlay
    image_url = f"{base_url}/inspection/{inspection_id}/image/final_overlay"
    print(f"  [6/7] GET {image_url} ...")
    status, body, headers = _request(image_url)
    if status != 200:
        print(f"FAIL: GET {image_url} returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    if len(body) == 0:
        print("FAIL: Received 0 bytes for final_overlay image", file=sys.stderr)
        return False
    print(f"        -> OK (retrieved {len(body)} bytes, content-type: {headers.get('content-type')})")

    # 7. DELETE /inspection/{id}
    delete_url = f"{base_url}/inspection/{inspection_id}"
    print(f"  [7/7] DELETE {delete_url} ...")
    status, body, _ = _request(delete_url, method="DELETE")
    if status != 200:
        print(f"FAIL: DELETE {delete_url} returned status {status}: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False
    try:
        del_data = json.loads(body.decode("utf-8"))
        if not del_data.get("deleted"):
            print(f"FAIL: Expected deleted: true, got {del_data}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"FAIL: Malformed JSON from delete: {exc}", file=sys.stderr)
        return False
    print("        -> OK (inspection deleted successfully)")

    print("[SUCCESS] All 7 smoke test checks passed!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="WaferInspect Smoke Test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the WaferInspect backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--sample",
        default="edge_ring",
        help="Sample wafer name to test (default: edge_ring)",
    )
    args = parser.parse_args()

    success = run_smoke_test(base_url=args.base_url, sample_name=args.sample)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
