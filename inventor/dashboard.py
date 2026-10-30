"""Invention Command Center & Escalation Review UI for Inventor-Argon (http://127.0.0.1:8050). The UI lives in dashboard.html."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import unquote

_HTML_PATH = Path(__file__).with_name("dashboard.html")
_STATIC_ROOT = Path(__file__).with_name("static").resolve()
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".cif": "chemical/x-cif",
}


def _load_dashboard_html() -> str:
    """Read the UI each request so edits to dashboard.html show up on refresh."""
    try:
        return _HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        return "<!DOCTYPE html><title>Inventor-Argon</title><p>dashboard.html is missing.</p>"


def start_dashboard_server(
    status_provider: Callable[[], Dict[str, Any]],
    host: str = "127.0.0.1",
    port: int = 8050,
    review_handler: Optional[Callable[[Dict[str, Any]], bool]] = None,
    escalate_handler: Optional[Callable[[str], bool]] = None,
    wave_runner: Optional[Callable[[], Any]] = None,
    toggle_continuous_handler: Optional[Callable[[], bool]] = None,
    simulate_handler: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Optional[ThreadingHTTPServer]:
    """Start background HTTP server serving the interactive Invention Command Center at http://127.0.0.1:8050."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                if self.path.startswith("/api/status"):
                    payload = json.dumps(status_provider()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path.startswith("/static/"):
                    self._serve_static()
                else:
                    payload = _load_dashboard_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

        def _serve_static(self) -> None:
            """Serve files under inventor/static, with Range support so <video> can seek."""
            rel = unquote(self.path.split("?", 1)[0][len("/static/"):])
            target = (_STATIC_ROOT / rel).resolve()
            if _STATIC_ROOT not in target.parents or not target.is_file():
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            size = target.stat().st_size
            start, end = 0, size - 1
            status = 200
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                first, _, last = rng[6:].partition("-")
                try:
                    start = int(first) if first else max(size - int(last), 0)
                    end = int(last) if first and last else size - 1
                    end = min(end, size - 1)
                    status = 206
                except ValueError:
                    start, end, status = 0, size - 1, 200
            length = max(end - start + 1, 0)
            self.send_response(status)
            self.send_header("Content-Type", _CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-cache")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(target, "rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = fh.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                body = json.loads(raw) if raw else {}
            except Exception:
                body = {}

            ok = False
            extra: Dict[str, Any] = {}
            if self.path.startswith("/api/review") and review_handler:
                ok = review_handler(body)
            elif self.path.startswith("/api/escalate") and escalate_handler:
                ok = escalate_handler(str(body.get("item_id", "")))
            elif self.path.startswith("/api/run_wave") and wave_runner:
                threading.Thread(target=wave_runner, daemon=True).start()
                ok = True
            elif self.path.startswith("/api/toggle_continuous") and toggle_continuous_handler:
                ok = toggle_continuous_handler()
            elif self.path.startswith("/api/simulate") and simulate_handler:
                sim_res = simulate_handler(str(body.get("item_id", "")))
                ok = bool(sim_res)
                extra["simulation"] = sim_res

            resp = json.dumps({"ok": ok, **extra}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server
    except OSError:
        return None
