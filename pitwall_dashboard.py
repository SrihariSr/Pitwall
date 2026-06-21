"""
Pitwall dashboard server.

Serves the dashboard HTML and a JSON endpoint that reads
decisions/decisions.jsonl to expose the live state of the Orchestrator.

Run from the project root:
    uv run python pitwall_dashboard.py

Then open http://localhost:8765 in a browser. Keep watch_race.py running
in another terminal — every new decision appended to decisions.jsonl will
appear in the dashboard on its next poll.
"""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from race_config import DISPLAY_NAME, TOTAL_LAPS

_DECISIONS_PATH = Path("decisions/decisions.jsonl")
_HTML_PATH = Path(__file__).parent / "pitwall_dashboard.html"
_PORT = 8765

def _load_state() -> dict:
    """Read decisions.jsonl and return a dashboard-friendly snapshot.

    Returns a dict with three keys:
      - current: the most recent full decision record (or None)
      - history: a compact list of all decisions for the timeline
      - race: derived metadata (event, lap, driver, position)
    """
    if not _DECISIONS_PATH.exists():
        return {"current": None, "history": [], "race": None}

    records = []
    with _DECISIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines — e.g. partial write captured mid-poll.
                continue

    if not records:
        return {"current": None, "history": [], "race": None}

    current = records[-1]

    history = [
        {
            "lap": r["lap"],
            "call": r["decision"]["call"],
            "confidence": r["decision"]["confidence"],
            "trigger": r["decision"].get("trigger", "scheduled"),
            "primary_reason": r["decision"]["primary_reason"],
        }
        for r in records
    ]

    gap = current["subagents"]["gap"]
    race = {
        "event": DISPLAY_NAME,
        "lap": current["lap"],
        "lap_total": TOTAL_LAPS,
        "driver": current["driver"],
        "position": f"P{gap['focal_position']}",
    }

    return {"current": current, "history": history, "race": race}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/state":
            self._send_json(_load_state())
            return
        if self.path in ("/", "/index.html"):
            self._send_html()
            return
        self.send_response(404)
        self.end_headers()

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        if not _HTML_PATH.exists():
            self.send_response(500)
            self.end_headers()
            self.wfile.write(
                b"pitwall_dashboard.html not found next to pitwall_dashboard.py"
            )
            return
        body = _HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Silence stdout — polling every 2s would otherwise flood the terminal.
        pass


def main() -> None:
    server = HTTPServer(("127.0.0.1", _PORT), Handler)
    print(f"Pitwall dashboard running at http://localhost:{_PORT}")
    print(f"Reading decisions from {_DECISIONS_PATH.resolve()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
        server.shutdown()


if __name__ == "__main__":
    main()
