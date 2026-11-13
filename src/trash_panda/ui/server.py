"""A small local web server that shows a trace in the browser. It uses only the standard library."""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .load import TraceError, load_trace

STATIC = Path(__file__).with_name("static")
_PAGES = {                                    # the only files that are served, by their address
    "/": ("index.html", "text/html; charset=utf-8"),
    "/replay.js": ("replay.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def make_server(trace_path, port=0):
    """A server on this machine only, for the page files and the trace at `trace_path` (port 0 picks a free port).

    The trace is read again on every request, so a new recording shows after a refresh of the page."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            address = self.path.split("?", 1)[0]
            if address == "/trace.json":
                try:
                    body, status = json.dumps(load_trace(trace_path)).encode("utf-8"), 200
                except (TraceError, OSError) as exc:
                    body, status = json.dumps({"error": f"{Path(trace_path).name}: {exc}"}).encode("utf-8"), 422
                self._send(status, body, "application/json")
            elif address in _PAGES:
                name, content_type = _PAGES[address]
                self._send(200, (STATIC / name).read_bytes(), content_type)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, *args):         # no line per request on the terminal
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(trace_path, port=0, open_browser=True, echo=print):
    """Serve the viewer until Ctrl+C."""
    server = make_server(trace_path, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    echo(f"Showing {Path(trace_path).name} at {url}   (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        echo("Stopped.")
    finally:
        server.server_close()
