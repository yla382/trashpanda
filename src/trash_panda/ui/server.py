"""A small local web server that shows a trace in the browser. It uses only the standard library."""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .. import paths
from .load import TraceError, load_trace

STATIC = Path(__file__).with_name("static")
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def _page(address):
    """(file, content type) that serves `address`, or None. Only the files that sit directly in static/ are served, and only
    of the types above; `/` is the index page."""
    name = "index.html" if address == "/" else address[1:]
    path = STATIC / name
    if not name or "/" in name or "\\" in name or path.suffix not in _TYPES or not path.is_file():
        return None
    return path, _TYPES[path.suffix]


def _listing(trace_path, directory):
    """What the sidebar shows: the trace files of the folder, newest first, and the name of the one that is open."""
    files = []
    for path in paths.trace_files(directory) if directory is not None else []:
        try:
            info = path.stat()
        except OSError:
            continue
        files.append({"name": path.name, "size": info.st_size, "modified": info.st_mtime})
    current = Path(trace_path)
    inside = directory is not None and current.parent.resolve() == Path(directory).resolve()
    return {"current": current.name, "current_in_folder": inside, "folder": str(directory) if directory is not None else None,
            "files": files}


def make_server(trace_path, port=0, directory=None):
    """A server on this machine only, for the page files and the trace at `trace_path` (port 0 picks a free port).

    `directory` is the folder of trace files that the sidebar lists, and from which one can be chosen (a plain file name of a
    trace file that is there: nothing else is read). The trace is read again on every request, so a new recording shows after a
    refresh of the page."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parts = urlsplit(self.path)
            address = parts.path
            if address == "/trace.json":
                chosen = parse_qs(parts.query).get("name", [None])[0]
                path = trace_path
                if chosen is not None:
                    path = paths.safe_trace(directory, chosen) if directory is not None else None
                    if path is None:
                        self._send(404, json.dumps({"error": f"{chosen}: no such trace file"}).encode("utf-8"), "application/json")
                        return
                try:
                    body, status = json.dumps(load_trace(path)).encode("utf-8"), 200
                except (TraceError, OSError) as exc:
                    body, status = json.dumps({"error": f"{Path(path).name}: {exc}"}).encode("utf-8"), 422
                self._send(status, body, "application/json")
            elif address == "/api/traces":
                self._send(200, json.dumps(_listing(trace_path, directory)).encode("utf-8"), "application/json")
            elif _page(address) is not None:
                path, content_type = _page(address)
                self._send(200, path.read_bytes(), content_type)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, *args):         # no line per request on the terminal
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(trace_path, port=0, open_browser=True, echo=print, directory=None):
    """Serve the viewer until Ctrl+C."""
    server = make_server(trace_path, port, directory)
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
