import json
import threading
import urllib.error
import urllib.request

import pytest

from trash_panda.ui.load import DEMO
from trash_panda.ui.server import make_server


@pytest.fixture
def serving():
    """Starts a server for a trace (the demo by default): returns a function that fetches an address from it, and the server."""
    started = []

    def start(path=DEMO):
        server = make_server(path)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def get(address):
            try:
                with urllib.request.urlopen(base + address, timeout=5) as r:
                    return r.status, r.headers, r.read()
            except urllib.error.HTTPError as err:
                return err.code, err.headers, err.read()

        return get, server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def test_the_server_listens_on_this_machine_only(serving):
    _, server = serving()
    assert server.server_address[0] == "127.0.0.1"


def test_the_page_is_served(serving):
    get, _ = serving()
    status, headers, body = get("/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b"trash-panda trace" in body


def test_the_script_and_the_style_are_served(serving):
    get, _ = serving()
    status, headers, _ = get("/replay.js")
    assert status == 200 and "javascript" in headers["Content-Type"]
    status, headers, _ = get("/style.css")
    assert status == 200 and headers["Content-Type"].startswith("text/css")


def test_the_trace_is_served_as_json(serving):
    get, _ = serving()
    status, headers, body = get("/trace.json")
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(body) == json.loads(DEMO.read_text(encoding="utf-8"))


def test_a_query_string_is_ignored(serving):
    get, _ = serving()
    assert get("/?reload=1")[0] == 200
    assert get("/trace.json?x=1")[0] == 200


@pytest.mark.parametrize("address", ["/nope", "/dummy_trace.json", "/../load.py", "/%2e%2e/load.py", "/static/index.html"])
def test_nothing_else_is_served(serving, address):
    get, _ = serving()
    assert get(address)[0] == 404


def test_pages_are_not_cached_so_a_new_recording_shows_after_a_refresh(serving):
    get, _ = serving()
    assert get("/trace.json")[1]["Cache-Control"] == "no-store"


def test_the_trace_is_read_again_on_each_request(serving, tmp_path):
    path = tmp_path / "t.json"
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    path.write_text(json.dumps(data), encoding="utf-8")
    get, _ = serving(path)
    assert json.loads(get("/trace.json")[2])["trashPanda"]["note"] == data["trashPanda"]["note"]
    data["trashPanda"]["note"] = "second recording"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert json.loads(get("/trace.json")[2])["trashPanda"]["note"] == "second recording"


def test_a_bad_trace_is_an_error_message_not_a_crash(serving, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{nope", encoding="utf-8")
    get, _ = serving(path)
    status, _, body = get("/trace.json")
    assert status == 422
    assert "broken.json" in json.loads(body)["error"]
    assert get("/")[0] == 200                           # the page itself still loads, and shows the message
