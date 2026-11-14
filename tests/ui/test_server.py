import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from trash_panda.ui.server import STATIC, make_server

from .helpers import SIMPLE


@pytest.fixture
def serving():
    """Starts a server for a trace (the simple dummy trace by default): returns a function that fetches an address from it, and the server."""
    started = []

    def start(path=SIMPLE, directory=None):
        server = make_server(path, directory=directory)
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
    status, headers, _ = get("/app.js")
    assert status == 200 and "javascript" in headers["Content-Type"]
    status, headers, _ = get("/style.css")
    assert status == 200 and headers["Content-Type"].startswith("text/css")


def test_every_file_of_the_page_is_served_with_its_type(serving):
    get, _ = serving()
    types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
    for file in STATIC.iterdir():
        if file.suffix not in types:
            continue                                    # package.json is there for Node, which runs the tests of the page's code
        status, headers, body = get("/" if file.name == "index.html" else "/" + file.name)
        assert status == 200 and headers["Content-Type"].startswith(types[file.suffix]), file.name
        assert body == file.read_bytes()


def test_the_files_that_are_not_part_of_the_page_are_not_served(serving):
    get, _ = serving()
    assert get("/package.json")[0] == 404


def test_the_page_only_imports_files_that_exist_and_are_served(serving):
    get, _ = serving()
    html = get("/")[2].decode("utf-8")
    pending, seen = re.findall(r'(?:src|href)="/([\w.]+)"', html), set()
    while pending:                                      # the page, then what each module imports, and so on
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        assert (STATIC / name).is_file(), name
        status, _, body = get("/" + name)
        assert status == 200, name
        if name.endswith(".js"):
            pending += re.findall(r'from "\./([\w.]+)"', body.decode("utf-8"))
    assert {"app.js", "style.css", "svg.js", "data.js"} <= seen


def test_the_trace_is_served_as_json(serving):
    get, _ = serving()
    status, headers, body = get("/trace.json")
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(body) == json.loads(SIMPLE.read_text(encoding="utf-8"))


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
    data = json.loads(SIMPLE.read_text(encoding="utf-8"))
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


# -- the sidebar: the trace files of a folder, and choosing one of them

@pytest.fixture
def folder(tmp_path):
    """A folder with three recordings, the middle one the simple dummy trace and the others with a note of their own."""
    names = ("20261112-100000-000.json", "20261113-080000-100.json", "20261114-093012-482.json")
    for i, name in enumerate(names):
        data = json.loads(SIMPLE.read_text(encoding="utf-8"))
        data["trashPanda"]["note"] = f"recording {i}"
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_without_a_folder_the_list_is_empty_and_has_no_folder(serving):
    get, _ = serving()
    status, headers, body = get("/api/traces")
    listing = json.loads(body)
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert listing["folder"] is None and listing["files"] == []


def test_the_list_has_the_files_of_the_folder_newest_first_and_the_open_one(serving, folder):
    get, _ = serving(folder / "20261113-080000-100.json", folder)
    listing = json.loads(get("/api/traces")[2])
    assert [f["name"] for f in listing["files"]] == ["20261114-093012-482.json", "20261113-080000-100.json", "20261112-100000-000.json"]
    assert all(f["size"] > 0 and f["modified"] > 0 for f in listing["files"])
    assert listing["current"] == "20261113-080000-100.json" and listing["current_in_folder"] is True
    assert listing["folder"] == str(folder)


def test_a_file_from_outside_the_folder_is_open_but_not_in_it(serving, folder):
    get, _ = serving(SIMPLE, folder)
    listing = json.loads(get("/api/traces")[2])
    assert listing["current"] == SIMPLE.name and listing["current_in_folder"] is False


def test_only_json_files_are_listed(serving, folder):
    (folder / "notes.txt").write_text("hello", encoding="utf-8")
    get, _ = serving(SIMPLE, folder)
    assert "notes.txt" not in {f["name"] for f in json.loads(get("/api/traces")[2])["files"]}


def test_a_missing_folder_gives_an_empty_list(serving, tmp_path):
    get, _ = serving(SIMPLE, tmp_path / "none")
    listing = json.loads(get("/api/traces")[2])
    assert listing["files"] == [] and listing["folder"] is not None


def test_a_file_of_the_folder_is_served_by_its_name(serving, folder):
    get, _ = serving(SIMPLE, folder)
    status, _, body = get("/trace.json?name=20261112-100000-000.json")
    assert status == 200 and json.loads(body)["trashPanda"]["note"] == "recording 0"
    assert json.loads(get("/trace.json")[2])["trashPanda"]["note"] != "recording 0"          # without a name: the one the command opened


@pytest.mark.parametrize("name", ["missing.json", "../dummy_trace.json", "..%2Fdummy_trace.json", "%2e%2e%2fdummy_trace.json",
                                  "sub%2Ffile.json", "C%3A%5Cwindows%5Cwin.ini", "notes.txt"])
def test_a_name_that_is_not_a_trace_file_of_the_folder_is_not_served(serving, folder, name):
    (folder / "notes.txt").write_text("hello", encoding="utf-8")
    get, _ = serving(SIMPLE, folder)
    status, _, body = get("/trace.json?name=" + name)
    assert status == 404 and "error" in json.loads(body)


def test_a_name_is_refused_when_there_is_no_folder(serving):
    get, _ = serving()
    assert get("/trace.json?name=dummy_trace.json")[0] == 404
