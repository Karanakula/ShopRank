"""Every dashboard page renders without errors against a live API (tiny fixture models)."""

import socket
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py"] + sorted(f"pages/{p.name}" for p in (APP / "pages").glob("*.py"))


@pytest.fixture(scope="module")
def live_api(tiny_models, tmp_path_factory):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = tmp_path_factory.mktemp("app_logs") / "predictions.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("SHOPRANK_MODELS", str(tiny_models))
    mp.setenv("SHOPRANK_PREDICTION_LOG", str(log))
    mp.setenv("API_URL", f"http://127.0.0.1:{port}")

    from api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                break
        except requests.ConnectionError:
            time.sleep(0.05)
    mp.syspath_prepend(str(APP))
    yield
    server.should_exit = True
    thread.join(timeout=5)
    mp.undo()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(live_api, page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / page), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title, "every page has a title"
