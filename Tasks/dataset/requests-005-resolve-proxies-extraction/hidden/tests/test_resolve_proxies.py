"""Grading tests for requests-005: extracting resolve_proxies() out of Session.

A refactor breaks no existing test, so passing the old suite proves nothing. This
file grades three things instead: the new function exists with the documented
contract, it behaves the way the inline code did, and the restructuring actually
happened rather than the logic being copied.

Everything here runs offline -- proxy resolution is string and environment work,
and the one end-to-end check uses a recording adapter instead of a socket.
"""
import inspect

import pytest

import requests
import requests.adapters
import requests.models
import requests.sessions
from requests.models import PreparedRequest

PROXY_ENV_VARS = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY")


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    """The host's own proxy configuration must not decide whether this passes."""
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def prepared(url="http://target.example.com/path"):
    request = PreparedRequest()
    request.prepare_url(url, None)
    request.prepare_headers(None)
    return request


class RecordingAdapter(requests.adapters.BaseAdapter):
    """Captures the request that would have gone out, and never opens a socket."""

    def __init__(self):
        super().__init__()
        self.request = None

    def send(self, request, **kwargs):
        self.request = request
        response = requests.models.Response()
        response.status_code = 200
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


def test_resolve_proxies_is_a_module_level_function():
    resolve_proxies = getattr(requests.utils, "resolve_proxies", None)
    assert callable(resolve_proxies), "requests.utils.resolve_proxies is missing"
    signature = inspect.signature(resolve_proxies)
    assert list(signature.parameters)[:3] == ["request", "proxies", "trust_env"]
    assert signature.parameters["trust_env"].default is True


def test_none_proxies_is_tolerated():
    assert requests.utils.resolve_proxies(prepared(), None, False) == {}


def test_caller_proxies_are_returned_without_being_mutated():
    proxies = {"http": "http://proxy.example:8080"}
    resolved = requests.utils.resolve_proxies(prepared(), proxies, False)
    assert resolved == proxies
    assert resolved is not proxies, "resolve_proxies must not hand back the caller's dict"


def test_environment_proxy_is_used_when_trust_env(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://env.example:3128")
    resolved = requests.utils.resolve_proxies(prepared(), {}, True)
    assert resolved["http"] == "http://env.example:3128"


def test_environment_proxy_is_ignored_when_not_trust_env(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://env.example:3128")
    assert requests.utils.resolve_proxies(prepared(), {}, False) == {}


def test_no_proxy_key_inside_proxies_bypasses_the_environment(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://env.example:3128")
    resolved = requests.utils.resolve_proxies(prepared(), {"no_proxy": "target.example.com"}, True)
    assert "http" not in resolved


def test_explicit_proxy_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://env.example:3128")
    resolved = requests.utils.resolve_proxies(prepared(), {"http": "http://explicit.example:8080"}, True)
    assert resolved["http"] == "http://explicit.example:8080"


def test_rebuild_proxies_still_returns_the_resolved_mapping(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://env.example:3128")
    session = requests.Session()
    request = prepared()
    assert session.rebuild_proxies(request, {}) == requests.utils.resolve_proxies(
        request, {}, True
    )


def test_rebuild_proxies_still_strips_proxy_authorization():
    """Redirect-time behavior is deliberately unchanged."""
    session = requests.Session()
    session.trust_env = False
    request = prepared()
    request.headers["Proxy-Authorization"] = "Bearer XXX"
    session.rebuild_proxies(request, {})
    assert "Proxy-Authorization" not in request.headers


def test_send_no_longer_strips_proxy_authorization():
    """The reason for the refactor: send() must not inherit redirect behavior."""
    session = requests.Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)
    session.headers["Proxy-Authorization"] = "Bearer XXX"

    session.get("http://target.example.com/path")

    assert adapter.request is not None, "the adapter was never reached"
    assert adapter.request.headers.get("Proxy-Authorization") == "Bearer XXX"


def test_resolution_logic_is_not_duplicated_in_rebuild_proxies():
    source = inspect.getsource(requests.sessions.SessionRedirectMixin.rebuild_proxies)
    assert "resolve_proxies" in source, "rebuild_proxies should delegate to resolve_proxies"
    for symbol in ("should_bypass_proxies", "get_environ_proxies"):
        assert symbol not in source, (
            f"rebuild_proxies still calls {symbol}; that logic belongs to "
            f"requests.utils.resolve_proxies now"
        )


def test_send_resolves_proxies_directly():
    source = inspect.getsource(requests.sessions.Session.send)
    assert "resolve_proxies" in source
    assert "rebuild_proxies" not in source
