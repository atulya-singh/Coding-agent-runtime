"""Grading tests for requests-008: the build_connection_pool_key_attributes hook.

Two halves, matching the two halves of the task: the new method exists and is
genuinely dispatched through self (so a subclass can change the pool key), and
the default answers are byte-for-byte what the adapter produced before.

Building a urllib3 pool object opens no socket, so this runs offline.
"""
import ssl

import pytest

from requests.adapters import HTTPAdapter
from requests.exceptions import InvalidURL
from requests.models import PreparedRequest

BUNDLE = "/etc/ssl/certs/ca-certificates.crt"


def prepared(url="https://example.com/path"):
    request = PreparedRequest()
    request.prepare_url(url, None)
    return request


def key(adapter, request, verify, cert=None):
    return adapter.build_connection_pool_key_attributes(request, verify, cert)


def test_the_hook_exists_and_returns_a_two_tuple():
    adapter = HTTPAdapter()
    assert hasattr(adapter, "build_connection_pool_key_attributes"), (
        "HTTPAdapter.build_connection_pool_key_attributes is missing"
    )
    host_params, pool_kwargs = key(adapter, prepared(), True)
    assert isinstance(host_params, dict) and isinstance(pool_kwargs, dict)


def test_cert_is_optional():
    """Documented signature is (request, verify, cert=None)."""
    adapter = HTTPAdapter()
    assert adapter.build_connection_pool_key_attributes(prepared(), True) == key(
        adapter, prepared(), True, None
    )


def test_host_params_describe_the_request_url():
    adapter = HTTPAdapter()
    host_params, _ = key(adapter, prepared("https://example.com:8443/path"), True)
    assert host_params == {"scheme": "https", "host": "example.com", "port": 8443}


def test_verify_true_pins_certificate_checking_to_a_reused_context():
    adapter = HTTPAdapter()
    _, first = key(adapter, prepared(), True)
    _, second = key(adapter, prepared(), True)
    assert first["cert_reqs"] == "CERT_REQUIRED"
    assert isinstance(first["ssl_context"], ssl.SSLContext)
    # One shared context, not a fresh one per request: rebuilding it reloads the
    # root store, which is the cost requests deliberately stopped paying.
    assert first["ssl_context"] is second["ssl_context"]


def test_verify_false_disables_certificate_checking():
    _, pool_kwargs = key(HTTPAdapter(), prepared(), False)
    assert pool_kwargs["cert_reqs"] == "CERT_NONE"
    assert "ssl_context" not in pool_kwargs


def test_verify_path_becomes_ca_certs():
    _, pool_kwargs = key(HTTPAdapter(), prepared(), BUNDLE)
    assert pool_kwargs["ca_certs"] == BUNDLE
    assert pool_kwargs["cert_reqs"] == "CERT_REQUIRED"


def test_verify_directory_becomes_ca_cert_dir(tmp_path):
    _, pool_kwargs = key(HTTPAdapter(), prepared(), str(tmp_path))
    assert pool_kwargs["ca_cert_dir"] == str(tmp_path)
    assert "ca_certs" not in pool_kwargs


@pytest.mark.parametrize(
    "cert, expected",
    (
        (("client.pem", "client.key"), {"cert_file": "client.pem", "key_file": "client.key"}),
        ("combined.pem", {"cert_file": "combined.pem"}),
    ),
)
def test_client_cert_is_carried_through(cert, expected):
    _, pool_kwargs = key(HTTPAdapter(), prepared(), True, cert)
    for name, value in expected.items():
        assert pool_kwargs[name] == value


def test_get_connection_with_tls_context_dispatches_through_self():
    """The point of the change: a subclass can steer the pool key."""

    class OverridingAdapter(HTTPAdapter):
        def __init__(self):
            super().__init__()
            self.calls = []

        def build_connection_pool_key_attributes(self, request, verify, cert=None):
            host_params, pool_kwargs = super().build_connection_pool_key_attributes(
                request, verify, cert
            )
            self.calls.append((request.url, verify, cert))
            host_params["port"] = 4443
            return host_params, pool_kwargs

    adapter = OverridingAdapter()
    connection = adapter.get_connection_with_tls_context(prepared(), verify=True)

    assert adapter.calls, "get_connection_with_tls_context bypassed the hook"
    assert connection.port == 4443, "the override did not affect the connection pool"


def test_malformed_url_still_raises_invalid_url():
    """A ValueError out of the pool-key step is still surfaced as InvalidURL."""
    adapter = HTTPAdapter()
    request = PreparedRequest()
    # Set directly: prepare_url would reject this before the adapter sees it. An
    # out-of-range port makes urlparse().port raise while the pool key is built,
    # which is the path that must stay wrapped.
    request.url = "https://example.com:99999/path"
    with pytest.raises(InvalidURL):
        adapter.get_connection_with_tls_context(request, verify=True)
