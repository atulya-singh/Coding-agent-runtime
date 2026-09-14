"""Grading tests for requests-009: chardet -> charset_normalizer.

Importing this module at all is most of the criterion -- before the migration,
`import requests` raises ModuleNotFoundError and the whole file errors during
collection.

Note the environment assertion up front. The agent could otherwise "fix" the
import by installing chardet inside its container, which is not a migration.
Phase 4 regrades in a fresh container built from the task's setup steps, so a
stray pip install cannot survive into grading anyway, but asserting it here
makes the requirement explicit rather than implicit.
"""
import importlib.util
import warnings

import pytest


def test_chardet_really_is_absent():
    assert importlib.util.find_spec("chardet") is None, (
        "chardet is installed, so this run does not demonstrate the migration"
    )


def test_requests_imports_cleanly():
    import requests  # noqa: F401


def test_import_emits_no_dependency_warning():
    """The startup version check must recognise the installed charset_normalizer."""
    import requests
    from requests.exceptions import RequestsDependencyWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(requests)

    offenders = [str(w.message) for w in caught if issubclass(w.category, RequestsDependencyWarning)]
    assert not offenders, offenders


def test_compat_detector_is_charset_normalizer():
    from requests.compat import chardet as detector

    assert detector.__name__ == "charset_normalizer"


def test_packages_alias_is_preserved():
    """Backwards compatibility: requests.packages.chardet must still resolve."""
    import requests.packages

    assert requests.packages.chardet.__name__ == "charset_normalizer"


@pytest.mark.parametrize(
    "text, encoding",
    (
        ("hello world, this is plain ascii content for detection", "ascii"),
        ("héllo wörld, this is latin text encoded as utf-8 for detection", "utf-8"),
    ),
)
def test_apparent_encoding_detects_through_the_new_library(text, encoding):
    import requests

    response = requests.Response()
    response._content = text.encode(encoding)
    assert response.apparent_encoding is not None
    assert response.apparent_encoding.lower().replace("_", "-") in (encoding, "utf-8")
