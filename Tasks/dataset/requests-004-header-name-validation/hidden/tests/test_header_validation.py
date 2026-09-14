"""Grading tests for requests-004: header *name* validation.

These call requests.utils.check_header_validity directly rather than going
through requests.get(), so the file runs with no network and no httpbin fixture.
The cases mirror the ones the upstream fix (PR #6154) added to
tests/test_requests.py.
"""
import pytest

from requests.exceptions import InvalidHeader
from requests.utils import check_header_validity


@pytest.mark.parametrize(
    "header",
    (
        ("foo", "bar baz qux"),
        ("bar", b"fbbq"),
        ("baz", ""),
        ("qux", "1"),
        (b"key", b"value"),
        (b"key", "value"),
    ),
)
def test_valid_headers_are_accepted(header):
    check_header_validity(header)


@pytest.mark.parametrize(
    "header",
    (
        ("foo", "bar\r\nbaz: qux"),
        ("foo", "bar\n\rbaz: qux"),
        ("foo", "bar\nbaz: qux"),
        ("foo", "bar\rbaz: qux"),
        ("foo", " bar"),
        ("foo", "    bar"),
        ("foo", "\tbar"),
        ("foo", b"bar\r\nbaz: qux"),
    ),
)
def test_invalid_header_values_are_still_rejected(header):
    """The pre-existing value rules must survive the change."""
    with pytest.raises(InvalidHeader):
        check_header_validity(header)


@pytest.mark.parametrize(
    "header",
    (
        ("fo\ro", "bar"),
        ("fo\r\no", "bar"),
        ("fo\n\ro", "bar"),
        ("fo\no", "bar"),
        (" foo", "bar"),
        ("\tfoo", "bar"),
        ("    foo", "bar"),
        (" ", "bar"),
        ("foo:bar", "baz"),
        (b"fo\ro", b"bar"),
        (b" foo", b"bar"),
    ),
)
def test_invalid_header_names_are_rejected(header):
    """The new behaviour: the name half is validated too."""
    with pytest.raises(InvalidHeader):
        check_header_validity(header)


@pytest.mark.parametrize(
    "header, offender",
    (
        (("foo", 3), "3"),
        (("bar", {"foo": "bar"}), "foo"),
        (("baz", ["foo", "bar"]), "foo"),
        ((3, "foo"), "3"),
        ((None, "foo"), "None"),
    ),
)
def test_non_string_header_parts_are_rejected(header, offender):
    with pytest.raises(InvalidHeader) as excinfo:
        check_header_validity(header)
    assert offender in str(excinfo.value)
