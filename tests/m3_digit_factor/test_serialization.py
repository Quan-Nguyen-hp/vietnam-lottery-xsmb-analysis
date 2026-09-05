"""Tests for XPIS v3 M3 canonical serialization (Slice M3-10)."""

from __future__ import annotations

import datetime
from enum import Enum
import gzip
import struct
from typing import Any
import zlib

import numpy as np
import pytest

from src.m3_digit_factor.authority import canonical_authority_json_bytes
from src.m3_digit_factor.serialization import (
    CSV_BOM,
    CSV_DELIMITER,
    CSV_ENCODING,
    CSV_FINAL_LF,
    CSV_FLOAT_FORMAT,
    CSV_LINE_TERMINATOR,
    CSV_QUOTECHAR,
    CSV_QUOTING,
    GZIP_COMPRESSION_LEVEL,
    GZIP_HEADER_BYTES,
    GZIP_MEMBER_COUNT,
    GZIP_MTIME,
    GZIP_OS,
    GZIP_XFL,
    JSON_ALLOW_NAN,
    JSON_ENSURE_ASCII,
    JSON_FINAL_LF,
    JSON_SEPARATORS,
    JSON_SORT_KEYS,
    SerializationError,
    format_csv_scalar,
    serialize_csv,
    serialize_gzip,
    serialize_json,
)


# --- 1. JSON Serialization Tests ---


def test_json_constants() -> None:
    """JSON serialization constants match frozen specification."""
    assert JSON_ALLOW_NAN is False
    assert JSON_ENSURE_ASCII is False
    assert JSON_SORT_KEYS is True
    assert JSON_SEPARATORS == (",", ":")
    assert JSON_FINAL_LF is True


def test_json_utf8_no_bom_and_final_lf() -> None:
    """JSON output must be UTF-8 without BOM and terminate with exactly one LF."""
    payload = {"key": "value"}
    res = serialize_json(payload)
    assert isinstance(res, bytes)
    assert not res.startswith(b"\xef\xbb\xbf")  # No UTF-8 BOM
    assert res.endswith(b"\n")
    assert not res.endswith(b"\r\n")
    assert res.count(b"\n") == 1


def test_json_ensure_ascii_false_handles_unicode() -> None:
    """Unicode characters are emitted as raw UTF-8, not \\uXXXX."""
    payload = {"message": "Xổ số Miền Bắc — Hà Nội"}
    res = serialize_json(payload)
    assert "Xổ số Miền Bắc — Hà Nội".encode("utf-8") in res
    assert b"\\u" not in res


def test_json_key_sorting_nested_and_array_order_preservation() -> None:
    """Object keys are lexicographically sorted at all depths; array order is preserved."""
    payload = {
        "z_key": 1,
        "a_key": [3, 1, 2],  # Array order MUST NOT be sorted
        "m_key": {"b_sub": 20, "a_sub": 10},
    }
    res = serialize_json(payload)
    expected_str = '{"a_key":[3,1,2],"m_key":{"a_sub":10,"b_sub":20},"z_key":1}\n'
    assert res == expected_str.encode("utf-8")


def test_json_compact_separators_no_pretty_whitespace() -> None:
    """No spaces after commas or colons."""
    payload = {"a": 1, "b": 2}
    res = serialize_json(payload)
    assert res == b'{"a":1,"b":2}\n'


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {"nested": float("nan")},
        [1, 2, float("inf")],
    ],
)
def test_json_rejects_non_finite_floats(bad_value: Any) -> None:
    """Non-finite floats must be rejected anywhere in the structure."""
    with pytest.raises(SerializationError, match="Out of range float values|Non-finite float"):
        serialize_json(bad_value)


def test_json_deterministic_repeatability() -> None:
    """Identical input produces bitwise identical bytes."""
    payload = {"k": [1, 3, 5, 10], "v": {"pi": 3.14159}}
    res1 = serialize_json(payload)
    res2 = serialize_json(payload)
    assert res1 == res2


def test_m3_01_fingerprint_regression_distinct_from_artifact_json() -> None:
    """M3-01 authority JSON has NO trailing LF; M3-10 artifact JSON HAS trailing LF."""
    auth_data = {"version": "V3", "seed": 20260831}
    auth_bytes = canonical_authority_json_bytes(auth_data)
    artifact_bytes = serialize_json(auth_data)

    assert not auth_bytes.endswith(b"\n")
    assert artifact_bytes.endswith(b"\n")
    assert artifact_bytes == auth_bytes + b"\n"


def test_json_literal_golden_fixture() -> None:
    """Golden literal byte-level JSON test."""
    obj = {"b": 2, "a": 1, "tags": ["m3", "test"]}
    res = serialize_json(obj)
    expected = b'{"a":1,"b":2,"tags":["m3","test"]}\n'
    assert res == expected


# --- 2. CSV Scalar Formatting Tests ---


def test_csv_scalar_formatting() -> None:
    """Verify scalar formatting rules for CSV."""
    assert format_csv_scalar(None) == ""
    assert format_csv_scalar(True) == "true"
    assert format_csv_scalar(False) == "false"
    assert format_csv_scalar(0) == "0"
    assert format_csv_scalar(42) == "42"
    assert format_csv_scalar(np.int64(99)) == "99"

    # Float .17g format
    assert format_csv_scalar(0.27) == "0.27000000000000002"
    assert format_csv_scalar(1.0) == "1"
    assert format_csv_scalar(-0.0) == "-0"
    assert format_csv_scalar(1.0 / 3.0) == "0.33333333333333331"

    # Date format
    d = datetime.date(2026, 1, 15)
    assert format_csv_scalar(d) == "2026-01-15"

    # Enum format
    class DummyEnum(str, Enum):
        VAL = "MY_VAL"

    assert format_csv_scalar(DummyEnum.VAL) == "MY_VAL"


def test_csv_boolean_subclass_precedence() -> None:
    """bool subclasses int in Python, so bool must be formatted as true/false, not 1/0."""
    assert isinstance(True, int)
    assert format_csv_scalar(True) == "true"
    assert format_csv_scalar(False) == "false"


@pytest.mark.parametrize(
    "bad_float",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_csv_rejects_non_finite_floats(bad_float: Any) -> None:
    """CSV scalar formatter must reject non-finite float values."""
    with pytest.raises(SerializationError, match="Non-finite float"):
        format_csv_scalar(bad_float)


# --- 3. CSV Serialization Tests ---


def test_csv_constants() -> None:
    """CSV serialization constants match frozen specification."""
    assert CSV_ENCODING == "UTF-8"
    assert CSV_BOM is False
    assert CSV_DELIMITER == ","
    assert CSV_QUOTECHAR == '"'
    assert CSV_QUOTING == 0  # csv.QUOTE_MINIMAL
    assert CSV_LINE_TERMINATOR == "\n"
    assert CSV_FINAL_LF is True
    assert CSV_FLOAT_FORMAT == ".17g"


def test_csv_minimal_quoting_and_escapes() -> None:
    """Verify minimal quoting, comma handling, quote escaping, and newline quoting."""
    header = ["normal", "with_comma", "with_quote", "with_newline"]
    rows = [
        ["plain", "a,b", 'he said "hello"', "line1\nline2"],
    ]
    res = serialize_csv(header, rows)
    expected = (
        'normal,with_comma,with_quote,with_newline\n'
        'plain,"a,b","he said ""hello""","line1\nline2"\n'
    ).encode("utf-8")
    assert res == expected


def test_csv_no_cr_bytes_and_utf8_encoding() -> None:
    """Verify strictly LF line endings and UTF-8 characters."""
    header = ["stt", "tên"]
    rows = [
        [1, "Hà Nội"],
        [2, "Hải Phòng"],
    ]
    res = serialize_csv(header, rows)
    assert b"\r" not in res
    assert res.endswith(b"\n")
    assert not res.startswith(b"\xef\xbb\xbf")  # No BOM
    assert "Hà Nội".encode("utf-8") in res


def test_csv_header_and_row_order_preserved() -> None:
    """Serializer must never reorder headers or rows."""
    header = ["z", "a", "m"]
    rows = [
        [10, 20, 30],
        [1, 2, 3],
    ]
    res = serialize_csv(header, rows)
    expected = b"z,a,m\n10,20,30\n1,2,3\n"
    assert res == expected


def test_csv_literal_golden_fixture() -> None:
    """Golden literal byte-level CSV test."""
    header = ["stage", "model_id", "p", "qualifies"]
    rows = [
        ["DEV", "B0", 0.27, True],
        ["VAL", "M3", 1.0, False],
    ]
    res = serialize_csv(header, rows)
    expected = (
        b"stage,model_id,p,qualifies\n"
        b"DEV,B0,0.27000000000000002,true\n"
        b"VAL,M3,1,false\n"
    )
    assert res == expected


# --- 4. Gzip Serialization Tests ---


def test_gzip_constants() -> None:
    """Gzip constants match frozen specification."""
    assert GZIP_MEMBER_COUNT == 1
    assert GZIP_COMPRESSION_LEVEL == 9
    assert GZIP_HEADER_BYTES == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    assert GZIP_MTIME == 0
    assert GZIP_XFL == 2
    assert GZIP_OS == 255


def test_gzip_first_10_bytes_exact() -> None:
    """Gzip output starts with the exact frozen 10-byte canonical header."""
    payload = b"simple test payload"
    gz = serialize_gzip(payload)
    assert gz[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    assert gz[:10].hex() == "1f8b08000000000002ff"


def test_gzip_trailer_crc32_and_isize() -> None:
    """Gzip trailer contains exact little-endian CRC-32 and ISIZE."""
    payload = b"test payload for crc verification 12345"
    gz = serialize_gzip(payload)

    expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
    expected_isize = len(payload) % (2**32)

    actual_crc, actual_isize = struct.unpack("<II", gz[-8:])
    assert actual_crc == expected_crc
    assert actual_isize == expected_isize


def test_gzip_single_member_verification() -> None:
    """Gzip stream must contain exactly one member with no trailing unused data."""
    payload = b"data for single member check"
    gz = serialize_gzip(payload)

    decompressor = zlib.decompressobj(wbits=31)
    uncompressed = decompressor.decompress(gz)
    assert uncompressed == payload
    assert decompressor.unused_data == b""


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"small ascii",
        "Unicode: Giải đặc biệt xổ số".encode("utf-8"),
        b"target_date,model_id\n2026-01-01,B0\n" * 50,
        bytes(range(256)) * 10,
    ],
)
def test_gzip_round_trip_and_determinism(payload: bytes) -> None:
    """Gzip round-trips identically and produces deterministic bytes across repeated calls."""
    gz1 = serialize_gzip(payload)
    gz2 = serialize_gzip(payload)
    assert gz1 == gz2  # Deterministic

    decompressed = gzip.decompress(gz1)
    assert decompressed == payload


# --- 5. Input Immutability Tests ---


def test_input_immutability() -> None:
    """Serialization functions must not mutate input collections or byte buffers."""
    # JSON input
    json_in = {"b": [1, 2], "a": {"sub": 3}}
    json_copy = {"b": [1, 2], "a": {"sub": 3}}
    _ = serialize_json(json_in)
    assert json_in == json_copy

    # CSV input
    header_in = ["col1", "col2"]
    header_copy = list(header_in)
    rows_in = [["a", "b"], ["c", "d"]]
    rows_copy = [list(r) for r in rows_in]
    _ = serialize_csv(header_in, rows_in)
    assert header_in == header_copy
    assert rows_in == rows_copy

    # Gzip input
    bytes_in = b"immutable bytes"
    bytes_copy = bytes(bytes_in)
    _ = serialize_gzip(bytes_in)
    assert bytes_in == bytes_copy
