"""Canonical deterministic serialization primitives for XPIS v3 M3 artifacts (Slice M3-10).

This module implements:
1. Canonical compact JSON serialization with sorted keys, no NaN, UTF-8, and terminating LF.
2. Canonical RFC-compliant CSV serialization with .17g float representation, minimal quoting,
   lowercase boolean formatting, and LF termination.
3. Deterministic single-member gzip byte compression with fixed canonical headers (OS=255,
   MTIME=0, XFL=2, FLG=0) and standard CRC32/ISIZE trailer.
"""

from __future__ import annotations

import csv
from enum import Enum
import io
import json
import math
import struct
from typing import Any, Sequence
import zlib

import numpy as np


JSON_ENCODING: str = "UTF-8"
JSON_BOM: bool = False
JSON_ALLOW_NAN: bool = False
JSON_ENSURE_ASCII: bool = False
JSON_SORT_KEYS: bool = True
JSON_SEPARATORS: tuple[str, str] = (",", ":")
JSON_FINAL_LF: bool = True

CSV_ENCODING: str = "UTF-8"
CSV_BOM: bool = False
CSV_DELIMITER: str = ","
CSV_QUOTECHAR: str = '"'
CSV_QUOTING: int = csv.QUOTE_MINIMAL
CSV_LINE_TERMINATOR: str = "\n"
CSV_FINAL_LF: bool = True
CSV_FLOAT_FORMAT: str = ".17g"

GZIP_MEMBER_COUNT: int = 1
GZIP_COMPRESSION_LEVEL: int = 9
GZIP_HEADER_BYTES: bytes = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
GZIP_MTIME: int = 0
GZIP_XFL: int = 2
GZIP_OS: int = 255


class SerializationError(ValueError):
    """Base error for all serialization and formatting errors."""


def serialize_json(value: Any) -> bytes:
    """Serialize a data structure to canonical compact JSON bytes.

    Settings:
      - UTF-8 without BOM
      - allow_nan=False (strictly rejects NaN and Inf)
      - ensure_ascii=False (preserves direct UTF-8 Unicode)
      - sort_keys=True (lexicographical key sorting at all depths)
      - separators=(',', ':') (no extra whitespace)
      - final terminating LF
    """
    try:
        json_str = json.dumps(
            value,
            allow_nan=JSON_ALLOW_NAN,
            ensure_ascii=JSON_ENSURE_ASCII,
            sort_keys=JSON_SORT_KEYS,
            separators=JSON_SEPARATORS,
        )
    except ValueError as err:
        raise SerializationError(f"JSON serialization error: {err}") from err

    return (json_str + "\n").encode("utf-8")


def format_csv_scalar(value: Any) -> str:
    """Format an individual Python scalar for canonical CSV representation.

    Rules:
      - None -> "" (empty field)
      - bool -> "true" / "false" (lowercase; checked before int)
      - int / np.integer -> base-10 textual integer
      - float / np.floating -> format(value, ".17g") (rejects non-finite floats)
      - datetime.date / object with .isoformat() -> YYYY-MM-DD
      - Enum -> str(value.value)
      - str -> raw string
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, np.integer)):
        return str(value)

    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise SerializationError(f"Non-finite float value {value!r} cannot be serialized to CSV")
        return format(value, CSV_FLOAT_FORMAT)

    if hasattr(value, "isoformat") and callable(value.isoformat):
        return str(value.isoformat())

    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def serialize_csv(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    """Serialize tabular data into canonical UTF-8 CSV bytes.

    Settings:
      - UTF-8 without BOM
      - Comma delimiter, double-quote quotechar, QUOTE_MINIMAL
      - LF line terminator only (no CR or CRLF)
      - Preserves caller-supplied header order and row order
      - Terminating LF
    """
    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=CSV_DELIMITER,
        quotechar=CSV_QUOTECHAR,
        quoting=CSV_QUOTING,
        lineterminator=CSV_LINE_TERMINATOR,
    )

    writer.writerow([str(h) for h in header])
    for row in rows:
        writer.writerow([format_csv_scalar(val) for val in row])

    return output.getvalue().encode("utf-8")


def serialize_gzip(payload: bytes) -> bytes:
    """Compress payload into canonical deterministic single-member gzip bytes.

    Construction:
      - 10-byte fixed header: 1f 8b 08 00 00 00 00 00 02 ff
        (ID1=1f, ID2=8b, CM=8, FLG=0, MTIME=0, XFL=2, OS=255)
      - Raw DEFLATE level-9 compressed stream
      - 8-byte trailer: CRC32 (little endian) + ISIZE (len % 2^32, little endian)
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise SerializationError(f"Gzip payload must be bytes or bytearray, got {type(payload).__name__}")

    raw_bytes = bytes(payload)

    compressor = zlib.compressobj(
        level=GZIP_COMPRESSION_LEVEL,
        method=zlib.DEFLATED,
        wbits=-zlib.MAX_WBITS,
        memLevel=9,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )
    deflated = compressor.compress(raw_bytes) + compressor.flush()

    crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
    isize = len(raw_bytes) % (2**32)
    trailer = struct.pack("<II", crc, isize)

    return GZIP_HEADER_BYTES + deflated + trailer
