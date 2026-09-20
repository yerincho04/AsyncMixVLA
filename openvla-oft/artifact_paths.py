"""Resolve frozen artifact paths after moving the repository.

The published JSON artifacts retain their original absolute paths so their
SHA-256 hashes remain identical to the evaluated artifacts. This helper maps a
missing historical ``.../openvla-oft/...`` path into this checkout without
changing the artifact bytes.
"""

from pathlib import Path


OPENVLA_ROOT = Path(__file__).resolve().parent


def resolve_artifact_path(value, *, anchor=None):
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()

    marker = "openvla-oft/"
    normalized = path.as_posix()
    if marker in normalized:
        candidate = OPENVLA_ROOT / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate.resolve()

    if anchor is not None:
        candidate = Path(anchor).resolve().parent / path.name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve frozen artifact path {value!r} inside {OPENVLA_ROOT}"
    )
