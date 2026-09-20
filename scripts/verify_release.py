#!/usr/bin/env python3
"""Fast checks that do not require policy checkpoints or a GPU."""

import compileall
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_hashes():
    for name in ("CURRENT_MANIFEST.sha256",):
        manifest = ROOT / "results" / name
        for line in manifest.read_text().splitlines():
            expected, rel = line.split(maxsplit=1)
            path = ROOT / rel
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"hash mismatch: {rel}")


def check_portable_manifest():
    path = ROOT / "results/manifests/final_test/test_episode_manifest_portable.json"
    rows = json.loads(path.read_text())
    assert len(rows) == 238
    assert len({(x["task_id"], x["trial_index"], x["condition"]) for x in rows}) == 238
    for row in rows:
        value = row.get("perturbation_manifest")
        if value:
            assert not Path(value).is_absolute()
            assert (path.parent / value).resolve().is_file()


def check_artifact_links():
    import sys
    sys.path.insert(0, str(ROOT / "openvla-oft"))
    from artifact_paths import resolve_artifact_path

    v2_path = ROOT / "openvla-oft/observable_cascade_v1/models/v2_candidate.json"
    v2 = json.loads(v2_path.read_text())
    v1_path = resolve_artifact_path(v2["v1_artifact"], anchor=v2_path)
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == v2["v1_artifact_sha256"]
    v1 = json.loads(v1_path.read_text())
    weights = resolve_artifact_path(v1["checkpoint"], anchor=v1_path)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == v2["v1_weights_sha256"]


def check_tree():
    forbidden = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in ROOT.rglob("*"):
        if path.name in forbidden:
            raise AssertionError(f"generated cache present: {path}")
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            raise AssertionError(f"unexpected file larger than 50 MB: {path}")


def check_private_paths():
    # Byte-frozen provenance files intentionally retain their evaluated paths;
    # artifact_paths.py and the portable manifest relocate runtime references.
    manifests = [ROOT / "results/CURRENT_MANIFEST.sha256"]
    allowed = {ROOT / line.split(maxsplit=1)[1]
               for manifest in manifests for line in manifest.read_text().splitlines()}
    allowed.add(Path(__file__).resolve())
    pattern = re.compile(rb"/home/yerincho04|/scratch2/yerincho04|Mixture-Of-VLA-master")
    offenders = []
    for path in ROOT.rglob("*"):
        if (not path.is_file() or path in allowed or ".git" in path.parts
                or "__pycache__" in path.parts):
            continue
        try:
            if pattern.search(path.read_bytes()):
                offenders.append(str(path.relative_to(ROOT)))
        except OSError:
            pass
    if offenders:
        raise AssertionError(f"machine-specific paths remain: {offenders}")


def remove_caches():
    for name in ("__pycache__", ".pytest_cache", ".ruff_cache"):
        for path in sorted(ROOT.rglob(name), reverse=True):
            shutil.rmtree(path)


def main():
    remove_caches()
    check_tree()
    check_hashes()
    check_portable_manifest()
    check_artifact_links()
    check_private_paths()
    ok = compileall.compile_dir(ROOT / "openvla-oft", quiet=1)
    ok &= compileall.compile_dir(ROOT / "VLA-Adapter", quiet=1)
    if not ok:
        raise AssertionError("Python compilation failed")
    # Remove caches created by compileall so verification leaves a clean tree.
    remove_caches()
    print("release verification: PASS")


if __name__ == "__main__":
    main()
