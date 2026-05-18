"""Content-addressable artifact store.

Large tool payloads (> ARTIFACT_THRESHOLD_BYTES) are persisted here.
Memory holds the handle string; Decision sees bytes only when Perception
explicitly attaches them via Goal.attach_artifact_id.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schemas import Artifact

ARTIFACT_THRESHOLD_BYTES = 4096  # 4 KB

_STATE_DIR = Path("state")
_ARTIFACTS_DIR = _STATE_DIR / "artifacts"


class ArtifactStore:
    def __init__(self) -> None:
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        blob: bytes,
        *,
        content_type: str,
        source: str,
        descriptor: str,
    ) -> str:
        """Store bytes and return handle of the form 'art:<sha256-prefix>'."""
        sha = hashlib.sha256(blob).hexdigest()[:16]
        artifact_id = f"art:{sha}"
        bin_path = _ARTIFACTS_DIR / f"{sha}.bin"
        meta_path = _ARTIFACTS_DIR / f"{sha}.json"
        if not bin_path.exists():
            bin_path.write_bytes(blob)
            meta = Artifact(
                id=artifact_id,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor,
            )
            meta_path.write_text(meta.model_dump_json(), encoding="utf-8")
        return artifact_id

    def get_bytes(self, artifact_id: str) -> bytes:
        sha = artifact_id.removeprefix("art:")
        return (_ARTIFACTS_DIR / f"{sha}.bin").read_bytes()

    def get_meta(self, artifact_id: str) -> Artifact:
        sha = artifact_id.removeprefix("art:")
        data = json.loads((_ARTIFACTS_DIR / f"{sha}.json").read_text(encoding="utf-8"))
        return Artifact.model_validate(data)

    def exists(self, artifact_id: str) -> bool:
        sha = artifact_id.removeprefix("art:")
        return (_ARTIFACTS_DIR / f"{sha}.bin").exists()


# Module-level singleton
artifacts = ArtifactStore()
