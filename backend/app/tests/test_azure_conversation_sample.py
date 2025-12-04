"""Integration test helpers for generating Azure Speech sample audio."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.app.azure_conversation_generation import (
    build_three_speaker_script,
    script_to_ssml,
    synthesize_script_to_file,
)


pytest.importorskip("azure.cognitiveservices.speech", reason="Azure Speech SDK not installed")


@pytest.mark.skipif(
    not os.getenv("AZURE_SPEECH_KEY") or not os.getenv("AZURE_SPEECH_REGION"),
    reason="Azure Speech credentials are not configured",
)
def test_generate_three_speaker_sample(tmp_path: Path) -> None:
    """Generate a ~1 minute Azure sample for diarization, transcript, and embeddings."""

    script = build_three_speaker_script(cycles=2)
    ssml = script_to_ssml(script)
    assert "Facilitator" in ssml or "Engineer" in ssml

    output_file = tmp_path / "three_speaker_minute.wav"
    synthesize_script_to_file(
        script,
        output_file,
        subscription_key=os.environ["AZURE_SPEECH_KEY"],
        region=os.environ["AZURE_SPEECH_REGION"],
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 1024, "synthesized audio should not be empty"
