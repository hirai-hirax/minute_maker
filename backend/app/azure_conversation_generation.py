"""Utilities to synthesize Azure-based test audio for diarization and summarization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping


@dataclass
class ConversationTurn:
    """Single speaker turn used to build the conversation SSML."""

    speaker: str
    text: str


DEFAULT_TURNS: List[ConversationTurn] = [
    ConversationTurn(
        speaker="Facilitator",
        text=(
            "Thanks for joining today's design sync. "
            "We need to lock down the minute maker prototype for next week's demo."
        ),
    ),
    ConversationTurn(
        speaker="Engineer",
        text=(
            "From the backend side we're sticking with FastAPI and would like the audio"
            " processing queue to stay optional so we can demo locally."
        ),
    ),
    ConversationTurn(
        speaker="Product Manager",
        text=(
            "On the product side, we need to show diarization and a short summary."
            " Azure Speech should cover transcription and embeddings if we wire it up."
        ),
    ),
    ConversationTurn(
        speaker="Facilitator",
        text=(
            "Exactly. Let's also capture a mock transcript so the frontend can render"
            " speaker labels."
        ),
    ),
    ConversationTurn(
        speaker="Engineer",
        text=(
            "I'll generate a three-speaker sample about a minute long using different"
            " voices so diarization can be validated."
        ),
    ),
    ConversationTurn(
        speaker="Product Manager",
        text=(
            "Great. I'll verify that the summary matches and that we can extract"
            " speaker embeddings for the handoff."
        ),
    ),
]


def build_three_speaker_script(cycles: int = 2) -> List[ConversationTurn]:
    """Return a repeatable three-speaker conversation script.

    The default cycles value repeats the base script twice, yielding roughly one
    minute of synthesized audio with natural pauses injected by SSML breaks. The
    script is deterministic, which makes it useful for reproducible tests.
    """

    script: List[ConversationTurn] = []
    for _ in range(cycles):
        script.extend(DEFAULT_TURNS)
    return script


def script_to_ssml(
    script: Iterable[ConversationTurn],
    voices: Mapping[str, str] | None = None,
    locale: str = "en-US",
) -> str:
    """Convert a conversation script into an SSML payload with per-speaker voices."""

    voice_overrides = voices or {
        "Facilitator": "en-US-JennyMultilingualNeural",
        "Engineer": "en-US-GuyNeural",
        "Product Manager": "en-US-SaraNeural",
    }

    speak_body = []
    for turn in script:
        voice_name = voice_overrides.get(turn.speaker, "en-US-JennyMultilingualNeural")
        speak_body.append(
            (
                f"<voice name=\"{voice_name}\">"
                f"<prosody rate=\"0%\">{turn.text}</prosody>"
                "</voice>"
                "<break time=\"1.5s\"/>"
            )
        )

    return (
        f"<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" "
        f"xml:lang=\"{locale}\">" + "".join(speak_body) + "</speak>"
    )


def synthesize_script_to_file(
    script: Iterable[ConversationTurn],
    output_path: Path,
    *,
    subscription_key: str,
    region: str,
    voices: Mapping[str, str] | None = None,
    locale: str = "en-US",
) -> Path:
    """Synthesize the provided script to ``output_path`` using Azure Speech.

    Azure credentials are passed explicitly to make the function easy to use from
    tests. ``output_path`` is created if it does not exist. Returns the path for
    convenience.
    """

    import azure.cognitiveservices.speech as speechsdk

    output_path.parent.mkdir(parents=True, exist_ok=True)

    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm
    )

    ssml = script_to_ssml(script, voices=voices, locale=locale)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Synthesis failed: {result.reason}")

    return output_path
