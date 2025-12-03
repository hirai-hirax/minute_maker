"""Command line interface for minute_maker.

This script removes the Streamlit UI and exposes the previous functionality as
CLI subcommands so that all operations are driven by arguments. The primary
features are:

- Transcribing audio files with Azure OpenAI (gpt-4o-transcribe-diarize or Whisper).
- Optional prompt injection from a reference document for Whisper requests.
- Generating summaries from arbitrary text input.
- Converting a video file to an MP3.

Environment variables
---------------------
AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set for API access.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import fitz
import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydub import AudioSegment
from speechbrain.pretrained import EncoderClassifier
from docx import Document as DocxDocument
from pptx import Presentation

# Load environment variables from .env if present
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-03-01-preview"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# File extraction utilities
# ---------------------------------------------------------------------------

def _extract_pdf(file: BytesIO) -> str:
    file_bytes = file.read()
    file.seek(0)
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text() for page in pdf_document)
    pdf_document.close()
    return text


def _extract_docx(file: BytesIO) -> str:
    doc = DocxDocument(file)
    text = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        text += "\n" + "\n".join(" ".join(cell.text for cell in row.cells) for row in table.rows)
    return text.strip()


def _extract_pptx(file: BytesIO) -> str:
    presentation = Presentation(file)
    text_parts = []
    for slide_num, slide in enumerate(presentation.slides, 1):
        text_parts.append(f"\n--- スライド {slide_num} ---\n")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_parts.append(shape.text + "\n")
            if shape.has_table:
                for row in shape.table.rows:
                    text_parts.append(" ".join(cell.text for cell in row.cells) + "\n")
    return "".join(text_parts).strip()


def _extract_msg(file: BytesIO) -> str:
    try:
        import extract_msg  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "MSGファイルの処理には extract-msg ライブラリが必要です。"
            "pip install extract-msg を実行してください。"
        ) from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=".msg") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    try:
        msg = extract_msg.Message(temp_path)
        parts = []
        if msg.subject:
            parts.append(f"件名: {msg.subject}\n")
        if msg.sender:
            parts.append(f"送信者: {msg.sender}\n")
        if msg.to:
            parts.append(f"宛先: {msg.to}\n")
        if msg.body:
            parts.append(f"\n{msg.body}")
        msg.close()
        return "\n".join(parts).strip()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _extract_txt(file: BytesIO) -> str:
    file_content = file.read()
    for encoding in ["utf-8", "cp932", "shift_jis", "utf-16", "iso-2022-jp"]:
        try:
            return file_content.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise RuntimeError("テキストファイルのエンコーディングを判定できませんでした")


FILE_EXTRACTORS = {
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "pptx": _extract_pptx,
    "txt": _extract_txt,
    "msg": _extract_msg,
}


def extract_text_from_file(path: Path) -> str:
    """Extract text from supported file types.

    Parameters
    ----------
    path: Path
        Path to the input file.
    """

    extractor = FILE_EXTRACTORS.get(path.suffix.lower().lstrip("."))
    if not extractor:
        raise RuntimeError(f"サポートされていないファイル形式: {path.suffix}")
    with path.open("rb") as handle:
        content = extractor(BytesIO(handle.read()))
    return content


def extract_text_from_bytesio(file: BytesIO) -> str:
    extension = file.name.split(".")[-1].lower()
    extractor = FILE_EXTRACTORS.get(extension)
    if not extractor:
        raise RuntimeError(f"サポートされていないファイル形式: {extension}")
    content = extractor(BytesIO(file.getvalue()))
    file.seek(0)
    return content


# ---------------------------------------------------------------------------
# Azure client helpers
# ---------------------------------------------------------------------------

def _create_azure_client(api_version: str = API_VERSION) -> AzureOpenAI:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT と AZURE_OPENAI_API_KEY を設定してください")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=api_version,
    )


def generate_summary(model: str, prompt: str, text: str) -> str:
    client = _create_azure_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

@contextmanager
def temp_file_path(data: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _format_timedelta(seconds: float) -> str:
    return str(timedelta(seconds=seconds))


def _log_progress(current: int, total: int):
    percent = (current / total) * 100
    logging.info("進捗: %s/%s (%.1f%%)", current, total, percent)


@dataclass
class TranscriptResult:
    segments: pd.DataFrame
    model: str

    def to_csv(self, path: Path):
        self.segments.to_csv(path, index=False)

    def to_json(self, path: Path):
        self.segments.to_json(path, orient="records", force_ascii=False, indent=2)

    def summary(self) -> str:
        if self.segments.empty:
            return "セグメントはありません"
        start = self.segments["start"].min()
        end = self.segments["end"].max()
        duration = _format_timedelta(end - start)
        return f"{len(self.segments)} セグメント, 長さ {duration}"


def _transcribe_audio_single(uploaded_file: BytesIO, reference_file: Optional[BytesIO], model: str) -> pd.DataFrame:
    if model == "whisper":
        api_version = "2024-06-01"
        logging.info("Whisperモデルを使用します (API %s)", api_version)
    else:
        api_version = API_VERSION

    client = _create_azure_client(api_version)
    suffix = f".{uploaded_file.name.split('.')[-1]}"

    if model == "gpt-4o-transcribe-diarize":
        if reference_file:
            logging.warning("gpt-4o-transcribe-diarize は参考資料をサポートしません。無視します。")
        logging.info("文字起こしを開始します (gpt-4o-transcribe-diarize, 話者識別あり)")
        with temp_file_path(uploaded_file.getvalue(), suffix) as tmp_path:
            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe-diarize",
                    file=(uploaded_file.name, audio_file, f"audio/{uploaded_file.name.split('.')[-1]}"),
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
        transcript_dict = transcript.model_dump()
        segments = transcript_dict.get("segments", [])
        if segments:
            seg_list = [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "speaker": seg.get("speaker", ""),
                    "text": seg.get("text", ""),
                }
                for seg in segments
            ]
            logging.info("文字起こし完了: %d セグメント", len(seg_list))
            return pd.DataFrame(seg_list)
        text = transcript_dict.get("text", "")
        if text:
            logging.warning("セグメント情報が無いため 1 セグメントで返します")
            return pd.DataFrame(
                [{"start": 0, "end": 0, "speaker": "", "text": text}]
            )
        logging.error("文字起こし結果が空でした")
        return pd.DataFrame(columns=["start", "end", "speaker", "text"])

    if model == "whisper":
        logging.info("文字起こしを開始します (Whisper, 話者識別なし)")
        with temp_file_path(uploaded_file.getvalue(), suffix) as tmp_path:
            with open(tmp_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper",
                    "file": (uploaded_file.name, audio_file, f"audio/{uploaded_file.name.split('.')[-1]}"),
                    "response_format": "verbose_json",
                }
                if reference_file:
                    reference_text = extract_text_from_bytesio(reference_file)
                    if reference_text:
                        kwargs["prompt"] = reference_text[:1000]
                        logging.info("参考資料を prompt として使用します")
                transcript = client.audio.transcriptions.create(**kwargs)
        transcript_dict = transcript.model_dump()
        segments = transcript_dict.get("segments", [])
        if segments:
            seg_list = [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "speaker": "",
                    "text": seg.get("text", ""),
                }
                for seg in segments
            ]
            logging.info("文字起こし完了: %d セグメント", len(seg_list))
            return pd.DataFrame(seg_list)
        text = transcript_dict.get("text", "")
        if text:
            logging.warning("セグメント情報が無いため 1 セグメントで返します")
            return pd.DataFrame(
                [{"start": 0, "end": 0, "speaker": "", "text": text}]
            )
        logging.error("文字起こし結果が空でした")
        return pd.DataFrame(columns=["start", "end", "speaker", "text"])

    raise RuntimeError(f"サポートされていないモデル: {model}")


def _transcribe_large_audio_chunked(uploaded_file: BytesIO, reference_file: Optional[BytesIO], model: str) -> pd.DataFrame:
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with temp_file_path(uploaded_file.getvalue(), suffix) as tmp_path:
        audio = AudioSegment.from_file(tmp_path)
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000
    chunk_duration_ms = 10 * 60 * 1000
    num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
    logging.info("音声長 %.1f 分を %d チャンクに分割して処理します", total_duration_sec / 60, num_chunks)

    all_segments = []
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
        _log_progress(i + 1, num_chunks)
        chunk = audio[start_ms:end_ms]
        chunk_io = BytesIO()
        chunk.export(chunk_io, format="mp3", bitrate="192k")
        chunk_io.seek(0)
        chunk_io.name = f"chunk_{i}.mp3"
        try:
            chunk_df = _transcribe_audio_single(chunk_io, reference_file, model)
            if not chunk_df.empty and {"start", "end"}.issubset(chunk_df.columns):
                offset_sec = start_ms / 1000
                chunk_df["start"] = chunk_df["start"] + offset_sec
                chunk_df["end"] = chunk_df["end"] + offset_sec
            all_segments.append(chunk_df)
        except Exception as exc:  # pragma: no cover - runtime errors are logged
            logging.exception("チャンク %d の処理中にエラー: %s", i + 1, exc)
    if all_segments:
        return pd.concat(all_segments, ignore_index=True)
    logging.error("すべてのチャンクの処理に失敗しました")
    return pd.DataFrame(columns=["start", "end", "speaker", "text"])


def transcribe_audio_to_dataframe(uploaded_file: BytesIO, reference_file: Optional[BytesIO] = None, model: str = "gpt-4o-transcribe-diarize") -> pd.DataFrame:
    max_size = 25 * 1024 * 1024
    uploaded_file.seek(0)
    file_bytes = uploaded_file.getvalue()
    uploaded_file.seek(0)
    if len(file_bytes) > max_size:
        logging.warning("ファイルサイズが25MBを超えています。分割処理を行います。")
        return _transcribe_large_audio_chunked(uploaded_file, reference_file, model)
    return _transcribe_audio_single(uploaded_file, reference_file, model)


def load_speaker_encoder() -> EncoderClassifier:
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )


def _compute_embedding_from_wav_bytes(wav_bytes: bytes) -> torch.Tensor:
    with temp_file_path(wav_bytes, ".wav") as wav_path:
        waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_sr = 16000
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)
    encoder = load_speaker_encoder()
    waveform = waveform.to(dtype=torch.float32)
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)
    return embedding.squeeze().cpu()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _handle_transcribe(args: argparse.Namespace):
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    reference_io = None
    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        reference_io = BytesIO(reference_path.read_bytes())
        reference_io.name = reference_path.name

    audio_io = BytesIO(audio_path.read_bytes())
    audio_io.name = audio_path.name
    df = transcribe_audio_to_dataframe(audio_io, reference_io, args.model)
    result = TranscriptResult(segments=df, model=args.model)
    logging.info("結果: %s", result.summary())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            result.to_csv(output_path)
        else:
            result.to_json(output_path)
        logging.info("保存しました: %s", output_path)

    if args.summary_model and args.summary_prompt:
        summary_text = generate_summary(args.summary_model, args.summary_prompt, "\n".join(result.segments["text"].tolist()))
        print("\n===== 要約 =====\n")
        print(summary_text)


def _handle_summary(args: argparse.Namespace):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)
    text = "\n".join(df.get("text", []))
    summary = generate_summary(args.model, args.prompt, text)
    print(summary)


def _handle_video_to_audio(args: argparse.Namespace):
    input_path = Path(args.video)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    audio = AudioSegment.from_file(input_path)
    output_path = Path(args.output or input_path.with_suffix(".mp3"))
    audio.export(output_path, format="mp3")
    logging.info("音声を保存しました: %s", output_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="minute_maker CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe_p = sub.add_parser("transcribe", help="音声を文字起こし")
    transcribe_p.add_argument("audio", help="入力音声ファイルのパス")
    transcribe_p.add_argument("--reference", help="参考資料ファイルのパス（Whisperのみ）")
    transcribe_p.add_argument(
        "--model",
        default="gpt-4o-transcribe-diarize",
        choices=["gpt-4o-transcribe-diarize", "whisper"],
        help="使用するモデル",
    )
    transcribe_p.add_argument("--output", help="結果の保存先 (csv または json)")
    transcribe_p.add_argument("--summary-model", help="要約に使用するモデル")
    transcribe_p.add_argument("--summary-prompt", help="要約用プロンプト")
    transcribe_p.set_defaults(func=_handle_transcribe)

    summary_p = sub.add_parser("summary", help="文字起こし結果を要約")
    summary_p.add_argument("input", help="文字起こし結果 (csv/json)")
    summary_p.add_argument("--model", required=True, help="使用するモデル")
    summary_p.add_argument("--prompt", required=True, help="要約プロンプト")
    summary_p.set_defaults(func=_handle_summary)

    video_p = sub.add_parser("video-to-audio", help="動画から音声を抽出")
    video_p.add_argument("video", help="入力動画のパス")
    video_p.add_argument("--output", help="出力MP3のパス")
    video_p.set_defaults(func=_handle_video_to_audio)

    return parser


def main(argv: Optional[Iterable[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
