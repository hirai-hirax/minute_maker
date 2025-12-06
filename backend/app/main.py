from uuid import uuid4
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os
import logging
import tempfile
import shutil
import json
from pathlib import Path
from io import BytesIO
from contextlib import contextmanager

from dotenv import load_dotenv
from openai import AzureOpenAI

import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from pydub import AudioSegment

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-03-01-preview"

# Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SPEAKERS_DIR = BASE_DIR / "data" / "speakers"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_DATA = {
    "standard": {
        "name": "標準校正",
        "description": "バランスの取れた標準的な要約・校正",
        "system_prompt": "あなたは議事録作成の専門家です。以下の会議の文字起こしテキストを読み、要約、決定事項、アクションアイテムを抽出してください。\n\n出力はJSON形式で、以下のキーを含めてください:\n- summary: 会議の要約\n- decisions: 決定事項のリスト\n- action_items: アクションアイテムのリスト"
    },
    "detailed": {
        "name": "詳細",
        "description": "詳細な分析と背景情報を含む要約",
        "system_prompt": "あなたは議事録作成の専門家です。以下の文字起こしを詳細に分析し、背景情報、議論の経緯を含めて要約してください。\n\n出力はJSON形式で、以下のキーを含めてください:\n- summary: 詳細な要約（背景含む）\n- decisions: 決定事項（経緯含む）\n- action_items: アクションアイテム（担当者明確化）"
    },
    "simple": {
        "name": "簡潔",
        "description": "要点のみを箇条書きで",
        "system_prompt": "あなたは議事録作成の専門家です。以下の文字起こしから、要点のみを極めて簡潔に抽出してください。\n\n出力はJSON形式で、以下のキーを含めてください:\n- summary: 超簡潔な要約（3行以内）\n- decisions: 決定事項の箇条書き\n- action_items: アクションアイテムの箇条書き"
    }
}


CHAT_MODEL_NAME = "gpt-4o"  # Default to gpt-4o for chat operations


class MinuteBase(BaseModel):
    title: str = Field(..., description="Meeting or section title")
    summary: str = Field(..., description="Concise summary of the discussion")
    decisions: List[str] = Field(default_factory=list, description="Key decisions made")
    action_items: List[str] = Field(
        default_factory=list, description="Follow-up action items for attendees"
    )


class Minute(MinuteBase):
    id: str


app = FastAPI(title="Minute Maker API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_minutes: List[Minute] = [
    Minute(
        id=str(uuid4()),
        title="Project Kickoff",
        summary="Introduced project goals, timelines, and stakeholder expectations.",
        decisions=[
            "Adopt FastAPI for backend services",
            "Use React with Vite and TypeScript for the frontend",
        ],
        action_items=[
            "Set up repository scaffolding",
            "Draft initial UI for capturing meeting notes",
        ],
    ),
    Minute(
        id=str(uuid4()),
        title="API Contract Review",
        summary="Reviewed the REST contract for the minutes API and confirmed payload shapes.",
        decisions=["Expose /api/minutes for listing and creation"],
        action_items=["Add validation rules", "Document example payloads"],
    ),
]


@app.get("/", summary="Service health")
async def root() -> dict[str, str]:
    return {"message": "Minute Maker API is running"}


@app.get("/api/minutes", response_model=List[Minute], summary="List minutes")
async def list_minutes() -> List[Minute]:
    return _minutes


@app.get(
    "/api/minutes/{minute_id}",
    response_model=Minute,
    summary="Retrieve a single set of minutes",
)
async def get_minute(minute_id: str) -> Minute:
    for minute in _minutes:
        if minute.id == minute_id:
            return minute
    raise HTTPException(status_code=404, detail="Minute not found")


@app.post(
    "/api/minutes",
    response_model=Minute,
    status_code=201,
    summary="Create new meeting minutes",
)
async def create_minute(minute: MinuteBase) -> Minute:
    new_minute = Minute(id=str(uuid4()), **minute.model_dump())
    _minutes.append(new_minute)
    return new_minute


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = ""

class ProcessingResult(BaseModel):
    id: str
    transcript: str
    segments: List[TranscriptSegment]
    summary: str
    speakers: List[str]


def _create_azure_client(api_version: str = API_VERSION) -> AzureOpenAI:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=api_version,
    )


# ---------------------------------------------------------------------------
# Speaker Identification Utilities
# ---------------------------------------------------------------------------

_SPEAKER_ENCODER = None

def load_speaker_encoder() -> EncoderClassifier:
    global _SPEAKER_ENCODER
    if _SPEAKER_ENCODER is None:
        logger.info("Loading SpeechBrain EncoderClassifier...")
        _SPEAKER_ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir="tmp_model"
        )
    return _SPEAKER_ENCODER


@contextmanager
def temp_file_path(data: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")


def _compute_embedding_from_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    with temp_file_path(wav_bytes, ".wav") as wav_path:
        # Load audio file
        waveform, sample_rate = torchaudio.load(wav_path)
    
    # Resample if necessary
    target_sr = 16000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    # Convert to single channel if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    encoder = load_speaker_encoder()
    waveform = waveform.to(dtype=torch.float32)
    
    # helper: classify_batch expects a batch, so we might need to ensure it's treated as one
    # 但是 encode_batch handles it. 
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)
    
    # embedding is (batch, 1, emb_dim) -> squeeze to (emb_dim,)
    return embedding.squeeze().cpu().numpy()


def _calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(emb1, emb2) / (norm1 * norm2)

def _apply_speaker_identification(
    audio_path: str,
    segments_data: List[dict],
    known_speakers: dict,
    threshold: float = 0.65
) -> (List[dict], set):
    """
    Returns updated segments (as dicts) and a set of all speakers found.
    """
    logger.info("Loading full audio for identification...")
    sound = AudioSegment.from_file(audio_path)
    
    updated_segments = []
    all_speakers = set()
    
    logger.info(f"Processing {len(segments_data)} segments...")

    for i, seg in enumerate(segments_data):
        start_ms = int(seg.get("start", 0) * 1000)
        end_ms = int(seg.get("end", 0) * 1000)
        
        # Avoid extremely short segments
        if end_ms - start_ms < 500:
            updated_segments.append(seg)
            if seg.get("speaker"): all_speakers.add(seg.get("speaker"))
            continue

        # Extract segment audio
        segment_audio = sound[start_ms:end_ms]
        seg_io = BytesIO()
        segment_audio.export(seg_io, format="wav")
        seg_io.seek(0)
        
        # Compute Embedding
        try:
            seg_emb = _compute_embedding_from_wav_bytes(seg_io.read())
            
            # Identify
            best_spk = seg.get("speaker", "")
            best_sim = -1.0
            best_spk_candidate = None
            
            for name, ref_emb in known_speakers.items():
                sim = _calculate_cosine_similarity(seg_emb, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_spk_candidate = name
            
            if best_spk_candidate and best_sim >= threshold:
                best_spk = best_spk_candidate
            
            seg["speaker"] = best_spk
            if best_spk:
                all_speakers.add(best_spk)
            
        except Exception as e:
            logger.warning(f"Error identification segment {i}: {e}")
        
        updated_segments.append(seg)
        
    return updated_segments, all_speakers


# ---------------------------------------------------------------------------
# Transcription Logic
# ---------------------------------------------------------------------------

def _transcribe_audio(audio_path: str, original_filename: str, model: str = "gpt-4o") -> List[dict]:
    # Determine API version based on model
    api_version = API_VERSION
    if model == "whisper":
        api_version = "2024-06-01"
    
    client = _create_azure_client(api_version=api_version)
    logger.info(f"Starting transcription with model: {model}, api_version: {api_version}")
    
    # Determine mime type/extension for the API
    ext = os.path.splitext(original_filename)[1]
    if not ext:
        ext = ".mp3"
    
    with open(audio_path, "rb") as audio_file:
        if model == "whisper":
            response = client.audio.transcriptions.create(
                model="whisper",
                file=(original_filename, audio_file, f"audio/{ext.lstrip('.')}"),
                response_format="verbose_json",
            )
            # Whisper returns valid JSON with segments but NO speaker info in verbose_json
            transcript_dict = response.model_dump()
            segments_data = transcript_dict.get("segments", [])
            
        else:
            # Default to gpt-4o-transcribe-diarize
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=(original_filename, audio_file, f"audio/{ext.lstrip('.')}"),
                response_format="diarized_json",
                chunking_strategy="auto",
            )
            transcript_dict = response.model_dump()
            segments_data = transcript_dict.get("segments", [])
    
    results = []
    for seg in segments_data:
        results.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", ""), # Whisper will imply empty string here
        })
        
    return results


@app.post("/api/process_audio", response_model=ProcessingResult, summary="Process audio file")
async def process_audio(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o")  # 'gpt-4o' or 'whisper'
):
    # Generate ID and save file to uploads dir for later access
    process_id = str(uuid4())
    suffix = Path(file.filename).suffix
    saved_filename = f"{process_id}{suffix}"
    saved_path = UPLOAD_DIR / saved_filename
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    tmp_path = str(saved_path)

    try:
        # Run transcription with diarization logic
        segments_data = _transcribe_audio(tmp_path, file.filename, model)
        
        # Load system speakers
        sys_speakers = list(SPEAKERS_DIR.glob("*.npy"))
        known_speakers = {}
        if sys_speakers:
            logger.info(f"Loading {len(sys_speakers)} system speakers")
            for spk_file in sys_speakers:
                try:
                    known_speakers[spk_file.stem] = np.load(spk_file)
                except Exception as e:
                    logger.warning(f"Failed to load speaker {spk_file}: {e}")
            
            # Apply identification if we have known speakers
            if known_speakers:
                segments_data, speakers = _apply_speaker_identification(tmp_path, segments_data, known_speakers)
        
        # Convert to Pydantic models
        segments = [TranscriptSegment(**seg) for seg in segments_data]
        
        # Construct full transcript text
        full_transcript = ""
        speakers = set()
        for seg in segments:
            speaker_label = f"[{seg.speaker}] " if seg.speaker else ""
            full_transcript += f"[{seg.start:.2f} - {seg.end:.2f}] {speaker_label}{seg.text}\n"
            if seg.speaker:
                speakers.add(seg.speaker)
        
        return ProcessingResult(
            id=process_id,
            transcript=full_transcript.strip(),
            segments=segments,
            summary="Summary generation not yet implemented.",
            speakers=sorted(list(speakers))
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        # Clean up file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    # We do NOT delete the file in finally block anymore, to allow subsequent operations like speaker registration



@app.post("/api/create_speaker_embedding", summary="Create speaker embedding .npy file")
async def create_speaker_embedding(file: UploadFile = File(...)):
    content = await file.read()
    try:
        embedding = _compute_embedding_from_wav_bytes(content)
        
        bio = BytesIO()
        np.save(bio, embedding)
        bio.seek(0)
        
        filename = Path(file.filename).stem + ".npy"
        
        return StreamingResponse(
            bio, 
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SpeakerRegistration(BaseModel):
    process_id: str
    start: float
    end: float
    speaker_name: str


@app.post("/api/register_speaker", summary="Register speaker embedding from transcript segment")
async def register_speaker(data: SpeakerRegistration):
    # Find the audio file
    # We don't know the extension, so look for files starting with process_id
    files = list(UPLOAD_DIR.glob(f"{data.process_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="Audio file not found for this session")
    
    audio_path = files[0]
    
    try:
        # Load audio segment
        start_ms = int(data.start * 1000)
        end_ms = int(data.end * 1000)
        
        # Verify duration
        if end_ms - start_ms < 500:
             raise HTTPException(status_code=400, detail="Segment too short for embedding (min 0.5s)")

        sound = AudioSegment.from_file(str(audio_path))
        segment_audio = sound[start_ms:end_ms]
        
        seg_io = BytesIO()
        segment_audio.export(seg_io, format="wav")
        seg_io.seek(0)
        
        # Compute embedding
        embedding = _compute_embedding_from_wav_bytes(seg_io.read())
        
        # Save embedding
        safe_name = "".join([c for c in data.speaker_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not safe_name:
            safe_name = "unknown_speaker"
            
        save_path = SPEAKERS_DIR / f"{safe_name}.npy"
        np.save(save_path, embedding)
        
        return {"message": f"Speaker {safe_name} registered successfully", "path": str(save_path)}

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/identify_speakers", response_model=ProcessingResult, summary="Identify speakers using embeddings")
async def identify_speakers(
    audio: Optional[UploadFile] = File(None),  # Optional now, can use process_id or new upload
    process_id: Optional[str] = Form(None),
    transcript_json: str = Form(..., description="JSON string of transcript segments"),
    embedding_files: List[UploadFile] = File(default=[]),
    threshold: float = Form(0.65, description="Similarity threshold for identification"),
):
    # Parse transcript
    try:
        segments_data = json.loads(transcript_json)
        # Validate minimal structure
        if not isinstance(segments_data, list):
            raise ValueError("Transcript must be a list of segments")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid transcript JSON: {e}")

    # Load Embeddings
    known_speakers = {}
    
    # 1. Load system speakers
    sys_speakers = list(SPEAKERS_DIR.glob("*.npy"))
    logger.info(f"Loading {len(sys_speakers)} system speakers from {SPEAKERS_DIR}")
    for spk_file in sys_speakers:
        try:
            emb = np.load(spk_file)
            known_speakers[spk_file.stem] = emb
        except Exception as e:
            logger.warning(f"Failed to load system speaker {spk_file}: {e}")

    # 2. Load provided files
    for ef in embedding_files:
        try:
            content = await ef.read()
            bio = BytesIO(content)
            # Assuming strictly .npy files
            emb = np.load(bio)
            # Use filename (without extension) as speaker name
            name = Path(ef.filename).stem
            # Clean up name similar to reference code
            for delimiter in ['‗', '_']:
                if delimiter in name:
                    name = name.split(delimiter)[0]
                    break
            known_speakers[name] = emb
        except Exception as e:
            logger.warning(f"Failed to load embedding {ef.filename}: {e}")

    if not known_speakers:
        # If no valid embeddings, just return original
        # Reconstruct full transcript
        full_transcript = ""
        segments = []
        speakers = set()
        for seg in segments_data:
            s = TranscriptSegment(**seg)
            segments.append(s)
            speaker_label = f"[{s.speaker}] " if s.speaker else ""
            full_transcript += f"[{s.start:.2f} - {s.end:.2f}] {speaker_label}{s.text}\n"
            if s.speaker:
                speakers.add(s.speaker)
        
        return ProcessingResult(
            id=str(uuid4()),
            transcript=full_transcript.strip(),
            segments=segments,
            summary="No valid embeddings provided.",
            speakers=sorted(list(speakers))
        )

    # Prepare Audio Path
    tmp_path = None
    should_cleanup = False
    
    if audio:
        suffix = Path(audio.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(audio.file, tmp_file)
            tmp_path = tmp_file.name
        should_cleanup = True
    elif process_id:
        files = list(UPLOAD_DIR.glob(f"{process_id}.*"))
        if files:
            tmp_path = str(files[0])
            should_cleanup = False # Don't delete stored upload
        else:
            raise HTTPException(status_code=404, detail="Process ID not found")
    else:
        raise HTTPException(status_code=400, detail="Either audio file or process_id must be provided")

    try:
        updated_segments, all_speakers = _apply_speaker_identification(tmp_path, segments_data, known_speakers, threshold)
        
        # Reconstruct Response
        segments_models = [TranscriptSegment(**s) for s in updated_segments]
        full_transcript = ""
        for s in segments_models:
            speaker_label = f"[{s.speaker}] " if s.speaker else ""
            full_transcript += f"[{s.start:.2f} - {s.end:.2f}] {speaker_label}{s.text}\n"
        
        return ProcessingResult(
            id=str(uuid4()),
            transcript=full_transcript.strip(),
            segments=segments_models,
            summary="Speaker identification applied.",
            speakers=sorted(list(all_speakers))
        )
        
    finally:
        if should_cleanup and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class SummarizeRequest(BaseModel):
    transcript: str
    prompt_id: str = "standard"

class SummarizeResponse(BaseModel):
    summary: str
    action_items: List[str]
    decisions: List[str]

@app.get("/api/prompts", summary="Get available prompt presets")
async def get_prompts():
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in PROMPTS_DATA.items()
    ]

@app.post("/api/generate_summary", response_model=SummarizeResponse, summary="Generate summary from transcript")
async def generate_summary(req: SummarizeRequest):
    client = _create_azure_client()
    
    prompt_config = PROMPTS_DATA.get(req.prompt_id, PROMPTS_DATA["standard"])
    system_prompt = prompt_config["system_prompt"]
    
    user_prompt = f"""
    Transcript:
    {req.transcript}
    """
    
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = completion.choices[0].message.content

        data = json.loads(content)
        
        return SummarizeResponse(
            summary=data.get("summary", ""),
            action_items=data.get("action_items", []),
            decisions=data.get("decisions", [])
        )
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # Fallback if JSON parsing fails or model fails
        return SummarizeResponse(
            summary="Failed to generate summary.",
            action_items=[],
            decisions=[]
        )



@app.get("/api/minutes/{minute_id}/download", summary="Download minutes as file")
async def download_minutes(minute_id: str, format: str = "docx"):
    # Placeholder for file generation logic
    # Should generate .docx or .xlsx based on 'format'
    
    # For now, return a simple text response or file
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(f"Content for minute {minute_id} in format {format}")

