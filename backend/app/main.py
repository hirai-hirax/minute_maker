from uuid import uuid4
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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


class ProcessingResult(BaseModel):
    id: str
    transcript: str
    summary: str
    speakers: List[str]


@app.post("/api/process_audio", response_model=ProcessingResult, summary="Process audio file")
async def process_audio(file: UploadFile = File(...)):
    # Placeholder for actual processing logic:
    # 1. Save file temporarily
    # 2. Transcribe (Whisper/SpeechBrain)
    # 3. Diarize (SpeechBrain)
    # 4. Summarize (AOAI GPT)
    
    # Mock response for frontend development
    import time
    time.sleep(2)  # Simulate processing time
    
    return ProcessingResult(
        id=str(uuid4()),
        transcript="[00:00:00] Speaker A: This is a simulated transcript.\n[00:00:05] Speaker B: It works!",
        summary="The meeting was successfully processed. This is a mock summary.",
        speakers=["Speaker A", "Speaker B"]
    )


@app.get("/api/minutes/{minute_id}/download", summary="Download minutes as file")
async def download_minutes(minute_id: str, format: str = "docx"):
    # Placeholder for file generation logic
    # Should generate .docx or .xlsx based on 'format'
    
    # For now, return a simple text response or file
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(f"Content for minute {minute_id} in format {format}")
