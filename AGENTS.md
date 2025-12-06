# AI Agents Development Log

## Current Status
- **Date**: 2025-12-06
- **Latest Feature**: Navigation UI Improvements
- **Build Status**: Stable

## Recent Accomplishments

### 2025-12-05
- **Feature: Transcript Display Improvement**
  - Converted the plain text transcript display to a structured table.
  - Columns: Start Time, End Time, Speaker, Text, Action.
- **Feature: Speaker Registration**
  - Implemented backend API `/api/register_speaker`.
    - Extracts audio segment from uploaded file.
    - Generates speaker embedding using SpeechBrain.
    - Saves embedding to `data/speakers/{name}.npy`.
  - Implemented frontend UI.
    - Added "Register" button (User+ icon) to transcript rows.
    - Created a modal for inputting speaker name.
- **Feature:  Persistent Speaker Identification**
  - Updated backend to save uploaded audio files to `data/uploads`.
  - Refactored identification logic to automatically load registered speakers from `data/speakers`.
  - Applied identification to new audio processing requests (`/api/process_audio`).

### 2025-12-06
- **Feature: Model Switching**
  - Added model selection UI (GPT-4o vs Whisper) in `MinuteGenerator`.
  - Updated backend to support `whisper` model via Azure OpenAI (API version 2024-06-01).
- **Feature: Workflow Navigation**
  - Implemented clickable progress steps to jump between "Review", "Summarize Setup", and "Completed" screens.
  - Added state validation logic to prevent invalid navigation.
- **Enhancement: UX & Reset Flow**
  - Added functionality to reset the application (return to file upload) by clicking the "Transcription" step or the App Logo.
  - Implemented confirmation dialogs to prevent accidental data loss when resetting.
  - Fixed logic to allow viewing transcription results (`reviewing` state) when clicking the "Processing" step.
- **Feature: Speaker Embedding File Generation Tool**
  - Added new section in Speaker Manager to generate speaker embeddings from audio files.
  - Implemented `/api/create_speaker_embedding` endpoint to create and download `.npy` embedding files.
  - Users can now create speaker embeddings without registering them to the system.
  - Supports various audio formats (MP3, WAV, M4A, etc.) via automatic conversion.
- **Fix: Model Download Solution**
  - Resolved 404 errors when downloading SpeechBrain model from Hugging Face.
  - Created `download_model.py` script for manual model download using `huggingface_hub`.
  - Updated `load_speaker_encoder()` to prioritize pre-downloaded model files.
  - Total model size: ~89.1 MB (embedding_model.ckpt: 79.46 MB + other files).
- **Enhancement: Navigation UI Improvements**
  - Added "議事録作成" (Minute Creation) button as the leftmost navigation item.
  - Allows users to return to the minute creation screen from management screens (話者管理/プロンプト管理/設定).
  - Implemented state preservation when switching between tabs.
  - All tab components remain mounted and use CSS display property for show/hide.
  - Users can now navigate between tabs without losing their work in progress.

## Pending Tasks
- [x] Implement summary generation logic (Done via Azure OpenAI).
- [ ] Implement file download generation (currently mocked).
- [x] Add ability to manage/delete registered speakers.
- [ ] Improve diarization accuracy.
- [x] **Feature: Model Switching**: Implement switching between Whisper and Azure OpenAI models.
- [x] **Feature: Workflow Navigation**: Allow users to click on step icons to transition between screens.
- [x] **Feature: Speaker Embedding Download**: Allow users to generate and download speaker embedding files.
- [x] **Fix: Model Download**: Resolve automatic model download issues.
