# AI Agents Development Log

## Current Status
- **Date**: 2025-12-05
- **Latest Feature**: Speaker Registration and Identification
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

## Pending Tasks
- [x] Implement summary generation logic (Done via Azure OpenAI).
- [ ] Implement file download generation (currently mocked).
- [ ] Add ability to manage/delete registered speakers.
- [ ] Improve diarization accuracy.
- [x] **Feature: Model Switching**: Implement switching between Whisper and Azure OpenAI models.
- [x] **Feature: Workflow Navigation**: Allow users to click on step icons ("文字起こし", "確認・編集", "要約・整形", "完了") to transition between screens.

### 2025-12-06 (Continued)
- **Feature: Model Switching**
  - Added model selection UI (GPT-4o vs Whisper) in `MinuteGenerator`.
  - Updated backend to support `whisper` model via Azure OpenAI (API version 2024-06-01).
- **Feature: Workflow Navigation**
  - Implemented clickable progress steps to jump between "Review", "Summarize Setup", and "Completed" screens.
  - Added state validation logic to prevent invalid navigation.
