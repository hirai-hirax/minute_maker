# Minute Maker

A starter project that pairs a FastAPI backend with a Vite + React + TypeScript frontend for capturing meeting minutes.

## Getting started

### Backend (FastAPI)
1. Create a virtual environment (optional) and install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run the API locally:
   ```bash
   uvicorn backend.app.main:app --reload
   ```
3. The service exposes a health endpoint at `/` and meeting minutes routes under `/api/minutes`.

### Frontend (Vite + React + TypeScript)
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev -- --host
   ```
3. The app reads the API base from the `VITE_API_BASE` environment variable. By default it points to `http://localhost:8000`.

### Building for production
- Frontend: `npm run build`
- Backend: Deploy the FastAPI app with any ASGI server (e.g., `uvicorn backend.app.main:app`).

## Project structure
```
backend/
  app/main.py         # FastAPI application and in-memory data
  requirements.txt    # Python dependencies
frontend/
  public/             # Static assets
  src/                # React application
  package.json        # Node dependencies and scripts
```
