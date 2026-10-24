"""
FastAPI Main Application Entry Point
Phase 11: FastAPI Backend Integration

NLP-Based News Article Summarization and Vocabulary Learning System
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, APIRouter, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_db, AUDIO_DIR
from api.schemas import HealthResponse, RootResponse
from api.routers.summarize import router as summarize_router
from api.routers.vocabulary import router as vocabulary_router
from api.routers.history import router as history_router
from api.routers.notes import router as notes_router
from api.routers.speech import router as speech_router
from api.routers.evaluate import router as evaluate_router
from api.routers.auth import router as auth_router
from database.notes_history import NotesHistoryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure necessary directories exist
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown logic if needed


app = FastAPI(
    title="NLP-Based News Article Summarization & Vocabulary Learning API",
    description=(
        "Production-ready classical NLP backend exposing extractive summarization "
        "(Frequency, TF-IDF, TextRank), rule-based vocabulary extraction, new-word detection, "
        "SQLite notes and article history persistence, offline text-to-speech synthesis, "
        "and ROUGE quantitative evaluation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for future frontend integration (e.g. React in Phase 13)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 Router
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(summarize_router)
api_v1_router.include_router(vocabulary_router)
api_v1_router.include_router(history_router)
api_v1_router.include_router(notes_router)
api_v1_router.include_router(speech_router)
api_v1_router.include_router(evaluate_router)
api_v1_router.include_router(auth_router)

app.include_router(api_v1_router)


# Root and Health Endpoints
@app.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    tags=["Root"],
    summary="API Root Information",
)
def read_root():
    return {
        "name": "NLP News Summarization & Vocabulary Learning API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "summarize": "/api/v1/summarize",
            "vocabulary_extract": "/api/v1/vocabulary/extract",
            "vocabulary_detect_new": "/api/v1/vocabulary/detect-new",
            "history": "/api/v1/history",
            "notes": "/api/v1/notes",
            "speech_synthesize": "/api/v1/speech/synthesize",
            "speech_voices": "/api/v1/speech/voices",
            "evaluate": "/api/v1/evaluate",
            "auth_register": "/api/v1/auth/register",
            "auth_login": "/api/v1/auth/login",
            "health": "/health",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="System Health & Database Connectivity Check",
)
def health_check(db: NotesHistoryManager = Depends(get_db)):
    db_connected = False
    try:
        # Check database connectivity
        conn = db.get_connection()
        cursor = conn.execute("SELECT 1;")
        res = cursor.fetchone()
        if res and res[0] == 1:
            db_connected = True
        if not db.is_memory:
            conn.close()
    except Exception:
        db_connected = False

    return {
        "status": "healthy" if db_connected else "degraded",
        "database_connected": db_connected,
        "version": "1.0.0",
        "engine": "classical_nlp_fastapi",
    }
