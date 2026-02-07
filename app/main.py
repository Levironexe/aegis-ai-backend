from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import auth, chat, documents, vote, api_compat, files
from app.database import engine, Base

app = FastAPI(title=settings.app_name, debug=settings.debug)

# CORS configuration - allow both localhost and production URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://localhost:3001",  # Next.js sometimes uses 3001
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(vote.router)
app.include_router(files.router)
app.include_router(api_compat.router)  # Compatibility routes


@app.on_event("startup")
async def startup():
    """Initialize database tables on startup"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
