# Aegis AI Backend

FastAPI backend with Google OAuth authentication.

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Run the server**:
```bash
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### Authentication
- `GET /auth/google/login` - Initiate Google OAuth
- `GET /auth/google/callback` - OAuth callback
- `POST /auth/establish-session` - Establish session
- `GET /auth/status` - Check auth status
- `GET /auth/me` - Get current user

### Health
- `GET /health` - Health check

## Configuration

Required environment variables in `.env`:
- `SECRET_KEY` - Session encryption key
- `DATABASE_URL` - PostgreSQL connection string
- `GOOGLE_CLIENT_ID` - Google OAuth client ID
- `GOOGLE_CLIENT_SECRET` - Google OAuth client secret
- `GOOGLE_REDIRECT_URI` - OAuth callback URL
- `FRONTEND_URL` - Frontend application URL
