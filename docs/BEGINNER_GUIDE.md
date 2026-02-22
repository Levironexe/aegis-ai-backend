# Aegis AI Backend - Beginner's Guide
## Complete System Understanding from Broad to Depth

> **Target Audience**: Developers new to this codebase who want to understand how the entire system works
> **Last Updated**: February 2026

---

## Table of Contents

1. [Big Picture Overview](#1-big-picture-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack Explained](#3-technology-stack-explained)
4. [Directory Structure Deep Dive](#4-directory-structure-deep-dive)
5. [Core Concepts](#5-core-concepts)
6. [Request Flow Walkthrough](#6-request-flow-walkthrough)
7. [Key Components Explained](#7-key-components-explained)
8. [Database Layer](#8-database-layer)
9. [AI Integration Layer](#9-ai-integration-layer)
10. [Authentication & Security](#10-authentication--security)
11. [Development Workflow](#11-development-workflow)
12. [Common Patterns & Best Practices](#12-common-patterns--best-practices)
13. [Troubleshooting Guide](#13-troubleshooting-guide)

---

## 1. Big Picture Overview

### What is Aegis AI Backend?

Aegis AI Backend is a **cybersecurity investigation platform** powered by Large Language Models (LLMs). Think of it as a smart assistant that helps security analysts investigate threats, analyze suspicious indicators, and provide actionable recommendations.

**Core Purpose**:
- Accept user queries about cybersecurity threats
- Use AI agents to investigate and analyze security indicators (IPs, domains, file hashes)
- Integrate with multiple AI providers (Claude, Gemini)
- Provide streaming responses in real-time
- Maintain conversation history and user sessions

### The 10,000 Foot View

```
┌─────────────┐
│   User      │
│  (Browser)  │
└──────┬──────┘
       │ HTTP Requests
       ↓
┌─────────────────────────────────────┐
│   Aegis AI Backend (FastAPI)        │
│  ┌───────────────────────────────┐  │
│  │  API Routes                   │  │
│  │  /auth, /chat, /documents     │  │
│  └───────────┬───────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │  Business Logic Layer         │  │
│  │  - Authentication             │  │
│  │  - Chat Management            │  │
│  │  - AI Agent Orchestration     │  │
│  └───────────┬───────────────────┘  │
│              ↓                       │
│  ┌──────────────────┬──────────────┐│
│  │  AI Layer        │ Data Layer   ││
│  │  - Claude API    │ - PostgreSQL ││
│  │  - Gemini API    │ - SQLAlchemy ││
│  │  - LangGraph     │              ││
│  └──────────────────┴──────────────┘│
└─────────────────────────────────────┘
```

### Key Capabilities

1. **Multi-Model AI Chat**: Support for Claude (Anthropic) and Gemini (Google)
2. **LangGraph Agent**: Multi-step reasoning agent for security investigations
3. **Tool Orchestration**: Automated execution of investigation tools
4. **Real-time Streaming**: Server-Sent Events (SSE) for live responses
5. **User Management**: OAuth authentication with Google
6. **Conversation History**: Persistent chat storage
7. **Document Management**: Upload and manage security-related documents

---

## 2. System Architecture

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                       FRONTEND                                 │
│              (Next.js / React - Not covered here)              │
└───────────────────────────┬────────────────────────────────────┘
                            │
                    HTTP/HTTPS (REST API)
                            │
┌───────────────────────────┴────────────────────────────────────┐
│                    AEGIS AI BACKEND                            │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    PRESENTATION LAYER                     │ │
│  │                        (FastAPI)                          │ │
│  │  ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌──────┐│ │
│  │  │  Auth  │ │  Chat  │ │Documents │ │ Files  │ │ Vote ││ │
│  │  │ Router │ │ Router │ │  Router  │ │ Router │ │Router││ │
│  │  └────────┘ └────────┘ └──────────┘ └────────┘ └──────┘│ │
│  └────────────────────────┬─────────────────────────────────┘ │
│                           │                                   │
│  ┌────────────────────────┴─────────────────────────────────┐ │
│  │                   BUSINESS LOGIC LAYER                    │ │
│  │                                                            │ │
│  │  ┌──────────────────┐     ┌──────────────────┐           │ │
│  │  │  Auth Services   │     │  Token Services  │           │ │
│  │  │  - OAuth Flow    │     │  - JWT Creation  │           │ │
│  │  │  - Session Mgmt  │     │  - Validation    │           │ │
│  │  └──────────────────┘     └──────────────────┘           │ │
│  │                                                            │ │
│  │  ┌────────────────────────────────────────────┐          │ │
│  │  │         AI Gateway Client                  │          │ │
│  │  │  Routes requests to appropriate AI backend │          │ │
│  │  └───┬─────────────────────┬──────────────────┘          │ │
│  │      │                     │                              │ │
│  │  ┌───┴─────┐ ┌─────────┴────┐ ┌───────────────┐         │ │
│  │  │ Claude  │ │    Gemini    │ │  LangGraph    │         │ │
│  │  │ Client  │ │    Client    │ │    Agent      │         │ │
│  │  └─────────┘ └──────────────┘ └───────┬───────┘         │ │
│  │                                        │                  │ │
│  │                              ┌─────────┴─────────┐        │ │
│  │                              │   Tool System     │        │ │
│  │                              │  - IOC Analysis   │        │ │
│  │                              │  - Weather, etc   │        │ │
│  │                              └───────────────────┘        │ │
│  └────────────────────────────────────────────────────────── │ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                      DATA LAYER                           │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐    │ │
│  │  │ SQLAlchemy   │  │   Models    │  │   Schemas    │    │ │
│  │  │   (ORM)      │  │ - User      │  │  (Pydantic)  │    │ │
│  │  │              │  │ - Chat      │  │              │    │ │
│  │  │              │  │ - Message   │  │              │    │ │
│  │  │              │  │ - Document  │  │              │    │ │
│  │  └──────┬───────┘  └─────────────┘  └──────────────┘    │ │
│  └─────────┼──────────────────────────────────────────────── │ │
│            │                                                  │
└────────────┼──────────────────────────────────────────────────┘
             │
    ┌────────┴─────────┐
    │   PostgreSQL     │
    │    Database      │
    └──────────────────┘

┌─────────────────────────────────────┐
│      EXTERNAL SERVICES              │
│  ┌──────────┐  ┌──────────────────┐│
│  │ Anthropic│  │  Google Cloud    ││
│  │   API    │  │  - OAuth         ││
│  │ (Claude) │  │  - Gemini API    ││
│  └──────────┘  └──────────────────┘│
└─────────────────────────────────────┘
```

### Architectural Principles

1. **Layered Architecture**: Clean separation between presentation, business logic, and data
2. **Dependency Injection**: FastAPI's dependency system for database sessions, authentication
3. **Gateway Pattern**: Single entry point (`gateway_client`) for multiple AI providers
4. **Factory Pattern**: Tool creation and registration
5. **Repository Pattern**: Database access through SQLAlchemy ORM
6. **Observer Pattern**: SSE streaming for real-time updates

---

## 3. Technology Stack Explained

### Core Framework: **FastAPI**

**What is it?**
FastAPI is a modern, high-performance Python web framework for building APIs.

**Why FastAPI?**
- **Async Support**: Built on `asyncio` for handling concurrent requests efficiently
- **Automatic Documentation**: Auto-generates OpenAPI (Swagger) docs
- **Type Safety**: Uses Python type hints for validation (via Pydantic)
- **Performance**: One of the fastest Python frameworks (comparable to Node.js/Go)

**Example from our code**:
```python
# app/main.py
from fastapi import FastAPI

app = FastAPI(title="Aegis AI Backend")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

### Database: **PostgreSQL + SQLAlchemy**

**PostgreSQL**: Industry-standard relational database
**SQLAlchemy**: Python ORM (Object-Relational Mapping) framework

**Why this combo?**
- **ACID Compliance**: Guaranteed data consistency
- **Async Support**: SQLAlchemy 2.0+ supports async operations
- **Type Safety**: Models define database schema in Python

**Example**:
```python
# app/models/user.py
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String)
```

### AI Frameworks

#### 1. **LangChain**
- Framework for building LLM applications
- Provides abstractions for prompts, chains, agents

#### 2. **LangGraph**
- Extension of LangChain for building stateful agents
- Directed graph structure for multi-step reasoning
- Used for our cybersecurity investigation agent

#### 3. **Anthropic SDK** (Claude)
- Official Python SDK for Claude API
- Streaming support via SSE

#### 4. **Google AI SDK** (Gemini)
- Official SDK for Gemini models

### Supporting Libraries

| Library | Purpose |
|---------|---------|
| `pydantic` | Data validation using Python type hints |
| `uvicorn` | ASGI server for running FastAPI |
| `python-jose` | JWT token creation/validation |
| `httpx` | Async HTTP client |
| `alembic` | Database migrations (not yet implemented) |

---

## 4. Directory Structure Deep Dive

```
aegis-ai-backend/
│
├── app/                          # Main application package
│   │
│   ├── main.py                   # 🚪 Application entry point
│   │                             # - Creates FastAPI app
│   │                             # - Configures CORS
│   │                             # - Registers routers
│   │                             # - Initializes database
│   │
│   ├── config.py                 # ⚙️ Configuration management
│   │                             # - Environment variables
│   │                             # - API keys
│   │                             # - Feature flags
│   │
│   ├── database.py               # 🗄️ Database setup
│   │                             # - SQLAlchemy engine
│   │                             # - Session management
│   │                             # - Base model class
│   │
│   ├── routers/                  # 🛣️ API route handlers
│   │   ├── auth.py               # Authentication endpoints
│   │   ├── chat.py               # Chat/messaging endpoints
│   │   ├── documents.py          # Document management
│   │   ├── files.py              # File upload/download
│   │   ├── vote.py               # Message voting (thumbs up/down)
│   │   └── api_compat.py         # Backward compatibility routes
│   │
│   ├── models/                   # 📊 Database models (SQLAlchemy)
│   │   ├── user.py               # User table definition
│   │   ├── chat.py               # Chat & Message tables
│   │   └── document.py           # Document table
│   │
│   ├── schemas/                  # 📋 Request/Response schemas (Pydantic)
│   │   ├── auth.py               # Auth request/response models
│   │   ├── chat.py               # Chat message schemas
│   │   └── document.py           # Document schemas
│   │
│   ├── services/                 # 🔧 Business logic services
│   │   ├── google_auth_service.py  # OAuth flow implementation
│   │   └── token_service.py        # JWT token management
│   │
│   ├── ai/                       # 🤖 AI integration layer
│   │   ├── gateway_client.py     # Routes requests to AI providers
│   │   ├── langgraph_agent.py    # Multi-step reasoning agent
│   │   └── llms/                 # LLM client implementations
│   │       ├── claude_client.py  # Anthropic Claude integration
│   │       └── gemini_client.py  # Google Gemini integration
│   │
│   ├── tools/                    # 🛠️ Agent tools
│   │   ├── base.py               # Base tool class
│   │   ├── example_ioc_tool.py   # IOC analysis tool
│   │   └── weather.py            # Example weather tool
│   │
│   └── utils/                    # 🔨 Utility functions
│       └── session.py            # Session management helpers
│
├── docs/                         # 📚 Documentation
│   └── BEGINNER_GUIDE.md         # This file!
│
├── .env                          # 🔐 Environment variables (not in git)
├── .env.example                  # Example environment file
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # Quick start guide
```

### File Naming Conventions

- **`*_client.py`**: Clients for external APIs (claude_client, gemini_client)
- **`*_service.py`**: Business logic services (google_auth_service, token_service)
- **`*_tool.py`**: Agent tools (example_ioc_tool)
- **`schemas/*.py`**: Pydantic models for API validation
- **`models/*.py`**: SQLAlchemy models for database tables
- **`routers/*.py`**: FastAPI route handlers (like controllers in MVC)

---

## 5. Core Concepts

### Concept 1: **Routers** (Like Controllers)

In FastAPI, **routers** are collections of related API endpoints.

**Example**:
```python
# app/routers/chat.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.get("/")
async def list_chats():
    return {"chats": []}

@router.post("/")
async def create_chat():
    return {"id": "new-chat-id"}
```

**How it works**:
1. Create an `APIRouter` instance
2. Define routes using decorators (`@router.get`, `@router.post`)
3. Register router in `main.py`: `app.include_router(chat.router)`

### Concept 2: **Dependency Injection**

FastAPI uses function parameters to inject dependencies.

**Example**:
```python
from fastapi import Depends
from app.database import get_db

@router.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    # db is automatically injected
    result = await db.execute(select(User))
    return result.scalars().all()
```

**Common Dependencies**:
- `db: AsyncSession = Depends(get_db)` - Database session
- `user: User = Depends(get_current_user)` - Authenticated user

### Concept 3: **ORM (SQLAlchemy)**

ORM maps Python classes to database tables.

**Example**:
```python
# Define a model (Python class)
class Chat(Base):
    __tablename__ = "chats"

    id = Column(UUID, primary_key=True)
    title = Column(String)
    userId = Column(UUID, ForeignKey("users.id"))

# Use it
new_chat = Chat(id=uuid.uuid4(), title="My Chat", userId=user_id)
db.add(new_chat)
await db.commit()
```

### Concept 4: **Async/Await**

Python's way of handling concurrent operations.

**Synchronous (blocking)**:
```python
def slow_function():
    time.sleep(5)  # Blocks entire server
    return "done"
```

**Asynchronous (non-blocking)**:
```python
async def fast_function():
    await asyncio.sleep(5)  # Other requests can run
    return "done"
```

**Rule**: If a function is `async`, you must `await` it.

### Concept 5: **Streaming Responses (SSE)**

Server-Sent Events allow the server to push data to clients in real-time.

**Example**:
```python
from fastapi.responses import StreamingResponse

async def event_generator():
    for i in range(10):
        yield f"data: {i}\n\n"
        await asyncio.sleep(0.1)

@router.get("/stream")
async def stream():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**Output to client**:
```
data: 0

data: 1

data: 2
...
```

### Concept 6: **LangGraph State Machine**

LangGraph uses directed graphs for multi-step agent workflows.

**Example Structure**:
```python
from langgraph.graph import StateGraph

# Define state
class AgentState(TypedDict):
    messages: list
    next_action: str

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("planning", planning_function)
workflow.add_node("action", action_function)
workflow.add_node("response", response_function)

workflow.add_edge("planning", "action")
workflow.add_edge("action", "response")
workflow.set_entry_point("planning")

app = workflow.compile()
```

**Flow**:
```
START → planning → action → response → END
```

---

## 6. Request Flow Walkthrough

### Example: User sends a chat message

Let's trace what happens when a user sends: **"Analyze IP 45.142.213.100"**

#### Step 1: Frontend Request
```http
POST /api/chat HTTP/1.1
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "parts": [{"type": "text", "text": "Analyze IP 45.142.213.100"}]
    }
  ],
  "selectedChatModel": "agent/cyber-analyst"
}
```

#### Step 2: Router Receives Request
```python
# app/routers/chat.py

@router.post("/")
async def stream_chat_response(
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
```

**What happens**:
1. FastAPI validates request against `ChatRequest` schema
2. Dependency injection provides `db` session and `user` object
3. Router function executes

#### Step 3: Message Processing
```python
# Extract messages from request
message_list = chat_request.messages

# Convert to internal format
messages = []
for msg in message_list:
    content = msg.parts[0].text
    messages.append({
        "role": msg.role,
        "content": content
    })

# Add system prompt
system_prompt = "You are Aegis AI, a cybersecurity assistant..."
messages.insert(0, {"role": "system", "content": system_prompt})
```

#### Step 4: AI Gateway Routing
```python
# Get selected model
model = "agent/cyber-analyst"  # from request

# Route to appropriate backend
async for chunk in gateway_client.stream_chat_completion(
    model=model,
    messages=messages,
    temperature=0.7
):
    # Stream chunks to client
```

**Gateway Decision**:
```python
# app/ai/gateway_client.py

def get_client(self, model: str):
    provider = model.split("/")[0]  # "agent"

    if provider == "agent":
        return self.agent  # LangGraph agent
    elif provider == "claude":
        return self.claude  # Claude client
    elif provider == "gemini":
        return self.gemini  # Gemini client
```

#### Step 5: LangGraph Agent Execution

**Graph Flow**:
```
1. classify → Detects "Analyze IP" = security query
2. planning → Plans investigation approach
3. tool_selection → Detects IP pattern, forces tool usage
4. execute_tools → Runs analyze_ioc tool
5. analysis → Analyzes tool results
6. response → Generates final report
```

**Code**:
```python
# app/ai/langgraph_agent.py

async for event in self.app.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        # LLM generating text
        yield {"choices": [{"delta": {"content": chunk.content}}]}

    elif event["event"] == "on_tool_start":
        # Tool execution starting
        yield {"choices": [{"delta": {"content": "🔧 Using tool..."}}]}
```

#### Step 6: Tool Execution
```python
# app/tools/example_ioc_tool.py

async def execute(self, indicator: str, indicator_type: str):
    # Analyze the IP address
    result = {
        "indicator": "45.142.213.100",
        "reputation_score": 85,
        "threat_level": "malicious",
        "categories": ["botnet", "c2"],
        "geolocation": {"country": "Russia", "city": "Moscow"}
    }
    return result
```

#### Step 7: Analysis & Response Generation

```python
# LLM receives tool results
tool_output = "IP 45.142.213.100: Reputation 85/100, Malicious, Categories: botnet, c2"

# Generates analysis
analysis = """
THREAT INTELLIGENCE REPORT:
- IP: 45.142.213.100
- Reputation Score: 85/100 (High Risk)
- Threat Categories: Botnet, Command & Control
- Location: Moscow, Russia
- Recommendation: BLOCK IMMEDIATELY
"""
```

#### Step 8: Streaming to Client

```python
# Router yields SSE events
for chunk in response_chunks:
    yield f"data: {json.dumps({
        'type': 'text-delta',
        'delta': chunk
    })}\n\n"

# Final event
yield f"data: {json.dumps({'type': 'text-end'})}\n\n"
```

#### Step 9: Database Persistence

```python
# Save chat and messages to database
new_chat = Chat(
    id=chat_id,
    title="Security Investigation",
    userId=user.id
)
db.add(new_chat)

user_message = Message(
    chatId=chat_id,
    role="user",
    parts=[{"type": "text", "text": "Analyze IP..."}]
)
db.add(user_message)

assistant_message = Message(
    chatId=chat_id,
    role="assistant",
    parts=[{"type": "text", "text": analysis}]
)
db.add(assistant_message)

await db.commit()
```

---

## 7. Key Components Explained

### Component 1: `main.py` - Application Entry Point

**Purpose**: Creates and configures the FastAPI application.

**Key Sections**:

```python
# 1. Create FastAPI app
app = FastAPI(title="Aegis AI Backend", debug=True)

# 2. Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# 3. Register routers
app.include_router(auth.router)
app.include_router(chat.router)
# ... more routers

# 4. Startup event
@app.on_event("startup")
async def startup():
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# 5. Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

**What happens when you run `uvicorn app.main:app`**:
1. Python imports `app` from `main.py`
2. `@app.on_event("startup")` runs → creates DB tables
3. Uvicorn starts HTTP server on port 8000
4. Server listens for incoming requests
5. Requests are routed to appropriate handlers

---

### Component 2: `config.py` - Configuration Management

**Purpose**: Centralized configuration using environment variables.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    app_name: str = "Aegis AI"
    debug: bool = False

    # Database
    database_url: str

    # Authentication
    secret_key: str
    google_client_id: str
    google_client_secret: str

    # AI APIs
    anthropic_api_key: str
    google_api_key: str

    # Agent settings
    agent_enabled: bool = True
    agent_model: str = "claude-haiku-4-5"

    class Config:
        env_file = ".env"  # Load from .env file

# Create singleton instance
settings = Settings()
```

**Usage**:
```python
from app.config import settings

# Access configuration
api_key = settings.anthropic_api_key
model = settings.agent_model
```

**Environment File (`.env`)**:
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/aegis
SECRET_KEY=your-secret-key-here
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=xxx
```

---

### Component 3: `database.py` - Database Setup

**Purpose**: Configure SQLAlchemy engine and session management.

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,  # Log SQL queries in debug mode
    future=True
)

# 2. Create session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 3. Base class for models
Base = declarative_base()

# 4. Dependency for routes
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

**How it's used**:
```python
@router.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    # 'db' is automatically injected by FastAPI
    result = await db.execute(select(User))
    return result.scalars().all()
```

---

### Component 4: `gateway_client.py` - AI Provider Gateway

**Purpose**: Single interface to route requests to multiple AI backends.

**Pattern**: **Gateway Pattern** / **Strategy Pattern**

```python
class GatewayClient:
    def __init__(self):
        # Initialize all AI clients
        self.claude = ClaudeClient()
        self.gemini = GeminiClient()
        self.agent = LangGraphAgent()

        # Register tools with agent
        tool = ExampleIOCTool()
        self.agent.register_tools([tool.to_langchain_tool()])

    def get_client(self, model: str):
        """Route to appropriate client based on model name"""
        provider = model.split("/")[0]

        if provider == "agent":
            return self.agent
        elif provider == "claude":
            return self.claude
        elif provider == "gemini":
            return self.gemini
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def stream_chat_completion(self, model, messages, **kwargs):
        """Unified interface for all providers"""
        client = self.get_client(model)

        async for chunk in client.stream_chat_completion(
            model=model,
            messages=messages,
            **kwargs
        ):
            yield chunk

# Singleton instance
gateway_client = GatewayClient()
```

**Benefits**:
1. **Single Interface**: All AI providers use same method signature
2. **Easy to Add Providers**: Just implement `stream_chat_completion()`
3. **Centralized Tool Registration**: Tools are registered once
4. **Model Selection**: Frontend can switch models dynamically

---

### Component 5: `langgraph_agent.py` - Multi-Step Reasoning Agent

**Purpose**: Orchestrate multi-step cybersecurity investigations using tools.

**Key Concepts**:

1. **State Definition**:
```python
class CyberSecurityState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    investigation_steps: list[str]
    iocs_found: list[dict]
    severity_level: str
    tools_used: list[str]
    tool_results: list[dict]
```

2. **Graph Structure**:
```python
workflow = StateGraph(CyberSecurityState)

# Add nodes (functions)
workflow.add_node("classify", self._classify_node)
workflow.add_node("planning", self._planning_node)
workflow.add_node("tool_selection", self._tool_selection_node)
workflow.add_node("execute_tools", self._execute_tools_node)
workflow.add_node("analysis", self._analysis_node)
workflow.add_node("response", self._response_node)

# Define edges (flow)
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", self._is_security_query, {
    "security": "planning",
    "general": "simple_response"
})
workflow.add_edge("planning", "tool_selection")
# ... more edges
```

3. **Node Implementation**:
```python
async def _tool_selection_node(self, state: CyberSecurityState):
    """Select appropriate tools based on query"""
    messages = state["messages"]

    # Bind tools to LLM
    llm_with_tools = self.llm.bind_tools(self.tools)

    # LLM decides which tools to call
    response = await llm_with_tools.ainvoke(messages)

    # Return updated state
    return {
        **state,
        "messages": state["messages"] + [response]
    }
```

4. **Tool Execution**:
```python
async def _execute_tools_node(self, state: CyberSecurityState):
    """Execute tools selected by LLM"""
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls'):
        # Use ToolNode to execute all tool calls
        result = await self.tool_node.ainvoke(state)
        return result

    return state
```

**Execution Flow**:
```
User Query: "Analyze IP 45.142.213.100"
    ↓
[classify] → Detects security keywords → "security"
    ↓
[planning] → Creates investigation plan
    ↓
[tool_selection] → Detects IP pattern → Calls analyze_ioc
    ↓
[execute_tools] → Runs IOC analysis tool
    ↓
[analysis] → Correlates findings, maps to MITRE ATT&CK
    ↓
[response] → Generates formatted report
    ↓
Return to user
```

---

### Component 6: Tool System

**Architecture**:

```python
# Base Tool Class
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, **kwargs):
        """Execute the tool"""
        pass

    def to_langchain_tool(self):
        """Convert to LangChain format"""
        return StructuredTool.from_function(
            coroutine=self.execute,  # async function
            name=self.name,
            description=self.description,
            args_schema=self.input_schema
        )
```

**Example Tool**:
```python
class ExampleIOCTool(BaseTool):
    @property
    def description(self) -> str:
        return "Analyzes IP addresses, domains, file hashes, and URLs"

    @property
    def input_schema(self) -> type[BaseModel]:
        class IOCInput(BaseModel):
            indicator: str
            indicator_type: Literal["ip", "domain", "hash", "url"]
        return IOCInput

    async def execute(self, indicator: str, indicator_type: str):
        # Mock analysis
        return {
            "indicator": indicator,
            "reputation_score": 85,
            "threat_level": "malicious",
            "categories": ["botnet", "c2"],
            "recommendations": ["Block at firewall", "Investigate connections"]
        }
```

**Tool Registration**:
```python
# In gateway_client.py
tool = ExampleIOCTool()
self.agent.register_tools([tool.to_langchain_tool()])
```

**LLM Tool Calling**:
```python
# LLM automatically calls tool when it sees IP address
# Input: "Analyze IP 45.142.213.100"
# LLM generates:
{
    "name": "analyze_ioc",
    "args": {
        "indicator": "45.142.213.100",
        "indicator_type": "ip"
    }
}
```

---

## 8. Database Layer

### Database Models (SQLAlchemy)

#### User Model
```python
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    name = Column(String)
    picture = Column(String)  # Profile picture URL
    createdAt = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chats = relationship("Chat", back_populates="user")
```

#### Chat Model
```python
class Chat(Base):
    __tablename__ = "chats"

    id = Column(UUID(as_uuid=True), primary_key=True)
    title = Column(String, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    userId = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    visibility = Column(String, default="private")

    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")
```

#### Message Model
```python
class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True)
    chatId = Column(UUID(as_uuid=True), ForeignKey("chats.id"))
    role = Column(String, nullable=False)  # "user" or "assistant"
    parts = Column(JSON, nullable=False)  # [{"type": "text", "text": "..."}]
    attachments = Column(JSON, default=[])
    createdAt = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chat = relationship("Chat", back_populates="messages")
```

### Database Operations

#### Creating Records
```python
# Create new chat
new_chat = Chat(
    id=uuid.uuid4(),
    title="Security Investigation",
    userId=user.id
)
db.add(new_chat)
await db.commit()
await db.refresh(new_chat)  # Get updated object with DB-generated fields
```

#### Querying Records
```python
# Get all chats for a user
result = await db.execute(
    select(Chat)
    .where(Chat.userId == user_id)
    .order_by(Chat.createdAt.desc())
)
chats = result.scalars().all()

# Get chat with messages
result = await db.execute(
    select(Chat)
    .options(selectinload(Chat.messages))  # Eager load messages
    .where(Chat.id == chat_id)
)
chat = result.scalar_one_or_none()
```

#### Updating Records
```python
# Update chat title
chat.title = "New Title"
await db.commit()
```

#### Deleting Records
```python
await db.delete(chat)
await db.commit()
```

---

## 9. AI Integration Layer

### Claude Client

```python
class ClaudeClient:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key
        )

    async def stream_chat_completion(self, model, messages, **kwargs):
        """Stream responses from Claude"""
        async with self.client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=4096,
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield {
                    "choices": [{
                        "delta": {"content": text}
                    }]
                }
```

### Gemini Client

```python
class GeminiClient:
    def __init__(self):
        self.client = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            api_key=settings.google_api_key
        )

    async def stream_chat_completion(self, model, messages, **kwargs):
        """Stream responses from Gemini"""
        prompt = self._convert_messages_to_prompt(messages)

        async for chunk in self.client.generate_content_async(
            prompt,
            stream=True
        ):
            yield {
                "choices": [{
                    "delta": {"content": chunk.text}
                }]
            }
```

### LangGraph Agent

Already covered in Component 5. Key features:

1. **Multi-step reasoning**: Planning → Tool Selection → Execution → Analysis
2. **Tool integration**: Automatically detects and calls tools
3. **State management**: Maintains investigation context across steps
4. **Streaming**: Streams thinking process and results in real-time

---

## 10. Authentication & Security

### OAuth Flow (Google)

**Step-by-step process**:

1. **User clicks "Login with Google"**
```http
GET /auth/google/login
```

2. **Backend redirects to Google**
```python
@router.get("/google/login")
async def google_login():
    auth_url = google_auth_service.get_authorization_url()
    return RedirectResponse(auth_url)
```

3. **User authorizes on Google**
- User logs into Google
- Grants permissions to Aegis AI

4. **Google redirects back with code**
```http
GET /auth/google/callback?code=AUTH_CODE&state=STATE
```

5. **Backend exchanges code for tokens**
```python
@router.get("/google/callback")
async def google_callback(code: str, state: str, db: AsyncSession = Depends(get_db)):
    # Exchange code for access token
    user_info = await google_auth_service.get_user_info(code)

    # Create or get user
    user = await get_or_create_user(db, user_info)

    # Create session
    session_manager.set_session(response, "user_id", str(user.id))

    return RedirectResponse(f"{settings.frontend_url}")
```

6. **Frontend receives session cookie**
- Subsequent requests include session cookie
- Backend validates session and retrieves user

### JWT Tokens

```python
# Create token
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm="HS256"
    )
    return encoded_jwt

# Verify token
def verify_token(token: str):
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"]
        )
        return payload
    except JWTError:
        return None
```

### Session Management

```python
class SessionManager:
    def set_session(self, response: Response, key: str, value: str):
        """Set session cookie"""
        response.set_cookie(
            key=key,
            value=value,
            httponly=True,  # Prevent XSS
            secure=True,  # HTTPS only
            samesite="lax"  # CSRF protection
        )

    def get_session(self, request: Request, key: str):
        """Get session value"""
        return request.cookies.get(key)
```

### Protected Routes

```python
async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency that requires authentication"""
    user_id = session_manager.get_session(request, "user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user

# Use in routes
@router.get("/protected")
async def protected_route(user: User = Depends(get_current_user)):
    return {"message": f"Hello {user.name}"}
```

---

## 11. Development Workflow

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone <repo-url>
cd aegis-ai-backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start PostgreSQL (using Docker)
docker run -d \
  --name aegis-postgres \
  -e POSTGRES_USER=aegis \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=aegis \
  -p 5432:5432 \
  postgres:15

# 6. Run server
uvicorn app.main:app --reload --port 8000
```

### Project Commands

```bash
# Run development server (auto-reload on file changes)
uvicorn app.main:app --reload

# Run with specific host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run in production mode (no reload)
uvicorn app.main:app --workers 4

# Install new dependency
pip install package-name
pip freeze > requirements.txt

# Database migrations (when using Alembic)
alembic init alembic
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

### Testing API Endpoints

Using `curl`:
```bash
# Health check
curl http://localhost:8000/health

# Create chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "parts": [{"type": "text", "text": "Hello"}]
      }
    ],
    "selectedChatModel": "claude/claude-3-5-sonnet-20241022"
  }'
```

Using Python:
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/chat",
        json={
            "messages": [{"role": "user", "parts": [{"type": "text", "text": "Hello"}]}],
            "selectedChatModel": "claude/claude-3-5-sonnet-20241022"
        }
    )
    print(response.json())
```

### Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()

# Logging
import logging

logger = logging.getLogger(__name__)
logger.info("Info message")
logger.debug("Debug message")
logger.error("Error message")
```

---

## 12. Common Patterns & Best Practices

### Pattern 1: Error Handling

```python
from fastapi import HTTPException

@router.get("/users/{user_id}")
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return user

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Pattern 2: Request Validation (Pydantic)

```python
from pydantic import BaseModel, validator

class CreateChatRequest(BaseModel):
    title: str
    visibility: str = "private"

    @validator("title")
    def title_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()

    @validator("visibility")
    def visibility_must_be_valid(cls, v):
        if v not in ["private", "public", "shared"]:
            raise ValueError("Invalid visibility option")
        return v

# Use in route
@router.post("/")
async def create_chat(request: CreateChatRequest):
    # Request is automatically validated
    # If validation fails, FastAPI returns 422 error
    return {"title": request.title}
```

### Pattern 3: Response Models

```python
from pydantic import BaseModel

class UserResponse(BaseModel):
    id: UUID
    email: str
    name: str

    class Config:
        from_attributes = True  # Allow SQLAlchemy models

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    user = await fetch_user(db, user_id)
    # FastAPI automatically converts to UserResponse
    # Excludes fields not in response model (like password)
    return user
```

### Pattern 4: Background Tasks

```python
from fastapi import BackgroundTasks

def send_email(to: str, subject: str, body: str):
    # Simulate sending email
    time.sleep(2)
    print(f"Email sent to {to}")

@router.post("/send")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    # Add task to run in background
    background_tasks.add_task(send_email, email, "Welcome", "Hello!")

    # Return immediately
    return {"message": "Email queued"}
```

### Pattern 5: Transaction Management

```python
@router.post("/transfer")
async def transfer_funds(
    from_id: str,
    to_id: str,
    amount: float,
    db: AsyncSession = Depends(get_db)
):
    try:
        # Start transaction
        async with db.begin():
            # Deduct from sender
            sender = await db.get(Account, from_id)
            sender.balance -= amount

            # Add to recipient
            recipient = await db.get(Account, to_id)
            recipient.balance += amount

            # Both updates commit together
            # If any fails, entire transaction rolls back

        return {"status": "success"}

    except Exception as e:
        # Transaction automatically rolled back
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 13. Troubleshooting Guide

### Common Issues

#### Issue 1: Database Connection Errors

**Symptom**:
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions**:
1. Check PostgreSQL is running: `docker ps`
2. Verify `DATABASE_URL` in `.env`
3. Test connection: `psql -U aegis -d aegis`

#### Issue 2: Import Errors

**Symptom**:
```
ModuleNotFoundError: No module named 'anthropic'
```

**Solutions**:
1. Activate virtual environment: `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (need 3.10+)

#### Issue 3: CORS Errors

**Symptom**:
```
Access to fetch at 'http://localhost:8000/api/chat' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Solutions**:
1. Check `FRONTEND_URL` in `.env`
2. Verify CORS middleware in `main.py` includes frontend URL
3. Ensure `allow_credentials=True` for cookie support

#### Issue 4: LangGraph Tool Not Executing

**Symptom**:
```
RuntimeWarning: coroutine 'Tool.execute' was never awaited
```

**Solutions**:
1. Check `base.py`: Use `coroutine=self.execute` not `func=self.execute`
2. Ensure tool method is `async def execute(...)`

#### Issue 5: Streaming Not Working

**Symptom**:
- No real-time updates
- Response arrives all at once

**Solutions**:
1. Check `media_type="text/event-stream"` in `StreamingResponse`
2. Verify `yield` statements in generator
3. Ensure frontend handles SSE properly

### Debugging Checklist

- [ ] Virtual environment activated?
- [ ] All dependencies installed?
- [ ] `.env` file configured?
- [ ] Database running and accessible?
- [ ] API keys valid?
- [ ] Correct Python version (3.10+)?
- [ ] Port 8000 not already in use?
- [ ] Logs showing errors? (`uvicorn ... --log-level debug`)

### Useful Debugging Commands

```bash
# Check what's listening on port 8000
lsof -i :8000

# Kill process on port 8000
kill $(lsof -t -i:8000)

# View database tables
psql -U aegis -d aegis -c "\dt"

# View recent logs (if using systemd)
journalctl -u aegis-backend -n 100

# Test database connection
python -c "from app.database import engine; import asyncio; asyncio.run(engine.connect())"
```

---

## Next Steps for Learning

### Beginner Level ✅
- [x] Understand big picture architecture
- [x] Know what each directory contains
- [x] Follow a request through the system
- [x] Understand basic FastAPI concepts

### Intermediate Level 🎯
- [ ] Modify existing routes
- [ ] Create new database models
- [ ] Add new API endpoints
- [ ] Write custom tools for the agent
- [ ] Implement custom authentication logic

### Advanced Level 🚀
- [ ] Add new AI provider to gateway
- [ ] Implement database migrations with Alembic
- [ ] Create complex LangGraph workflows
- [ ] Add caching layer (Redis)
- [ ] Implement rate limiting
- [ ] Add comprehensive testing (pytest)
- [ ] Set up CI/CD pipeline

---

## Additional Resources

### Official Documentation
- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
- [Pydantic](https://docs.pydantic.dev/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)

### Related Docs in this Repo
- [`README.md`](../README.md) - Quick start guide
- [`LANGGRAPH_INTEGRATION.md`](LANGGRAPH_INTEGRATION.md) - LangGraph integration details

### Learning Path
1. **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
2. **SQLAlchemy Async Tutorial**: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
3. **LangChain Quickstart**: https://python.langchain.com/docs/get_started/quickstart
4. **LangGraph Tutorial**: https://langchain-ai.github.io/langgraph/tutorials/

---

## Glossary

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface - how programs communicate |
| **Async/Await** | Python syntax for non-blocking operations |
| **CORS** | Cross-Origin Resource Sharing - allows frontend to call backend |
| **Dependency Injection** | Automatic provision of dependencies to functions |
| **ORM** | Object-Relational Mapping - database tables as Python classes |
| **Pydantic** | Data validation library using Python type hints |
| **Router** | Collection of related API endpoints |
| **SQLAlchemy** | Python ORM framework |
| **SSE** | Server-Sent Events - one-way real-time updates |
| **UUID** | Universally Unique Identifier - unique ID format |
| **LangGraph** | Framework for building stateful AI agents |
| **LLM** | Large Language Model (Claude, Gemini, GPT) |
| **IOC** | Indicator of Compromise (IP, domain, hash, URL) |
| **MITRE ATT&CK** | Knowledge base of adversary tactics and techniques |

---

## Conclusion

You now have a comprehensive understanding of the Aegis AI Backend from broad architecture to deep implementation details. The system is designed with:

1. **Clean Architecture**: Separation of concerns across layers
2. **Scalability**: Async operations and streaming for performance
3. **Extensibility**: Easy to add new AI providers, tools, and features
4. **Security**: OAuth authentication and session management
5. **Developer Experience**: Type safety, auto-docs, and clear patterns

Start experimenting with the code, make small changes, and watch how the system responds. The best way to learn is by doing!

**Happy Coding! 🚀**
