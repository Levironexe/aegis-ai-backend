# RAG Integration Guide for Aegis AI Backend

## Table of Contents
1. [Overview](#overview)
2. [Architecture Decision](#architecture-decision)
3. [Prerequisites](#prerequisites)
4. [Implementation Phases](#implementation-phases)
5. [Technical Components](#technical-components)
6. [Code Examples](#code-examples)
7. [Knowledge Base Management](#knowledge-base-management)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Future Enhancements](#future-enhancements)

---

## Overview

### What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances LLM responses by retrieving relevant information from a knowledge base before generating answers. This reduces hallucinations and grounds responses in verified data.

### Why RAG for Aegis AI?

Your cybersecurity agent currently relies solely on:
- External APIs (IOC analysis tools)
- LLM's pre-trained knowledge (may be outdated)

**With RAG**, the agent gains access to:
- **MITRE ATT&CK Framework** - 700+ documented attack techniques
- **Threat Intelligence Reports** - Historical APT campaigns and IOCs
- **Incident Response Playbooks** - Proven investigation procedures
- **CVE Database** - Vulnerability context and exploitation details

### Benefits

✅ **Grounded Responses** - All investigations backed by verified threat intelligence
✅ **Reduced Hallucinations** - Facts from trusted sources, not speculation
✅ **Faster Analysis** - Instant access to 1000s of documented techniques
✅ **Better Recommendations** - Specific remediation steps from past incidents
✅ **Context-Aware** - Correlates findings with known TTPs and threat actors

---

## Architecture Decision

### Recommended Approach: Hybrid RAG

We recommend implementing RAG using a **hybrid architecture**:

1. **RAG as a Node** (Automatic) - Always retrieves context during planning
2. **RAG as a Tool** (Optional) - LLM can request additional context on-demand

### Current Graph Structure

```
START → classify → planning → tool_selection → execute_tools → analysis → response → END
```

### Enhanced Graph with RAG

```
START → classify → planning → threat_intel_retrieval → tool_selection → execute_tools → analysis → response → END
                                      ↓                                        ↓
                                  (RAG Node)                              (RAG Tool - optional)
```

**Key Integration Points:**

| Node | RAG Enhancement | Impact |
|------|----------------|--------|
| **planning** | Retrieves historical context for similar threats | ⭐⭐⭐⭐ |
| **threat_intel_retrieval** | **NEW NODE** - Fetches MITRE techniques and IOC context | ⭐⭐⭐⭐⭐ |
| **tool_selection** | Informed by knowledge base on which tools to use | ⭐⭐⭐⭐ |
| **analysis** | Correlates tool results with MITRE ATT&CK | ⭐⭐⭐⭐⭐ |
| **response** | Provides specific remediation from playbooks | ⭐⭐⭐⭐ |

### Vector Database Choice: pgvector

**Decision: Use pgvector extension in existing PostgreSQL**

**Why pgvector?**
- ✅ Already using PostgreSQL/Supabase - no new infrastructure
- ✅ Cost-effective - free extension vs. $500+/month for managed vector DBs
- ✅ ACID transactions - consistency with user/chat data
- ✅ Sufficient performance - handles <50M vectors with <100ms latency
- ✅ Native LangChain support - `langchain-postgres` package

**Performance Comparison:**

| Database | Query Speed | Latency (p95) | Best For | Cost |
|----------|-------------|---------------|----------|------|
| **pgvector** | 3,000 ops/sec | ~50ms | <50M vectors, existing PostgreSQL | Free |
| Pinecone | 5,000 ops/sec | 23ms | Production scale, managed | $$$ |
| Weaviate | 3,500 ops/sec | 34ms | Hybrid search, on-premise | $$ |
| Chroma | 2,000 ops/sec | 20ms | Prototyping | Free |

**When to migrate:** If vector dataset exceeds 50M entries or latency requirements drop below 30ms.

---

## Prerequisites

### 1. Dependencies

Add to `requirements.txt`:

```txt
# RAG Dependencies
langchain-postgres>=0.0.12
psycopg[binary]>=3.2.3
langchain-openai>=0.2.0
langchain-community>=0.3.0
```

Install:

```bash
pip install -r requirements.txt
```

### 2. Database Setup

Enable pgvector extension in PostgreSQL/Supabase:

```sql
-- Connect to your database
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Environment Variables

Add to `.env`:

```bash
# OpenAI API Key for embeddings (if not already present)
OPENAI_API_KEY=sk-...

# Optional: Alternative embedding providers
# VOYAGE_API_KEY=...  # Voyage AI (Anthropic partner)
```

Update `app/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # RAG Settings
    openai_api_key: str = ""  # For embeddings
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model
    embedding_dimensions: int = 1536  # Model dimension
    rag_chunk_size: int = 512  # Token size for document chunks
    rag_chunk_overlap: int = 100  # Overlap between chunks
    rag_retrieval_k: int = 5  # Number of documents to retrieve
```

---

## Implementation Phases

### Phase 1: Foundation Setup (Week 1-2)

#### 1.1 Create Database Tables

Create Alembic migration:

```bash
cd aegis-ai-backend
alembic revision --autogenerate -m "add_pgvector_tables"
```

Migration file (`alembic/versions/XXX_add_pgvector_tables.py`):

```python
def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create embeddings table
    op.execute('''
        CREATE TABLE threat_intelligence_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}',
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    ''')

    # Create indexes for performance
    op.execute('''
        CREATE INDEX idx_ti_embedding ON threat_intelligence_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    ''')

    op.execute('''
        CREATE INDEX idx_ti_metadata ON threat_intelligence_embeddings
        USING gin (metadata)
    ''')

def downgrade():
    op.drop_table('threat_intelligence_embeddings')
    op.execute('DROP EXTENSION IF EXISTS vector')
```

Run migration:

```bash
alembic upgrade head
```

#### 1.2 Create RAG Retriever Module

Create `app/ai/rag_retriever.py`:

```python
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from app.config import settings
from typing import List
import logging

logger = logging.getLogger(__name__)

class ThreatIntelligenceRetriever:
    """RAG retriever for cybersecurity threat intelligence."""

    def __init__(self):
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )

        # Initialize pgvector store
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="threat_intelligence",
            connection=settings.database_url.replace("asyncpg", "psycopg"),
            use_jsonb=True,
        )

        logger.info("ThreatIntelligenceRetriever initialized")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: dict = None
    ) -> List[Document]:
        """
        Retrieve relevant threat intelligence documents.

        Args:
            query: Search query (e.g., "ransomware TTPs", IP address)
            k: Number of documents to return
            filters: Metadata filters (e.g., {"severity": "high"})

        Returns:
            List of relevant documents with content and metadata
        """
        try:
            if filters:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)

            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def retrieve_with_scores(self, query: str, k: int = 5):
        """Retrieve with relevance scores for quality evaluation."""
        return self.vector_store.similarity_search_with_score(query, k=k)
```

#### 1.3 Create Embedding Pipeline

Create `app/ai/embedding_pipeline.py`:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings
from typing import List

class EmbeddingPipeline:
    """Pipeline for chunking and embedding documents."""

    def __init__(self):
        # Text splitter for general documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # Text splitter for MITRE ATT&CK (respects markdown structure)
        self.mitre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["## ", "\n\n", "\n"],
        )

    def chunk_document(
        self,
        content: str,
        metadata: dict,
        use_mitre_splitter: bool = False
    ) -> List[Document]:
        """
        Chunk a document into smaller pieces with metadata.

        Args:
            content: Document text
            metadata: Metadata dict (source, technique_id, severity, etc.)
            use_mitre_splitter: Use specialized splitter for MITRE docs

        Returns:
            List of Document objects ready for embedding
        """
        splitter = self.mitre_splitter if use_mitre_splitter else self.splitter
        chunks = splitter.split_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            )
            documents.append(doc)

        return documents
```

### Phase 2: Knowledge Base Population (Week 2-3)

#### 2.1 MITRE ATT&CK Ingestion Script

Create `app/scripts/ingest_mitre_attack.py`:

```python
import asyncio
import requests
from app.ai.rag_retriever import ThreatIntelligenceRetriever
from app.ai.embedding_pipeline import EmbeddingPipeline
from langchain_core.documents import Document

async def ingest_mitre_attack():
    """Download and ingest MITRE ATT&CK framework."""

    print("🔄 Downloading MITRE ATT&CK data...")
    response = requests.get(
        "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    )
    attack_data = response.json()

    # Initialize components
    retriever = ThreatIntelligenceRetriever()
    pipeline = EmbeddingPipeline()

    documents = []
    technique_count = 0

    for obj in attack_data["objects"]:
        if obj["type"] == "attack-pattern":
            technique_count += 1

            # Extract technique ID
            technique_id = next(
                (ref["external_id"] for ref in obj.get("external_references", [])
                 if ref.get("source_name") == "mitre-attack"),
                "Unknown"
            )

            # Build content
            content = f"""# {obj['name']} ({technique_id})

## Description
{obj.get('description', 'No description available')}

## Detection
{obj.get('x_mitre_detection', 'No detection guidance available')}
"""

            # Create metadata
            metadata = {
                "source": "mitre_attack",
                "technique_id": technique_id,
                "name": obj["name"],
                "tactic": obj.get("kill_chain_phases", [{}])[0].get("phase_name", "unknown"),
                "platform": ", ".join(obj.get("x_mitre_platforms", [])),
                "severity": _calculate_severity(obj),
                "last_updated": obj.get("modified", "2025-01-01"),
            }

            # Chunk document
            chunks = pipeline.chunk_document(content, metadata, use_mitre_splitter=True)
            documents.extend(chunks)

    # Ingest into vector database
    print(f"📦 Ingesting {len(documents)} chunks ({technique_count} techniques)...")
    retriever.vector_store.add_documents(documents)

    print(f"✅ Successfully ingested {len(documents)} MITRE ATT&CK chunks")

def _calculate_severity(technique_obj):
    """Simple heuristic for severity based on tactic."""
    tactics = [phase.get("phase_name", "") for phase in technique_obj.get("kill_chain_phases", [])]
    high_severity_tactics = ["privilege-escalation", "defense-evasion", "credential-access", "lateral-movement", "exfiltration", "impact"]

    if any(tactic in high_severity_tactics for tactic in tactics):
        return "high"
    return "medium"

if __name__ == "__main__":
    asyncio.run(ingest_mitre_attack())
```

Run ingestion:

```bash
cd aegis-ai-backend
python -m app.scripts.ingest_mitre_attack
```

#### 2.2 Verify Ingestion

Create `app/scripts/test_retrieval.py`:

```python
from app.ai.rag_retriever import ThreatIntelligenceRetriever

def test_retrieval():
    """Test RAG retrieval with sample queries."""

    retriever = ThreatIntelligenceRetriever()

    # Test queries
    queries = [
        "What are ransomware techniques?",
        "How do attackers achieve lateral movement?",
        "PowerShell execution techniques",
        "Phishing attack methods"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("=" * 60)

        results = retriever.retrieve(query, k=3)

        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.metadata.get('name', 'Unknown')} ({doc.metadata.get('technique_id', 'N/A')})")
            print(f"   Tactic: {doc.metadata.get('tactic', 'Unknown')}")
            print(f"   Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    test_retrieval()
```

### Phase 3: LangGraph Integration (Week 3-4)

#### 3.1 Add RAG Node to Graph

Modify `app/ai/langgraph_agent.py`:

```python
from app.ai.rag_retriever import ThreatIntelligenceRetriever

class LangGraphAgent:
    def __init__(self):
        # ... existing initialization ...

        # Initialize RAG retriever
        self.rag_retriever = ThreatIntelligenceRetriever()

    def _build_graph(self):
        workflow = StateGraph(CyberSecurityState)

        # Add nodes (including new RAG node)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("simple_response", self._simple_response_node)
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("threat_intel_retrieval", self._threat_intel_retrieval_node)  # NEW
        workflow.add_node("tool_selection", self._tool_selection_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("response", self._response_node)

        # Define edges
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges("classify", self._is_security_query, {
            "security": "planning",
            "general": "simple_response"
        })
        workflow.add_edge("simple_response", END)
        workflow.add_edge("planning", "threat_intel_retrieval")  # MODIFIED
        workflow.add_edge("threat_intel_retrieval", "tool_selection")  # NEW
        workflow.add_conditional_edges("tool_selection", self._should_use_tools, {
            "execute": "execute_tools",
            "skip": "analysis"
        })
        workflow.add_conditional_edges("execute_tools", self._continue_investigation, {
            "continue": "tool_selection",
            "analyze": "analysis"
        })
        workflow.add_edge("analysis", "response")
        workflow.add_edge("response", END)

        return workflow.compile()

    async def _threat_intel_retrieval_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        NEW NODE: Retrieval-Augmented Generation

        Queries threat intelligence knowledge base for relevant context.
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Build retrieval query
        retrieval_query = last_message
        filters = {}

        # Add severity filter for urgent queries
        if any(keyword in last_message.lower() for keyword in ["critical", "urgent", "breach"]):
            filters["severity"] = {"$in": ["high", "critical"]}

        # Retrieve relevant threat intelligence
        try:
            retrieved_docs = self.rag_retriever.retrieve(
                query=retrieval_query,
                k=5,
                filters=filters if filters else None
            )

            if retrieved_docs:
                # Format retrieved context
                context_text = "\n\n---\n\n".join([
                    f"**Technique**: {doc.metadata.get('name', 'Unknown')} ({doc.metadata.get('technique_id', 'N/A')})\n"
                    f"**Tactic**: {doc.metadata.get('tactic', 'Unknown')}\n"
                    f"**Content**: {doc.page_content[:300]}..."
                    for doc in retrieved_docs
                ])

                retrieval_message = f"""## 📚 Retrieved Threat Intelligence

Found {len(retrieved_docs)} relevant entries from MITRE ATT&CK:

{context_text}
"""

                logger.info(f"Retrieved {len(retrieved_docs)} threat intel documents")

                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=retrieval_message)],
                    "retrieved_context": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in retrieved_docs
                    ],
                    "knowledge_base_hits": len(retrieved_docs),
                }
            else:
                logger.info("No relevant threat intelligence found")
                return state

        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            # Gracefully degrade - continue without RAG
            return state
```

#### 3.2 Update State Schema

Add RAG fields to `CyberSecurityState`:

```python
class CyberSecurityState(TypedDict):
    # Existing fields
    messages: Annotated[Sequence[BaseMessage], add_messages]
    investigation_steps: list[str]
    iocs_found: list[dict]
    severity_level: str
    mitre_tactics: list[str]
    tools_used: list[str]
    tool_results: list[dict]
    pending_approval: dict | None
    final_response: str

    # NEW: RAG-specific fields
    retrieved_context: list[dict]  # Retrieved documents from knowledge base
    retrieval_scores: list[float]  # Relevance scores
    knowledge_base_hits: int       # Number of KB hits
```

#### 3.3 Enhance Analysis Node

Modify `_analysis_node` to use RAG context:

```python
async def _analysis_node(self, state: CyberSecurityState) -> Dict[str, Any]:
    """Node 4: Analysis - Enhanced with RAG context."""

    messages = state["messages"]
    tools_used = state.get("tools_used", [])
    retrieved_context = state.get("retrieved_context", [])

    # Skip if no tools used
    if not tools_used:
        logger.info("⏭️ Skipping analysis - no tools used")
        return state

    # Build context from RAG
    rag_context = ""
    if retrieved_context:
        techniques = [ctx["metadata"].get("technique_id") for ctx in retrieved_context if "technique_id" in ctx["metadata"]]
        rag_context = f"\n\n**MITRE Techniques in Knowledge Base:**\n{', '.join(set(techniques))}"

    analysis_prompt = f"""Analyze the investigation results and correlate with MITRE ATT&CK.

**IMPORTANT:** Start with "# 📊 Threat Analysis\n\n"

**Investigation Context:**
- Tools Used: {', '.join(tools_used)}
{rag_context}

**Your Analysis MUST:**
1. Reference specific MITRE techniques from retrieved context
2. Correlate tool results with known TTPs
3. Provide threat assessment based on documented patterns
4. Give actionable recommendations

Use ONLY data from tool results and retrieved threat intelligence."""

    response = await self.llm.ainvoke([
        SystemMessage(content=analysis_prompt),
        *messages
    ])

    # ... rest of analysis node ...
```

### Phase 4: Testing & Optimization (Week 4-5)

#### 4.1 Integration Test

Create `tests/test_rag_integration.py`:

```python
import pytest
from app.ai.langgraph_agent import LangGraphAgent

@pytest.mark.asyncio
async def test_rag_enhanced_investigation():
    """Test that RAG enhances investigation responses."""

    agent = LangGraphAgent()

    query = "Analyze suspicious PowerShell activity"

    # Run investigation
    result = await agent.stream_chat_completion(
        model="agent/cyber-analyst",
        messages=[{"role": "user", "content": query}],
        temperature=0.7
    )

    # Collect response
    response_text = ""
    async for chunk in result:
        if "choices" in chunk and chunk["choices"]:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            response_text += content

    # Assertions
    assert "T1059" in response_text, "Should reference PowerShell technique T1059"
    assert "MITRE" in response_text, "Should mention MITRE ATT&CK"
    assert len(response_text) > 500, "Response should be comprehensive"
```

Run tests:

```bash
pytest tests/test_rag_integration.py -v
```

#### 4.2 Performance Benchmark

Create `app/scripts/benchmark_rag.py`:

```python
import time
from app.ai.rag_retriever import ThreatIntelligenceRetriever

def benchmark_retrieval():
    """Benchmark RAG retrieval performance."""

    retriever = ThreatIntelligenceRetriever()

    queries = [
        "ransomware techniques",
        "lateral movement methods",
        "privilege escalation Windows",
        "phishing attack vectors",
        "command and control infrastructure"
    ]

    latencies = []

    for query in queries:
        start = time.time()
        results = retriever.retrieve(query, k=5)
        latency = (time.time() - start) * 1000  # Convert to ms

        latencies.append(latency)
        print(f"Query: {query[:30]}... | Latency: {latency:.2f}ms | Results: {len(results)}")

    print(f"\n📊 Performance Summary:")
    print(f"   Average Latency: {sum(latencies)/len(latencies):.2f}ms")
    print(f"   Max Latency: {max(latencies):.2f}ms")
    print(f"   Min Latency: {min(latencies):.2f}ms")

    # Target: <100ms average
    avg_latency = sum(latencies) / len(latencies)
    if avg_latency < 100:
        print(f"   ✅ Performance target met (<100ms)")
    else:
        print(f"   ⚠️ Performance needs optimization (target: <100ms, actual: {avg_latency:.2f}ms)")

if __name__ == "__main__":
    benchmark_retrieval()
```

---

## Technical Components

### 1. Vector Store (pgvector)

**Database Schema:**

```sql
CREATE TABLE threat_intelligence_embeddings (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,              -- Document text chunk
    metadata JSONB NOT NULL,            -- Structured metadata
    embedding vector(1536),             -- OpenAI embedding
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_ti_embedding ON threat_intelligence_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_ti_metadata ON threat_intelligence_embeddings
USING gin (metadata);
```

**Metadata Structure:**

```json
{
  "source": "mitre_attack",
  "technique_id": "T1059.001",
  "name": "PowerShell",
  "tactic": "execution",
  "platform": "Windows",
  "severity": "high",
  "last_updated": "2025-01-15"
}
```

### 2. Embedding Model

**OpenAI text-embedding-3-small:**
- Dimensions: 1536
- Cost: $0.02 per 1M tokens (~$0.0002 per document)
- Performance: Excellent for English cybersecurity text
- API: Via `langchain-openai` package

**Alternative: Voyage AI** (Anthropic partner):
- Better semantic understanding for technical domains
- Slightly higher cost but improved retrieval quality

### 3. Chunking Strategy

**General Documents:**
- Chunk size: 512 tokens
- Overlap: 100 tokens (20%)
- Separators: Paragraphs → Sentences → Words

**MITRE ATT&CK:**
- Chunk size: 800 tokens (techniques need more context)
- Overlap: 150 tokens
- Separators: Markdown headers (`##`) → Paragraphs → Sentences

**IOC Data:**
- Fixed-size chunks per IOC (256-512 tokens)
- Include all context (geolocation, reputation, historical analysis)

### 4. Retrieval Methods

**Semantic Search (Default):**
```python
results = vector_store.similarity_search(query, k=5)
```

**Filtered Search (Severity/Platform):**
```python
results = vector_store.similarity_search(
    query,
    k=5,
    filter={"severity": {"$in": ["high", "critical"]}}
)
```

**Hybrid Search (Advanced):**
```python
from langchain.retrievers import EnsembleRetriever

# Combine semantic + keyword (BM25)
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)
```

---

## Knowledge Base Management

### Adding New Documents

**Manual Upload (Admin API):**

```python
# app/routers/admin.py

@router.post("/admin/ingest-document")
async def ingest_document(
    file: UploadFile,
    source: str,
    severity: str = "medium",
    db: Session = Depends(get_db)
):
    """Upload and ingest a threat intelligence document."""

    from app.ai.rag_retriever import ThreatIntelligenceRetriever
    from app.ai.embedding_pipeline import EmbeddingPipeline

    # Read file
    content = await file.read()
    text = content.decode("utf-8")

    # Chunk and embed
    pipeline = EmbeddingPipeline()
    retriever = ThreatIntelligenceRetriever()

    metadata = {
        "source": source,
        "filename": file.filename,
        "severity": severity,
        "uploaded_at": datetime.now().isoformat()
    }

    documents = pipeline.chunk_document(text, metadata)
    retriever.vector_store.add_documents(documents)

    return {
        "status": "success",
        "chunks_ingested": len(documents),
        "filename": file.filename
    }
```

**Bulk Ingestion (CLI):**

```bash
# Ingest threat reports from directory
python -m app.scripts.ingest_threat_reports --dir ./data/threat_reports
```

### Updating Knowledge Base

**Re-embed Updated Documents:**

```python
async def update_mitre_attack():
    """Re-download and update MITRE ATT&CK data."""

    # Delete old MITRE entries
    vector_store.delete(filter={"source": "mitre_attack"})

    # Re-ingest fresh data
    await ingest_mitre_attack()
```

**Periodic Updates (Cron Job):**

```bash
# crontab entry: Update MITRE ATT&CK weekly
0 0 * * 0 cd /path/to/aegis-ai-backend && python -m app.scripts.ingest_mitre_attack
```

### Monitoring Knowledge Base

**Check Vector Count:**

```sql
SELECT COUNT(*) FROM threat_intelligence_embeddings;
SELECT source, COUNT(*) FROM threat_intelligence_embeddings GROUP BY source;
```

**Check Storage Size:**

```sql
SELECT pg_size_pretty(pg_total_relation_size('threat_intelligence_embeddings'));
```

---

## Performance Optimization

### 1. Index Tuning

**Optimize `lists` parameter for IVFFlat index:**

```sql
-- For 10,000 vectors: lists = 100
-- For 100,000 vectors: lists = 1000
-- For 1M+ vectors: lists = sqrt(total_vectors)

DROP INDEX idx_ti_embedding;
CREATE INDEX idx_ti_embedding ON threat_intelligence_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
```

**Trade-off:**
- Higher `lists` → Faster query, slower index build
- Lower `lists` → Slower query, faster index build

### 2. Connection Pooling

Update `app/database.py`:

```python
# For RAG queries (synchronous psycopg3)
import psycopg
from psycopg_pool import ConnectionPool

rag_pool = ConnectionPool(
    conninfo=settings.database_url.replace("asyncpg", "psycopg"),
    min_size=2,
    max_size=10
)
```

### 3. Caching Frequent Queries

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def retrieve_cached(query: str, k: int = 5):
    """Cache retrieval results for common queries."""
    return retriever.retrieve(query, k)
```

### 4. Batch Processing

For bulk ingestion, use batch embedding:

```python
# Embed multiple chunks in single API call
embeddings = OpenAIEmbeddings().embed_documents([
    chunk1.page_content,
    chunk2.page_content,
    # ... up to 100 chunks
])
```

---

## Troubleshooting

### Issue: Slow Retrieval (>200ms)

**Symptoms:** Vector searches take >200ms

**Solutions:**
1. Check index exists: `SELECT indexname FROM pg_indexes WHERE tablename = 'threat_intelligence_embeddings';`
2. Tune `lists` parameter (see Performance Optimization)
3. Verify PostgreSQL has enough memory: `SHOW shared_buffers;`
4. Consider upgrading PostgreSQL instance size

### Issue: Low Retrieval Quality

**Symptoms:** Retrieved documents not relevant to query

**Solutions:**
1. **Check embedding model:** Ensure using same model for indexing and querying
2. **Increase `k` value:** Retrieve more documents (e.g., k=10 instead of k=5)
3. **Add hybrid search:** Combine semantic + BM25 keyword matching
4. **Verify metadata:** Check documents have correct metadata (source, severity)
5. **Re-embed with better chunking:** Adjust chunk_size or overlap

### Issue: pgvector Extension Missing

**Error:** `extension "vector" does not exist`

**Solution (Supabase):**
```sql
-- Enable in SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

**Solution (Self-hosted PostgreSQL):**
```bash
# Install pgvector
cd /tmp
git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Issue: Out of Memory During Ingestion

**Error:** `MemoryError` or PostgreSQL OOM

**Solution:**
- Ingest in smaller batches (500 documents at a time)
- Use `batch_size` parameter in `add_documents()`
- Increase PostgreSQL `work_mem` setting

```python
# Batch ingestion
for i in range(0, len(documents), 500):
    batch = documents[i:i+500]
    vector_store.add_documents(batch)
    print(f"Ingested {i+500}/{len(documents)} documents")
```

---

## Future Enhancements

### Phase 5+: Advanced Features

#### 1. Hybrid Search (Semantic + Keyword)

Add BM25 keyword search alongside vector search:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine vector + BM25
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)
```

#### 2. Query Rewriting for Failed Retrieval

If initial retrieval returns no results, rewrite query:

```python
async def _rewrite_query_node(self, state):
    """Rewrite query if retrieval failed."""
    if state.get("knowledge_base_hits", 0) == 0:
        rewritten = await self.llm.ainvoke(
            "Rewrite this cybersecurity query for better retrieval: ..."
        )
        # Retry retrieval with new query
```

#### 3. Self-Reflective RAG

Validate retrieved context before using:

```python
# Grade retrieved documents
grader_prompt = "Is this document relevant to the query? Answer yes or no."
relevant_docs = [doc for doc in retrieved if llm.grade(doc) == "yes"]
```

#### 4. GraphRAG for Entity Relationships

Track relationships between threat actors, techniques, and infrastructure:

```sql
CREATE TABLE threat_entity_relationships (
    source_entity TEXT,
    relationship_type TEXT,  -- "uses", "targets", "associated_with"
    target_entity TEXT,
    confidence FLOAT
);
```

#### 5. Multi-Modal RAG

Ingest images (malware screenshots, network diagrams):

```python
from langchain.document_loaders import PDFPlumberLoader

# Extract images and text from PDFs
loader = PDFPlumberLoader("threat_report.pdf", extract_images=True)
```

#### 6. Continuous Learning

Improve embeddings based on user feedback:

```python
# Track query-result pairs
CREATE TABLE rag_feedback (
    query TEXT,
    retrieved_doc_id UUID,
    user_rating INT,  -- 1-5 stars
    created_at TIMESTAMP
);

# Periodically retrain embeddings with reinforcement learning
```

---

## Summary

### Quick Start Checklist

- [ ] Install dependencies (`langchain-postgres`, `psycopg`)
- [ ] Enable pgvector extension in PostgreSQL
- [ ] Create embeddings table via Alembic migration
- [ ] Create `app/ai/rag_retriever.py`
- [ ] Create `app/ai/embedding_pipeline.py`
- [ ] Run MITRE ATT&CK ingestion script
- [ ] Verify retrieval with test script
- [ ] Add `threat_intel_retrieval` node to LangGraph
- [ ] Update `CyberSecurityState` schema
- [ ] Test RAG-enhanced investigations
- [ ] Monitor performance (<100ms retrieval latency)

### Success Metrics

- ✅ **700+ MITRE techniques** ingested
- ✅ **<100ms average retrieval latency**
- ✅ **>0.7 average relevance score** (cosine similarity)
- ✅ **Investigation responses reference specific MITRE techniques**
- ✅ **30% improvement in threat detection accuracy**

### Resources

- **LangChain RAG Tutorial**: https://docs.langchain.com/docs/use-cases/rag
- **pgvector Documentation**: https://github.com/pgvector/pgvector
- **MITRE ATT&CK Data**: https://github.com/mitre/cti
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

---

## Contact & Support

For questions about RAG integration:
1. Check [BEGINNER_GUIDE.md](./BEGINNER_GUIDE.md) for system architecture
2. Check [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md) for visual flow
3. Raise an issue in the project repository

**Last Updated:** 2025-02-22
