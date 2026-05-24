# RAG Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve RAG response quality with better synthesis, a custom system prompt, off-topic query guarding, response caching, and streaming responses.

**Architecture:** All changes are in `app.py` — no new files needed. We add a system prompt to guide LLM behavior, switch response synthesis to `compact` mode, add an in-memory LRU cache for queries, add a topic guard that rejects off-topic questions before hitting the LLM, and add a `/query/stream` SSE endpoint for streaming responses.

**Tech Stack:** FastAPI, LlamaIndex (query engine config, streaming), `functools.lru_cache` or `dict`-based cache, SSE via `fastapi.responses.StreamingResponse`

---

### Task 1: Add Custom System Prompt and Compact Response Mode

**Files:**
- Modify: `app.py:52-55` (configuration section)
- Modify: `app.py:212` (query engine creation)

**Step 1: Add system prompt constant after the config section**

In `app.py`, after line 55 (`LLM_MODEL_NAME = ...`), add:

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Rahul Krishnan based on his resume, "
    "CV, and project documents. Be concise, professional, and friendly. "
    "If the provided context doesn't contain enough information to answer, say so honestly. "
    "Do not make up information that isn't in the documents."
)
```

**Step 2: Update query engine to use compact mode and system prompt**

Change line 212 from:
```python
query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k or 5)
```
to:
```python
from llama_index.core import PromptTemplate

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=top_k or 5,
    response_mode="compact",
    text_qa_template=PromptTemplate(
        SYSTEM_PROMPT + "\n\nContext:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
    ),
)
```

Note: Move the `from llama_index.core import PromptTemplate` to the top-level imports.

**Step 3: Test locally**

Run: `uv run python -c "from app import SYSTEM_PROMPT; print(SYSTEM_PROMPT)"`
Expected: Prints the system prompt string without errors.

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add system prompt and compact response mode for better RAG answers"
```

---

### Task 2: Add Off-Topic Query Guard

**Files:**
- Modify: `app.py` (add guard function, call it in `/query` endpoint)

**Step 1: Add off-topic detection function**

Add after the `check_daily_limit` function:

```python
OFF_TOPIC_KEYWORDS = [
    "weather", "stock", "sports", "news", "recipe", "cook",
    "movie", "music", "game", "crypto", "bitcoin",
]

def is_off_topic(query_text: str) -> bool:
    """Simple keyword-based check for obviously off-topic queries."""
    lower = query_text.lower()
    # Allow if it mentions Rahul or resume-related terms
    if any(term in lower for term in ["rahul", "resume", "cv", "skill", "experience", "project", "education", "work"]):
        return False
    # Block if it matches off-topic keywords
    if any(term in lower for term in OFF_TOPIC_KEYWORDS):
        return True
    return False
```

**Step 2: Call the guard in the `/query` endpoint**

In the `query()` function, after the `if not query_text:` check (line 210), add:

```python
if is_off_topic(query_text):
    return QueryResponse(
        query=query_text,
        answer="I can only answer questions about Rahul Krishnan's experience, skills, projects, and education. Please ask something related to his profile.",
    )
```

This returns a polite response without consuming a Gemini API call.

**Step 3: Test locally**

Run the app and test with:
- `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What is the weather today?"}'`
  Expected: Returns off-topic message, no Gemini call.
- `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What are Rahul skills?"}'`
  Expected: Normal RAG response.

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add off-topic query guard to save API calls"
```

---

### Task 3: Add Response Caching

**Files:**
- Modify: `app.py` (add cache dict and lookup logic in `/query`)

**Step 1: Add cache data structure**

After the daily limit variables (around line 72), add:

```python
# Query cache: stores {normalized_query: (answer, timestamp)}
_query_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 100
```

**Step 2: Add cache helper functions**

```python
def get_cached_response(query_text: str) -> str | None:
    """Return cached answer if exists and not expired."""
    key = query_text.strip().lower()
    if key in _query_cache:
        answer, ts = _query_cache[key]
        if time.time() - ts < CACHE_TTL:
            return answer
        del _query_cache[key]
    return None


def set_cache(query_text: str, answer: str) -> None:
    """Cache a query-answer pair. Evict oldest if full."""
    key = query_text.strip().lower()
    if len(_query_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_query_cache, key=lambda k: _query_cache[k][1])
        del _query_cache[oldest_key]
    _query_cache[key] = (answer, time.time())
```

**Step 3: Use cache in the `/query` endpoint**

In the `query()` function, after the off-topic check, before the `try:` block that creates the query engine, add:

```python
cached = get_cached_response(query_text)
if cached is not None:
    return QueryResponse(query=query_text, answer=cached)
```

After `response = query_engine.query(query_text)` and `_daily_count += 1`, add:

```python
set_cache(query_text, str(response))
```

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add in-memory query cache (1h TTL, 100 entries max)"
```

---

### Task 4: Add Streaming Response Endpoint

**Files:**
- Modify: `app.py` (add `/query/stream` SSE endpoint)

**Step 1: Add SSE streaming endpoint**

Add after the existing `/query` endpoint:

```python
from fastapi.responses import StreamingResponse

@app.post("/query/stream")
async def query_stream(request: QueryRequest, req: Request):
    global _daily_count
    client_ip = req.headers.get("X-Forwarded-For", req.client.host or "").split(",")[0].strip()
    check_rate_limit(client_ip)
    check_daily_limit()
    query_text = request.query
    top_k = request.top_k
    if index is None:
        raise HTTPException(status_code=500, detail="Index not built yet.")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")
    if is_off_topic(query_text):
        async def off_topic_gen():
            msg = "I can only answer questions about Rahul Krishnan's experience, skills, projects, and education."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(off_topic_gen(), media_type="text/event-stream")

    cached = get_cached_response(query_text)
    if cached is not None:
        async def cached_gen():
            yield f"data: {json.dumps({'token': cached})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cached_gen(), media_type="text/event-stream")

    try:
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=top_k or 5,
            response_mode="compact",
            streaming=True,
            text_qa_template=PromptTemplate(
                SYSTEM_PROMPT + "\n\nContext:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
            ),
        )
        streaming_response = query_engine.query(query_text)
        _daily_count += 1

        async def token_generator():
            full_response = ""
            for token in streaming_response.response_gen:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
            set_cache(query_text, full_response)

        return StreamingResponse(token_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Move `from fastapi.responses import StreamingResponse` to top-level imports.

**Step 2: Test locally**

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What are Rahul skills?"}'
```
Expected: SSE events streaming token-by-token, ending with `[DONE]`.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add /query/stream SSE endpoint for streaming responses"
```

---

### Task 5: Build and Push Docker Image

**Step 1: Build Docker image**

```bash
docker build -f DOCKERFILE -t rahulk98/resume-rag-system:v4 .
```

**Step 2: Push to Docker Hub**

```bash
docker push rahulk98/resume-rag-system:v4
```

**Step 3: Commit all remaining changes**

```bash
git add app.py requirements.txt
git commit -m "feat: RAG improvements - system prompt, caching, off-topic guard, streaming"
```

**Step 4: Provide deploy command**

```bash
gcloud run deploy resume-rag-system \
  --image=docker.io/rahulk98/resume-rag-system:v4 \
  --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest,SHEETS_JSON_KEY=sheet-json-key:latest \
  --region=europe-west1
```
