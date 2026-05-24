from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional
import os
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from contextlib import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import gspread
from google.oauth2.service_account import Credentials
import httpx

load_dotenv()


SHEET_ID = "1P3dd_kTPb9NFqn6SZ1kIpaJ0R8_9Cz6fJ0akZhzym5k"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models_and_index()
    init_sheets_client()
    yield


app = FastAPI(title="RAG Resume API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rahul-krishnan.is-a.dev",
        "https://rahulk98.github.io",
        "http://localhost:8001",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
    max_age=86400,
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment. RAG queries will not work.")

# configuration
INDEX_PERSIST_DIR = "./index"
EMBED_MODEL_NAME = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Rahul Krishnan based on his resume, "
    "CV, and project documents. Be concise, professional, and friendly. "
    "If the provided context doesn't contain enough information to answer, say so honestly. "
    "Do not make up information that isn't in the documents."
)


# global objects
index = None
embed_model = None
llm = None
sheets_worksheet = None

# Rate limiting: max 10 requests per IP per minute
RATE_LIMIT = 10
RATE_WINDOW = 60  # seconds
_rate_store: dict[str, list[float]] = defaultdict(list)

# Daily Gemini API call limit
DAILY_QUERY_LIMIT = 30
_daily_count = 0
_daily_reset_date = datetime.now(timezone.utc).date()

# Query cache: {normalized_query: (answer, timestamp)}
_query_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 100



def check_rate_limit(ip: str) -> None:
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    _rate_store[ip].append(now)


def check_daily_limit() -> None:
    global _daily_count, _daily_reset_date
    today = datetime.now(timezone.utc).date()
    if today != _daily_reset_date:
        _daily_count = 0
        _daily_reset_date = today
    if _daily_count >= DAILY_QUERY_LIMIT:
        raise HTTPException(status_code=429, detail="Daily query limit reached. Please try again tomorrow.")


def get_cached_response(query_text: str) -> str | None:
    key = query_text.strip().lower()
    if key in _query_cache:
        answer, ts = _query_cache[key]
        if time.time() - ts < CACHE_TTL:
            return answer
        del _query_cache[key]
    return None


def set_cache(query_text: str, answer: str) -> None:
    key = query_text.strip().lower()
    if len(_query_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_query_cache, key=lambda k: _query_cache[k][1])
        del _query_cache[oldest_key]
    _query_cache[key] = (answer, time.time())


# Response Models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str


def init_models_and_index():
    """
    Initialize embedding model, llm, and load index from disk (if exists).
    Called once at startup.
    """
    global embed_model, llm, index
    if not GOOGLE_API_KEY:
        print("Skipping model initialization — no API key.")
        return
    embed_model = GoogleGenAIEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=100,
        api_key=GOOGLE_API_KEY,
    )

    try:
        llm = GoogleGenAI(
            model=LLM_MODEL_NAME,
            api_key=GOOGLE_API_KEY,
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        llm = None

    if os.path.isdir(INDEX_PERSIST_DIR) and os.listdir(INDEX_PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=INDEX_PERSIST_DIR
            )
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            print("Index loaded from disk.")
        except Exception as e:
            print(f"Failed to load index from disk: {e}")
            index = None
    else:
        print("No index on disk yet. Call /reindex to build it.")
        index = None


def init_sheets_client():
    """Initialize Google Sheets client from SHEETS_JSON_KEY env var."""
    global sheets_worksheet
    sheets_json = os.getenv("SHEETS_JSON_KEY")
    if not sheets_json:
        print("SHEETS_JSON_KEY not set; skipping Sheets initialization.")
        sheets_worksheet = None
        return
    try:
        creds_info = json.loads(sheets_json)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sheets_worksheet = gc.open_by_key(SHEET_ID).sheet1
        print("Google Sheets client initialized.")
    except Exception as e:
        print(f"Failed to initialize Sheets client: {e}")
        sheets_worksheet = None


def get_country_from_ip(ip: str) -> str:
    try:
        resp = httpx.get(f"http://ip-api.com/json/{ip}?fields=country", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("country", "Unknown")
    except Exception:
        pass
    return "Unknown"


def log_to_sheets(query_text: str, response_text: str, client_ip: str) -> None:
    try:
        if sheets_worksheet is None:
            return
        country = get_country_from_ip(client_ip)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        sheets_worksheet.append_row(
            [timestamp, query_text, response_text, country],
            value_input_option="USER_ENTERED",
        )
    except Exception as e:
        print(f"Sheets log insert failed: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=index is not None)


def _build_query_engine(top_k: int = 5, streaming: bool = False):
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        response_mode="compact",
        streaming=streaming,
        text_qa_template=PromptTemplate(
            SYSTEM_PROMPT + "\n\nContext:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
        ),
    )


def _validate_query(request: QueryRequest, req: Request) -> tuple[str, str, int]:
    """Common validation for query endpoints. Returns (query_text, client_ip, top_k)."""
    client_ip = req.headers.get("X-Forwarded-For", req.client.host or "").split(",")[0].strip()
    check_rate_limit(client_ip)
    check_daily_limit()
    query_text = request.query
    if index is None:
        raise HTTPException(status_code=500, detail="Index not built yet. Call /reindex first.")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")
    return query_text, client_ip, request.top_k or 5


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, background_tasks: BackgroundTasks, req: Request):
    global _daily_count
    query_text, client_ip, top_k = _validate_query(request, req)
    cached = get_cached_response(query_text)
    if cached is not None:
        return QueryResponse(query=query_text, answer=cached)
    try:
        query_engine = _build_query_engine(top_k)
        response = query_engine.query(query_text)
        _daily_count += 1
        answer = str(response)
        set_cache(query_text, answer)
        background_tasks.add_task(log_to_sheets, query_text, answer, client_ip)
        return QueryResponse(query=query_text, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def query_stream(request: QueryRequest, background_tasks: BackgroundTasks, req: Request):
    global _daily_count
    query_text, client_ip, top_k = _validate_query(request, req)

    cached = get_cached_response(query_text)
    if cached is not None:
        def cached_gen():
            yield f"data: {json.dumps({'token': cached})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cached_gen(), media_type="text/event-stream")

    try:
        query_engine = _build_query_engine(top_k, streaming=True)
        streaming_response = query_engine.query(query_text)
        _daily_count += 1

        def token_generator():
            full_response = ""
            for token in streaming_response.response_gen:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
            set_cache(query_text, full_response)
            background_tasks.add_task(log_to_sheets, query_text, full_response, client_ip)

        return StreamingResponse(token_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    except Exception as e:
        print(f"Error starting server: {e}")
        raise e
