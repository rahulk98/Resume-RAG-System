from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import StorageContext, load_index_from_storage
from contextlib import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models_and_index()
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
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

# configuration
INDEX_PERSIST_DIR = "./index"
EMBED_MODEL_NAME = "text-embedding-004"
LLM_MODEL_NAME = "gemini-2.5-flash"






# global objects
index = None
embed_model = None
llm = None


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
    embed_model = GoogleGenAIEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=100,
        api_key=GOOGLE_API_KEY,
    )

    llm = GoogleGenAI(
        model=LLM_MODEL_NAME,
        api_key=GOOGLE_API_KEY,
    )

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


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=index is not None)

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    POST /query
    body: { "query": "some question", "top_k": 5 (optional) }
    """
    global index, llm, embed_model
    query_text = request.query
    top_k = request.top_k
    if index is None:
        raise HTTPException(status_code=500, detail="Index not built yet. Call /reindex first.")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")
    try:
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k or 5)
        response = query_engine.query(query_text)
        return QueryResponse(query=query_text, answer=str(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    except Exception as e:
        print(f"Error starting server: {e}")
        raise e