from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")


def query_index(query_text: str) -> str:
    # Setup the gemini model for embeddings
    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004", embed_batch_size=100, api_key=GOOGLE_API_KEY
    )
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./index")
    # load index
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    # Initliase the LLM
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,  # uses GOOGLE_API_KEY env var by default
    )

    # create a query engine and query
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query_text)
    print(response)
    return response
    
    
if __name__ == "__main__":
    query_index("Tell me about Rahul in 3 sentences.")