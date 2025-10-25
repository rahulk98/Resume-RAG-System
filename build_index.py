from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")


def build_index():
    # Read all the relevant documents in docs folder
    documents = SimpleDirectoryReader("./docs").load_data()

    # Setup the gemini model for embeddings
    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004", embed_batch_size=100, api_key=GOOGLE_API_KEY
    )
    # Build the vector store index from the documents in docs folder
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Store the index for reuse
    index.storage_context.persist(persist_dir="./index")

    # Initliase the LLM
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,  # uses GOOGLE_API_KEY env var by default
    )

    # create a query engine and query
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("What are Rahul's skills?")
    print(response)


if __name__ == "__main__":
    build_index()
