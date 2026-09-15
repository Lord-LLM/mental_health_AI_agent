import os
import time
import threading
import uuid
import fitz
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from mistralai.client import Mistral

from crisis import contains_crisis_keywords, SAFETY_MESSAGE
from logger import log_chat

# Load env vars
load_dotenv()

# Therapist system prompt
THERAPIST_PREFIX = "You are a compassionate therapist who speaks in a calm and understanding tone."

MODEL_NAME = "mistral-small-2603"
BOOKS_FOLDER = "books"
CHUNK_SIZE = 2048
TOP_K = 2

# Free-tier rate limit is 1 request/second, shared across every session of
# this deployed app. Serialize calls so concurrent users don't trip it.
MIN_CALL_INTERVAL = 1.1  # small buffer above the 1 RPS ceiling
MAX_RETRIES = 3
BASE_BACKOFF = 2.0  # seconds

_call_lock = threading.Lock()
_last_call_time = 0.0

# Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY environment variable is not set.")
client = Mistral(api_key=api_key)

# Embedding model, used for both indexing and query-time search
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_folder(folder_path: str = BOOKS_FOLDER) -> str:
    """Extract text from every PDF in folder_path. Returns an empty string
    if the folder doesn't exist or contains no readable PDFs."""
    if not os.path.isdir(folder_path):
        print(f"Books folder not found: {folder_path}. Skipping PDF context.")
        return ""

    all_text = ""
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, filename)
        print(f"Reading: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            pdf_text = ""
            for page_num in range(doc.page_count):
                page_text = doc.load_page(page_num).get_text()
                if page_text.strip():
                    pdf_text += page_text + "\n"

            if not pdf_text:
                print(f"No extractable text in: {filename}")

            all_text += pdf_text
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return all_text


def build_index(chunks):
    """Build a FAISS index from text chunks. Returns (index, dimension) or
    (None, None) if there are no chunks to index."""
    if not chunks:
        return None, None

    embeddings = embedder.encode(
        chunks, show_progress_bar=False, convert_to_numpy=True
    ).astype("float32")
    dimension = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dimension)
    idx.add(embeddings)
    return idx, dimension


# Load and index PDF context once at startup
_text = extract_text_from_folder()
chunks = [_text[i:i + CHUNK_SIZE] for i in range(0, len(_text), CHUNK_SIZE)]
index, _dim = build_index(chunks)


def _wait_for_rate_limit():
    """Block until at least MIN_CALL_INTERVAL seconds have passed since the
    last API call, across all threads/sessions sharing this process."""
    global _last_call_time
    with _call_lock:
        now = time.monotonic()
        elapsed = now - _last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)
        _last_call_time = time.monotonic()


def _is_rate_limit_error(err: Exception) -> bool:
    text = str(err).lower()
    return "429" in text or "rate_limited" in text or "rate limit" in text


def run_mistral(user_message: str, model_name: str = MODEL_NAME) -> str:
    messages = [{"role": "user", "content": user_message}]

    for attempt in range(MAX_RETRIES + 1):
        _wait_for_rate_limit()
        try:
            response = client.chat.complete(model=model_name, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF * (2 ** attempt)
                print(f"Rate limited, retrying in {backoff:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
                continue
            raise

    raise RuntimeError("Mistral API rate limit exceeded after retries.")


def _build_prompt(question: str) -> str:
    if index is None:
        return f"""{THERAPIST_PREFIX}

Query: {question}
Answer:"""

    question_embedding = embedder.encode([question]).astype("float32")
    _, matched_indices = index.search(question_embedding, k=min(TOP_K, len(chunks)))
    retrieved_chunks = [chunks[i] for i in matched_indices[0]]
    context = "\n".join(retrieved_chunks)

    return f"""{THERAPIST_PREFIX}
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:"""


def run_chat(question: str) -> str:
    session_id = str(uuid.uuid4())
    is_crisis = contains_crisis_keywords(question)

    if is_crisis:
        # Respond immediately with safety resources; skip the model call.
        answer = SAFETY_MESSAGE
        log_chat(session_id=session_id, query=question, response=answer, is_crisis=True)
        return answer

    prompt = _build_prompt(question)
    answer = run_mistral(prompt)

    log_chat(session_id=session_id, query=question, response=answer, is_crisis=False)
    return answer
