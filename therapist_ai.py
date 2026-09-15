import os
import uuid
import fitz
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from transformers import AutoTokenizer

from crisis import contains_crisis_keywords, SAFETY_MESSAGE
from logger import log_chat

# Load env vars
load_dotenv()

# Therapist system prompt
THERAPIST_PREFIX = "You are a compassionate therapist who speaks in a calm and understanding tone."

# Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# Tokenizer for future use (optional)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Extract text from PDFs in /books
def extract_text_from_folder(folder_path="books"):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Reading: {pdf_path}")
            try:
                doc = fitz.open(pdf_path)
                num_pages = doc.page_count
                pdf_text = ""
                
                for page_num in range(num_pages):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text.strip():  # Skip empty pages
                        pdf_text += page_text + "\n"
                
                if not pdf_text:
                    print(f"⚠️ No extractable text in: {filename}")

                all_text += pdf_text

            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    return all_text

# Load PDF text
text = extract_text_from_folder()

# Chunking
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Embeddings and FAISS index setup
model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True).astype('float32')
dimension = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(text_embeddings)

# Mistral LLM call
def run_mistral(user_message, model_name="mistral-large-latest"):
    messages = [{"role": "user", "content": user_message}]
    response = client.chat.complete(model=model_name, messages=messages)
    return response.choices[0].message.content

# Main chatbot logic
def run_chat(question: str) -> str:
    question_embedding = model.encode([question]).astype('float32')
    _, I = index.search(question_embedding, k=2)
    retrieved_chunk = [chunks[i] for i in I[0]]

    context = "\n".join(retrieved_chunk)
    prompt = f"""{THERAPIST_PREFIX}
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:"""

    answer = run_mistral(prompt)

    is_crisis = contains_crisis_keywords(question)
    if is_crisis:
        answer = SAFETY_MESSAGE + "\n\n" + answer

    session_id = str(uuid.uuid4())
    log_chat(session_id=session_id, query=question, response=answer, is_crisis=is_crisis)

    return answer
