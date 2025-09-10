# rag_engine.py
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import faiss
import numpy as np
import json, os
import datetime
import re


MEMORY_PATH = Path("memory/rag_memory.json")
INDEX_PATH = "memory/faiss.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

os.makedirs("memory", exist_ok=True)
model = SentenceTransformer(EMBEDDING_MODEL)

# FAISS index: 384 dim for MiniLM
index = faiss.IndexFlatL2(384)
memory_data = []

# Load memory if exists
if os.path.exists(MEMORY_PATH):
    memory_data = json.load(open(MEMORY_PATH, "r", encoding="utf-8"))
    if memory_data:
        vectors = model.encode([m["text"] for m in memory_data])
        index.add(np.array(vectors).astype("float32"))
        

def add_memory(text, reply):
    combined = f"User: {text}\nUtsi: {reply}"
    embedding = model.encode([combined])[0]
    index.add(np.array([embedding]).astype("float32"))
    memory_data.append({"text": combined})
    json.dump(memory_data, open(MEMORY_PATH, "w", encoding="utf-8"), indent=2)

def search_memory(query, top_k=3):
    if not memory_data:
        return "I don't have any deep memories yet."

    query_vec = model.encode([query])[0]
    D, I = index.search(np.array([query_vec]).astype("float32"), top_k)
    results = [memory_data[i]["text"] for i in I[0] if i < len(memory_data)]
    return "\n\n".join(results)

def extract_tags(text):
    # Simple tag extractor using capitalized keywords and nouns
    return list(set(re.findall(r"\b[A-Z][a-z]+", text)))

def save_fact(fact_text):
    tags = extract_tags(fact_text)
    fact = {
        "text": fact_text,
        "tags": tags,
        "timestamp": str(datetime.datetime.now())
    }
    memory_data.append(fact)
    MEMORY_PATH.write_text(json.dumps(memory_data, indent=2, ensure_ascii=False), encoding="utf-8")

def recall(query):
    matches = [fact for fact in memory_data if query.lower() in fact['text'].lower()]
    if not matches:
        return "I don't remember anything related to that."

    # Limit results
    recent = matches[-3:]

    return "\n\n".join(
    f"🧠 {m['text']} (saved on {m.get('timestamp', 'unknown date')})"
    for m in recent
  )

    query_emb = model.encode(query, convert_to_tensor=True)
    best = None
    best_score = 0.4  # similarity threshold

    for fact in memory_data:
        fact_emb = model.encode(fact["text"], convert_to_tensor=True)
        sim = float(util.cos_sim(query_emb, fact_emb))
        if sim > best_score:
            best = fact
            best_score = sim

    if best:
        return f"I remember: {best['text']} (saved on {best['timestamp']})"
    else:
        return "I don't remember anything relevant."

# ------------------------------
# Voice Skill Interface
# ------------------------------
def can_handle(text: str) -> bool:
    return any(phrase in text.lower() for phrase in [
        "remember this", "save this", "i want to tell you",
        "recall", "what do you know", "do you remember", "show memory", "facts tagged"
    ])

def handle(text: str):
    from utsiskills.utils import speak

    if any(x in text.lower() for x in ["remember this", "save this", "i want to tell you"]):
        fact = re.sub(r"^(remember this|save this|i want to tell you)\b", "", text, flags=re.I).strip()
        save_fact(fact)
        print(f"💾 Saved: {fact}")
        speak("Got it. I’ve saved that.")
    else:
        query = re.sub(r"^(recall|what do you know about|do you remember|show memory|facts tagged)\b", "", text, flags=re.I).strip()
        reply = recall(query)
        print(f"📚 Recalled:\n{reply}")
        speak(reply)
