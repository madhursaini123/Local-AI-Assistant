import re
from utsiskills.rag_engine2 import search_memory
from utsiskills.utils import speak

def can_handle(text: str) -> bool:
    return "recall" in text.lower() or "remember" in text.lower()

def handle(text: str):
    keyword = text.replace("recall", "").replace("remember", "").strip()
    results = search_memory(keyword or "sherlock")
    print("📚 Recalled:\n", results)
    speak(results)
