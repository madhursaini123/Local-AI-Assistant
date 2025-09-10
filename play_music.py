import webbrowser
import re

def can_handle(text: str) -> bool:
    return bool(re.search(r"\b(play|start)\b.*\bmusic\b", text, re.IGNORECASE))

def handle(text: str):
    query = re.sub(r"(play|start)\s+(some\s+)?music", "", text, flags=re.IGNORECASE).strip()
    if not query:
        query = "top trending music"
        
    print(f"searching youtube for: {query}")
    webbrowser.open(f"https://www.youtube.com/results?search_query={query}")