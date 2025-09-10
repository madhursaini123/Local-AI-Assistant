import webbrowser

def can_handle(text: str) -> bool:
    return "open youtube" in text.lower() or "launch youtube" in text.lower()

def handle(text: str):
    # Optional: extract query
    parts = text.lower().split("open youtube")
    if len(parts) > 1 and parts[1].strip():
        query = parts[1].strip()
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
    else:
        webbrowser.open("https://www.youtube.com")
