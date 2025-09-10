import re
import webbrowser

def can_handle(text: str) -> bool:
    return "play news" in text.lower() or "open news channel" in text.lower()

def handle(text: str):
    print("🔴 Opening YouTube live news…")
    webbrowser.open("https://www.youtube.com/results?search_query=live+news+India")
