import webbrowser
import re
import wikipedia

def can_handle(text: str) -> bool:
    return "search" in text.lower() or "tell me about" in text.lower()

def handle(text: str):
    match = re.search(r"(search|tell me about| what is| who is|find) (.+)", text, re.I)
    if not match:
        print("sorry, i couldn't understand the search query.")
        return
    
    query = match.group(2).strip()
    print(f"searching for: {query}")
    
    try:
        summary = wikipedia.summary(query, sentences=2)
        print(f"wikipedia says:", summary)
    except Exception as e:
        print("wikipedia error:", e)
        
        url = f"https://www.google.com/search?q={query.replace('', '+')}"
        print("opening google...")
        webbrowser.open(url)
        